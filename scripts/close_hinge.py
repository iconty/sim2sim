import time
import numpy as np
import math
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as R
from collections import deque

import pygame

import threading

import torch

from plot import plot

from isaaclab.utils.buffers import CircularBuffer, DelayBuffer

default_joint_angles = {
            'FL_hip_joint': -0.0,   # [rad] LF_HAA
            'RL_hip_joint': -0.0,   # [rad] LH_HAA
            'FR_hip_joint': 0.0 ,  # [rad] RF_HAA
            'RR_hip_joint': 0.0,   # [rad] RH_HAA

            'FL_thigh_joint': 0.9,   # [rad] LF_HFE 
            'RL_thigh_joint': 0.9,   # [rad] LH_HFE
            'FR_thigh_joint': 0.9,   # [rad] RF_HFE
            'RR_thigh_joint': 0.9,   # [rad] RH_HFE

            'FL_calf_joint': -1.55,   # [rad] LF_KFE
            'RL_calf_joint': -1.55,   # [rad] LH_KFE
            'FR_calf_joint': -1.55,   # [rad] RF_KFE
            'RR_calf_joint': -1.55,   # [rad] RH_KFE
        }


# 设置打印选项，避免科学计数法，并保留小数点后 3 位
np.set_printoptions(suppress=True, precision=3)

def to_torch(x, dtype=torch.float, device='cuda:0', requires_grad=False):
    return torch.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)

device = 'cuda:0'

paused = False
one_step = False
action_at = False
done = False
disable_mode = False
disturbance = False
hangup = False
reset = False

# joy define
#  if (js_type == "xbox")
# {
#     js_id_.axis["LX"] = 0; // Left stick axis left right [-1, 1]
#     js_id_.axis["LY"] = 1; // Left stick axis up donw [-1, 1]
#     js_id_.axis["RX"] = 3; // Right stick axis left right [-1, 1]
#     js_id_.axis["RY"] = 4; // Right stick axis up donw [-1, 1]
#     js_id_.axis["LT"] = 2; // Left trigger [-1, 1]
#     js_id_.axis["RT"] = 5; // Right trigger[-1, 1]
#     js_id_.axis["DX"] = 6; // Directional pad x none
#     js_id_.axis["DY"] = 7; // Directional pad y none 
    
#     js_id_.button["X"] = 2;
#     js_id_.button["Y"] = 3;
#     js_id_.button["B"] = 1;
#     js_id_.button["A"] = 0;
#     js_id_.button["LB"] = 4;
#     js_id_.button["RB"] = 5;
#     js_id_.button["SELECT"] = 6;
#     js_id_.button["START"] = 7;
#     js_id_.button["home"] = 8;
#     js_id_.button["return"] = 11;
# }
button = {
    'X': 2,
    'Y': 3,
    'B': 1,
    'A': 0,
    'LB': 4,
    'RB': 5,
    'SELECT': 6,
    'START': 7,
    'home': 8,
    'return': 11
}

class joy_data:
    Axis = [0] * 8
    button = 0
    button_last = 0

class cmd:
    vx = 0.0
    vy = 0.0
    dyaw = 0.0

def joy_init():
    global done
    while (done !=True):
        pygame.init()
        pygame.joystick.init()

        joystick_count = pygame.joystick.get_count()
        if joystick_count > 0:
            joystick = pygame.joystick.Joystick(0)
            joystick.init()
            print("Joystick Name: {}".format(joystick.get_name()))
            break
        else:
            time.sleep(0.5)

def key_callback(keycode):

    if keycode == 259:
        global reset
        reset = not reset

    # 空格键,暂停仿真
    if chr(keycode) == ' ':
        global paused
        paused = not paused

    if keycode == 320:
        global action_at
        action_at = not action_at

    if keycode == 321:
        global disable_mode
        disable_mode = not disable_mode

    if keycode == 322:
        global disturbance
        disturbance = not disturbance

    if keycode == 323:
        global hangup
        hangup = not hangup

    if keycode == 334:
        global one_step
        one_step = not one_step

    # 方向键上,前进速度
    if keycode == 265:
        cmd.vx = 1.0
    # 方向键下,后退速度
    elif keycode == 264:
        cmd.vx = -1.0
    else:
        cmd.vx = 0.0

    # 方向键左,左平移速度
    if keycode == 263:
        cmd.dyaw = 1.0  
    elif keycode == 262:
        cmd.dyaw = -1.0
    else:
        cmd.dyaw = 0.0

class ObsBuffer:
    def __init__(self, max_length: int, dt: float):
        self.buffer = deque(maxlen=max_length)
        self.dt = dt

    def add(self, value) -> None:
        self.buffer.append(value)
    
    def get_delayed_value(self, delay_time: float):

        # 计算需要回退的步数
        n_steps = int(round(delay_time / self.dt))

        # 如果缓存不足，返回最新一帧
        if n_steps >= len(self.buffer):
            return self.buffer[-1]

        # 正常返回指定延迟步的值
        return self.buffer[-n_steps - 1]

class StepFrequencyTracker:
    def __init__(self, window_size: int = 200, dt: float = 0.005):
        """
        初始化步频跟踪器
        Args:
            window_size: 保存多少帧的历史（如 100 帧）
            dt: 时间步长（每帧的间隔，单位：秒）
        """
        self.dt = dt
        self.buffer = deque(maxlen=window_size)
        self.T = window_size

    def update(self, foot_contact: np.ndarray):
        """
        添加当前一帧的 foot_contact 状态
        Args:
            foot_contact: shape=(4,), float 或 bool 数组
        """
        assert foot_contact.shape == (4,)
        self.buffer.append(foot_contact.astype(np.uint8))

    def get_step_frequency(self) -> np.ndarray:
        if len(self.buffer) < 2:
            return np.zeros(4)

        buffer_array = np.stack(self.buffer, axis=0).astype(np.float32)  # ✅ 确保类型
        transitions = np.abs(np.diff(buffer_array, axis=0))  # (T-1, 4)
        transition_count = np.sum(transitions, axis=0)

        window_time = self.dt * (len(self.buffer) - 1)
        step_freq = (transition_count / 2.0) / window_time  # Hz

        return step_freq

@torch.jit.script
def quat_rotate_inverse(q, v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * \
        torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
            shape[0], 3, 1)).squeeze(-1) * 2.0
    return a - b + c

def quaternion_to_euler_array(quat):
    # Ensure quaternion is in the correct format [x, y, z, w]
    x, y, z, w = quat
    
    # Roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)
    
    # Pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)
    
    # Yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)
    
    # Returns roll, pitch, yaw in a NumPy array in radians
    return np.array([roll_x, pitch_y, yaw_z])

def get_foot_contact(data, model):
    # 存储每个脚的总接触力模长
    foot_forces = np.zeros(4)
    foot_contact = np.zeros(4)
    foot_geom_names = [
        'FL_foot_fixed',
        'FR_foot_fixed',
        'RL_foot_fixed',
        'RR_foot_fixed'
    ]
    foot_idx = {
        'FL_foot_fixed': 0,
        'FR_foot_fixed': 1,
        'RL_foot_fixed': 2,
        'RR_foot_fixed': 3
    }

    # 每个 contact 可计算 6 维力（3 正交方向 + 3 摩擦方向）
    force = np.zeros(6)
    
    ground_geom_names = ['floor']
    for i in range(data.ncon):
        contact = data.contact[i]
        mujoco.mj_contactForce(model, data, i, force)
        g1 = model.geom(contact.geom1).name
        g2 = model.geom(contact.geom2).name
        
        # 检测脚与地面的接触
        if (g1 in ground_geom_names and g2 in foot_geom_names) or (g2 in ground_geom_names and g1 in foot_geom_names):
            foot_name = g1 if g2 in ground_geom_names else g2   
            # 力的模长（L2范数），只用前3维（接触方向 + 摩擦合力）
            force_norm = np.linalg.norm(force[0:3])
            foot_forces[foot_idx[foot_name]] = force_norm
            foot_contact[foot_idx[foot_name]] = 1.0
    return foot_contact, foot_forces

def get_obs(data, model):
    '''Extracts an observation from the mujoco data structure
    '''
    q = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)
    quat = data.sensor('orientation').data[[1, 2, 3, 0]].astype(np.double)
    r = R.from_quat(quat)
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)  # In the base frame
    omega = data.sensor('angular-velocity').data.astype(np.double)
    gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)

    return (q, dq, quat, v, omega, gvec)


def pd_control(target_q, q, kp, target_dq, dq, kd):
    '''Calculates torques from position commands
    '''
    return (target_q - q) * kp + (target_dq - dq) * kd

# 常量定义
LEAD = 0.016
max_pos = 73.0 * (0.008 / LEAD)

J2= (LEAD / 2.0 / np.pi)
J2_ = (2.0 * np.pi / LEAD) 

# 函数定义
def RAD2LEN(x):
    return x / 2.0 / np.pi * LEAD

def DEG2RAD(deg):
    return deg * np.pi / 180.0

def user_sqrt(value):
    return math.sqrt(value)

def calc_jacobian(motor_pos):
    SCREW_TATOL_LENGTH = 0.303
    LEADSCREW_CALF_ANGLE = 156.17
    CERTAIN_ANGLE = -0.81
    SQUARE_L1_L2_SUM = 0.0720502569
    DOUBLE_L1_L2_MUL = 0.0313956

    motor_pos = np.clip(motor_pos, 0.0, max_pos)
    # 计算螺杆长度
    Screw_Len = SCREW_TATOL_LENGTH - RAD2LEN(motor_pos)
    # 计算内角
    Inner_Angle = np.arccos(
        (SQUARE_L1_L2_SUM - Screw_Len ** 2) / DOUBLE_L1_L2_MUL
    )
    # 计算膝关节角度（弧度）
    KneeAngle_rad = (
        math.radians(LEADSCREW_CALF_ANGLE + CERTAIN_ANGLE) 
        - Inner_Angle 
        - np.pi
    )
    # 计算雅可比矩阵元素J1
    J1 = (DOUBLE_L1_L2_MUL / 2.0) * np.sin(Inner_Angle) / Screw_Len
    return J1, KneeAngle_rad

class Sim2simCfg():
    # 注意这里加载的是场景xml，该文件里包含了a1.xml
    class sim_config:
        mujoco_model_path = f'legged_lab/assets/actiger/calf_description/scene.xml'
        sim_duration = 60.0
        dt = 0.001
        decimation = (0.02 / dt)
        num_actions = 12
    # 定义机器人关节PD参数和扭矩限制，12自由度
    class robot_config:
        kps = np.array([60.0]*12, dtype=np.double)
        kds = np.array([1.5]*12, dtype=np.double)
        tau_limit = 40. * np.ones(12, dtype=np.double)

def run_mujoco(cfg: Sim2simCfg):
    """
    Run the Mujoco simulation using the provided policy and configuration.

    Args:
        policy: The policy used for controlling the simulation.
        cfg: The configuration object containing simulation settings.

    Returns:
        None
    """
    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
    data = mujoco.MjData(model)

    model.opt.timestep = cfg.sim_config.dt

    tau = np.zeros(2)

    data.qpos[-2] = -1.0

    step_count = 0

    sin_pos_last = 0

    count_lowlevel = 0

    sin_pos = 0.0
    sin_pos_last_last = 0.0
    sin_pos_last = 0.0

    ff_accel = np.zeros(1)

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        # Close the viewer automatically after 30 wall-seconds.
        while viewer.is_running():
            global paused, one_step
            if paused:
                if one_step:
                    one_step = False
                else:
                    continue
            step_start = time.time()

            count_lowlevel += 1

            calf_joint_pos = data.joint('Calf_Joint').qpos
            calf_joint_vel = data.joint('Calf_Joint').qvel
            calf_joint_tau = data.joint("Calf_Joint").qfrc_constraint + data.joint("Calf_Joint").qfrc_smooth

            calf_motor_pos = -data.joint('Motor_Joint').qpos
            calf_motor_vel = -data.joint('Motor_Joint').qvel

            J1, calf_pos_jaco = calc_jacobian(calf_motor_pos)
            calf_pos_jaco = calf_motor_pos / 30
            calf_pos_jaco = np.array(calf_pos_jaco)
            cafl_vel_jaco = calf_motor_vel / J1 / J2_
            cafl_vel_jaco = calf_motor_vel / 30

            step_count += 1

            global action_at
            if count_lowlevel % cfg.sim_config.decimation == 0:
                if action_at:
                    # 频率（单位：Hz）
                    freq = 2.0   # 一个周期 2 秒
                    omega = 2 * np.pi * freq  # ω = 2πf
                    # 假设 step_count 是当前步数（从外部维护）
                    t = step_count * cfg.sim_config.dt
                    # 生成目标正弦轨迹
                    sin_pos = 0.2 * np.sin(omega * t) - 1.75  # 范围 [-2.0, -1.0]
                    # 二阶差分：a = (x_t - 2x_t−1 + x_t−2) / dt^2
                    dt = cfg.sim_config.dt * cfg.sim_config.decimation
                    ff_accel = (sin_pos - 2 * sin_pos_last + sin_pos_last_last) / (dt ** 2)
                    ff_accel = np.clip(ff_accel, -5, 5)
                    # 更新历史值
                    sin_pos_last_last = sin_pos_last
                    sin_pos_last = sin_pos
            if action_at:
                tau[0] = pd_control(sin_pos, calf_pos_jaco, 80.0, 0.0, cafl_vel_jaco, 1.0)
                tau[1] = pd_control(sin_pos, calf_joint_pos, 40.0, 0.0, calf_joint_vel, 1.0)

            sin_pos = np.array(sin_pos)
            data.ctrl[0] = -tau[0] / 30
            # data.ctrl[0] = -tau[0] / J1 / J2_
            # data.ctrl[1] = tau[1] + 0.46 * ff_accel
            # data.ctrl[1] = tau[1]
            plot(ff_accel, "ff_accel")
            plot(sin_pos, "sin_pos")
            plot(tau, "sin_tau")

            calf_tau_jaco = data.ctrl[0] * J1 * J2_
            
            plot(calf_joint_pos, "calf_joint_pos")
            plot(calf_joint_vel, "calf_joint_vel")
            plot(calf_joint_tau, "calf_joint_tau")

            plot(calf_pos_jaco, "calf_pos_jaco")
            plot(cafl_vel_jaco, "cafl_vel_jaco")
            plot(calf_tau_jaco, "calf_tau_jaco")

            plot(calf_motor_pos, "calf_motor_pos")
            plot(calf_motor_vel, "calf_motor_vel")

            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(model, data)
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


# 这里是main函数
if __name__ == '__main__':
    cfg = Sim2simCfg()
    # 创建两个线程
    thread1 = threading.Thread(target=run_mujoco, args=(cfg,))

    # 启动线程
    thread1.start()

    # 等待线程完成
    thread1.join()
