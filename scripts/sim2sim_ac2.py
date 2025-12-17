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

from utils.buffers import CircularBuffer

# TODO Default_joint
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

stable_joint_angles = {
            'FL_hip_joint': -0.0219698,   # [rad] LF_HAA
            'RL_hip_joint': 0.0852241,   # [rad] LH_HAA
            'FR_hip_joint': -0.048603 ,  # [rad] RF_HAA
            'RR_hip_joint': -0.010456,   # [rad] RH_HAA

            'FL_thigh_joint': 0.880923,   # [rad] LF_HFE 
            'RL_thigh_joint': 0.919451,   # [rad] LH_HFE
            'FR_thigh_joint': 0.85193,   # [rad] RF_HFE
            'RR_thigh_joint': 0.874819,   # [rad] RH_HFE

            'FL_calf_joint': -1.56318,   # [rad] LF_KFE
            'RL_calf_joint': -1.63017,   # [rad] LH_KFE
            'FR_calf_joint': -1.64577,   # [rad] RF_KFE
            'RR_calf_joint': -1.5879,   # [rad] RH_KFE
        }

joint_names = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"
]


# 设置打印选项，避免科学计数法，并保留小数点后 3 位
np.set_printoptions(suppress=True, precision=3)

def to_torch(x, dtype=torch.float, device='cuda:0', requires_grad=False):
    return torch.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)

device = 'cuda:0'

paused = False
one_step = False
action_at = True
done = False
disable_mode = False
disturbance = False
hangup = False
reset = False

policy_class = None


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

def get_yaw_from_quat(quat):
    """从四元数 [w, x, y, z] 提取偏航角 yaw（绕 Z 轴）"""
    w, x, y, z = quat

    # 计算偏航角（绕 Z 轴）
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return yaw  # 单位是弧度 [-pi, pi]

def get_obs(data, model):
    '''Extracts an observation from the mujoco data structure
    '''
    q = np.zeros(12)
    dq = np.zeros(12)
    for i, name in enumerate(joint_names):
        q[i] = data.joint(name).qpos
    for i, name in enumerate(joint_names):
        dq[i] = data.joint(name).qvel
    quat = data.sensor('orientation').data[[1, 2, 3, 0]].astype(np.double)
    r = R.from_quat(quat)
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)  # In the base frame
    omega = data.sensor('angular-velocity').data.astype(np.double)
    gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)
    
    foot_contact, foot_forces = get_foot_contact(data, model)

    q_noise = 0.01
    dq_noise = 0.2
    omega_noise = 0.0
    gvec_noise = 0.05

    # # 添加噪声（只加到腿部部分）
    # q[-12:] += np.random.uniform(-1, 1, size=(12,)) * q_noise
    # dq[-12:] += np.random.uniform(-1, 1, size=(12,)) * dq_noise
    # omega += np.random.uniform(-1, 1, size=(3,)) * omega_noise
    # gvec += np.random.uniform(-1, 1, size=(3,)) * gvec_noise

    obs_buffer.add((q, dq, quat, v, omega, gvec))

    obs_delay = obs_buffer.get_delayed_value(0.002)
    # return (q, dq, quat, v, omega, gvec)
    return obs_delay, foot_contact, foot_forces

def pd_control(target_q, q, kp, target_dq, dq, kd):
    '''Calculates torques from position commands
    '''
    return (target_q - q) * kp + (target_dq - dq) * kd

class Sim2simCfg():
    # 注意这里加载的是场景xml，该文件里包含了a1.xml
    class sim_config:
        mujoco_model_path = f'assets/actiger/ac2/urdf/scene_terrain.xml'
        sim_duration = 60.0
        dt = 0.005
        decimation = (0.02 / dt)
        num_actions = 12
    #TODO Joint PD params and torque limits
    class robot_config:
        kps = np.array([40.0]*12, dtype=np.double)
        kds = np.array([1.0]*12, dtype=np.double)
        # kps = np.array([60.0]*12, dtype=np.double)
        # kds = np.array([1.5]*12, dtype=np.double)
        tau_limit = 60. * np.ones(12, dtype=np.double)
        # kps[[2, 5, 8, 11]] = 60.0

# 常量定义
LEAD = 0.008
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

class SimpleNavigator:
    def __init__(self, goal_x, goal_y):
        self.goal_x = goal_x
        self.goal_y = goal_y

        # 控制参数
        self.k_v = 1.0         # 前向速度比例增益
        self.k_w = 6.5        # 角速度比例增益
        self.k_vy = 0.0        # 横向修正速度增益
        self.stop_radius = 0.1

        # 最大速度限制
        self.max_v = 2.0
        self.max_vy = 0.5
        self.max_w = 1.0

    def compute_command(self, pos_x, pos_y, yaw):
        dx = self.goal_x - pos_x
        dy = self.goal_y - pos_y

        distance = math.hypot(dx, dy)
        target_yaw = math.atan2(dy, dx)

        yaw_error = (target_yaw - yaw + math.pi) % (2 * math.pi) - math.pi

        local_x = math.cos(yaw) * dx + math.sin(yaw) * dy
        local_y = -math.sin(yaw) * dx + math.cos(yaw) * dy

        if distance < self.stop_radius:
            v = 0.0
            vy = 0.0
            w = 0.0
        else:
            # --- 角度衰减控制因子 ---
            yaw_factor = math.exp(-1 * abs(yaw_error))  # 抑制未对准时的前进速度

            # --- 综合速度控制 ---
            v = self.k_v * local_x
            vy = self.k_vy * local_y
            w = self.k_w * math.tanh(2.0 * yaw_error)  # 非线性角度响应

            # 限速处理
            v = max(-self.max_v, min(self.max_v, v))
            vy = max(-self.max_vy, min(self.max_vy, vy))
            w = max(-self.max_w, min(self.max_w, w))

            if abs(w) < 0.1:
                w = 0.0

            if abs(v) < 0.1:
                v = 0.0

            # v *= yaw_factor

        return v, vy, w


def run_mujoco(policy, estimator, lidar_encoder, cfg: Sim2simCfg):
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
    target_q = np.zeros((cfg.sim_config.num_actions), dtype=np.double)
    action = np.zeros((cfg.sim_config.num_actions), dtype=np.double)

    default_pos = np.zeros(cfg.sim_config.num_actions, dtype=np.double)
    default_pos_re = np.zeros(cfg.sim_config.num_actions, dtype=np.double)

    stable_pos = np.zeros(cfg.sim_config.num_actions, dtype=np.double)

    # 使用循环进行简化
    for i, leg in enumerate(['FL', 'FR', 'RL', 'RR']):
        default_pos[3*i:3*i+3] = [default_joint_angles[f'{leg}_hip_joint'],
                                default_joint_angles[f'{leg}_thigh_joint'],
                                default_joint_angles[f'{leg}_calf_joint']]
        stable_pos[3*i:3*i+3] = [stable_joint_angles[f'{leg}_hip_joint'],
                                stable_joint_angles[f'{leg}_thigh_joint'],
                                stable_joint_angles[f'{leg}_calf_joint']]

    data.qpos[-12:] = default_pos
    data.qpos[2] = 0.45

    target_q = default_pos_re.copy()

    #TODO Observations_list
    observations_list = ["commands", "ang_vel_body", "gravity_vec", "dof_pos", "dof_vel", "actions"]
    # observations_list = ["gravity_vec", "commands", "dof_pos", "dof_vel", "actions"]

    count_lowlevel = 0

    tracker = StepFrequencyTracker(window_size=200, dt=cfg.sim_config.dt)

    navigator = SimpleNavigator(goal_x=42.0, goal_y=0.0)

    obs_history = CircularBuffer(max_len=6, batch_size=1, device=device)

    if (len(observations_list) == 5):
        actor_obs = torch.zeros((1, 42), dtype=torch.float32, device=device)
    if (len(observations_list) == 6):
        actor_obs = torch.zeros((1, 45), dtype=torch.float32, device=device)
    
    motor_pos = np.zeros(4, dtype=np.double)
    motor_vel = np.zeros(4, dtype=np.double)
    calf_pos_jaco = np.zeros(4, dtype=np.double)
    cafl_vel_jaco = np.zeros(4, dtype=np.double)
    cafl_tau_jaco = np.zeros(4, dtype=np.double)
    command = np.zeros(3, dtype=np.double)
    # 重置 circular buffer 并写入历史帧
    obs_history.reset()
    for _ in range(obs_history.max_length):
        obs_history.append(actor_obs)
    time_start = time.time()
    time_counter = 0
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        # Close the viewer automatically after 30 wall-seconds.
        while viewer.is_running():
            global paused
            global one_step
            if paused:
                if one_step:
                    one_step = False
                else:
                    continue
            step_start = time.time()
            global reset
            if reset:
                reset = False
                time_counter = 0
                action[:] = 0
                # 重置 circular buffer 并写入历史帧
                obs_history.reset()
                for _ in range(obs_history.max_length):
                    obs_history.append(actor_obs)

            # Obtain an observation
            (q, dq, quat, v, omega, gvec), foot_contact, foot_forces = get_obs(data, model)
            q = q[-cfg.sim_config.num_actions:]
            dq = dq[-cfg.sim_config.num_actions:]
            yaw = get_yaw_from_quat(data.sensor('orientation').data)
            des_v, des_y, des_yaw = navigator.compute_command(data.qpos[0], data.qpos[1], yaw)
            time_counter += 1
            des_v *= 0.0
            des_y *= 0.0
            des_yaw *= 0.0
            
            for _ in range(4):
                J1, calf_pos_jaco = calc_jacobian(motor_pos)
                cafl_vel_jaco = motor_vel / J1 / J2_

            eu_ang = quaternion_to_euler_array(quat)
            eu_ang[eu_ang > math.pi] -= 2 * math.pi
            tracker.update(foot_contact)
            step_freq = tracker.get_step_frequency()
            # print("当前每条腿步频:", step_freq)
            plot(foot_contact, "foot_contact")
            plot(omega, "omega")
            plot(foot_forces,"foot_forces")
            plot(gvec, "gvec")
            plot(v, "v")
            plot(q, "q")
            plot(dq, "dq")
            # policy duartion
            if count_lowlevel % cfg.sim_config.decimation == 0:
                obs_list = []
                # 遍历 observations_list 并根据不同的观测名称添加数据
                for obs_name in observations_list:
                    if obs_name == "foot_contact":
                        obs_list.extend(foot_contact)
                    if obs_name == "ang_vel_body":
                        # 将 omega 添加到 obs_list
                        obs_list.extend(omega * 0.25)
                    elif obs_name == "gravity_vec":
                        # 将 gvec 添加到 obs_list
                        obs_list.extend(gvec)
                    elif obs_name == "commands":
                        # 将线速度 vx 和 vy 转换并添加到 obs_list
                        command[0] = -joy_data.Axis[1] * 2.0 * 2.0 + cmd.vx * 2.0 * 1.0 + des_v * 2.0
                        command[1] = -joy_data.Axis[0]  * 2.0 * 1.0 + cmd.vy * 2.0 + des_y * 2.0 
                        command[2] = -joy_data.Axis[3] * 0.25 + cmd.dyaw * 0.25 * np.pi / 2 + des_yaw * 0.25
                        obs_list.extend([command[0],
                                        command[1],
                                        command[2]])
                    elif obs_name == "dof_pos":
                        # 将关节位置转换并添加到 obs_list
                        obs_list.extend((q - default_pos) * 1.0)
                    elif obs_name == "dof_vel":
                        # 将关节速度转换并添加到 obs_list
                        obs_list.extend(dq * 0.05)
                    elif obs_name == "actions":
                        # 将动作添加到 obs_list
                        obs_list.extend(action)

                # 将整个 obs_list 转换为 NumPy 数组
                obs = np.array([obs_list], dtype=np.float32)
                obs_buffer.add(obs)
                obs_torch = to_torch(obs_buffer.get_delayed_value(0.00))
                obs_history.append(to_torch(obs))

                if policy_class == "ac2_roa":
                    action[:] = policy(obs_history.buffer.reshape(1, -1))[0].detach().cpu().numpy()
                else:
                    action[:] = policy(obs_torch)[0].detach().cpu().numpy()
                action = np.clip(action, -100, 100)
                #TODO Hip_reduction
                # action[[0, 3, 6, 9]] *= 0.5
                if action_at:
                    target_q = action * 0.25 + default_pos
                    target_q_buffer.add(target_q)
                else:
                    target_q = action * 0.25 * 0 + default_pos
                    target_q_buffer.add(target_q)
                    
            plot(target_q, "target_q")
            plot(command, "cmd")
            target_dq = np.zeros((cfg.sim_config.num_actions), dtype=np.double)

            # Generate PD control
            tau = pd_control(target_q_buffer.get_delayed_value(0.00), q, cfg.robot_config.kps,
                            target_dq, dq, cfg.robot_config.kds)  # Calc torques

            tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)
            global disable_mode
            if (abs(tau) > cfg.robot_config.tau_limit).any():
                disable_mode = True
            
            for i in range(4):
                J1, _ = calc_jacobian(motor_pos)
                cafl_tau_jaco = tau[2+i*3] / J1 / J2_

            global hangup
            if hangup:
                data.qvel[0] = 1.0
                for i, name in enumerate(joint_names):
                    data.joint(name).qpos = default_pos[i]

            plot(tau, "tau")
            data.ctrl[:12] = tau

            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(model, data)
            viewer.sync()

            count_lowlevel += 1
            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
            else:
                print("Warning: step took {:.3f}s".format(time.time() - step_start))
                pass
        global done
        done = True

joystick = None

def init_joystick():
    global joystick
    pygame.joystick.init()  # 初始化手柄
    joystick_count = pygame.joystick.get_count()
    print(joystick_count)

    if joystick_count > 0:
        try:
            joystick = pygame.joystick.Joystick(1)  # 连接第一个手柄
            joystick.init()
            print("Joystick initialized!")
        except pygame.error as e:
            print(f"Failed to initialize joystick: {e}")
            joystick = None
    else:
        joystick = None
        print("No joystick detected!")

def joy_loop():
    global done
    global joystick
    pygame.init()

    # 初始化手柄
    init_joystick()

    while not done:
        pygame.event.pump()  # 处理事件队列

        # 检查手柄是否连接，使用 get_init() 判断是否初始化
        if joystick is None or not joystick.get_init():
            print("Joystick disconnected, attempting to reconnect...")
            init_joystick()  # 如果手柄断开，尝试重新连接
            time.sleep(1)  # 重连时增加 sleep 时间，给重连过程提供更多时间
        else:
            try:
                # 处理手柄轴输入
                for axis in range(joystick.get_numaxes()):
                    value = joystick.get_axis(axis)
                    joy_data.Axis[axis] = value

                    # 设定阈值，避免小幅度的摇动
                    if abs(joy_data.Axis[axis]) < 0.2:
                        joy_data.Axis[axis] = 0

                # 处理按键输入
                for button in range(joystick.get_numbuttons()):
                    if joystick.get_button(button):
                        joy_data.button_last = joy_data.button
                        joy_data.button = button

            except pygame.error:
                print("Joystick disconnected during use, attempting to reconnect...")
                joystick = None  # 手柄断开时，清除旧的 joystick 实例
                init_joystick()  # 尝试重新连接手柄

        time.sleep(0.01)  # 控制循环频率

# 这里是main函数
if __name__ == '__main__':
    #TODO Policy path
    path = f'pre-trian_policy/model_20000.pt'
    if path == f'pre-trian_policy/model_20000.pt':
        policy_class = "ac2_roa"
    else:
        policy_class = "ac2_amp"
    policy = torch.jit.load(path, map_location=torch.device('cuda'))

    # 实例化配置类obs_buffer
    obs_buffer = ObsBuffer(max_length=10, dt=Sim2simCfg.sim_config.dt)
    target_q_buffer = ObsBuffer(max_length=10, dt=Sim2simCfg.sim_config.dt)
    
    # 创建两个线程
    thread1 = threading.Thread(target=run_mujoco, args=(policy, None, None, Sim2simCfg))
    thread2 = threading.Thread(target=joy_loop, args=())

    # 启动线程
    thread1.start()
    thread2.start()

    # 等待线程完成
    thread1.join()
    thread2.join()
