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

from enum import Enum, auto

class StandState(Enum):
    IDLE = auto()
    PRE_STAND = auto()
    FULL_STAND = auto()
    PRE_CROUCH = auto()
    CROUCH = auto()
    STAND_DONE = auto()


class StandFSM:
    def __init__(self, cfg):
        self.cfg = cfg
        self.state = StandState.IDLE
        self.timer = 0.0
        self.start_pos = None

    def reset(self):
        self.state = StandState.IDLE
        self.timer = 0.0
        self.start_pos = None

    def _interpolate_motion(self, target_pos, q, dq, data, pd_control,
                            p_gain=100, d_gain=2, speed=1.0):
        """线性插值过渡 + PD 控制"""
        dt = self.cfg.sim_config.dt
        tau_limit = self.cfg.robot_config.tau_limit
        num_actions = self.cfg.sim_config.num_actions

        self.timer += dt / speed
        alpha = min(self.timer, 1.0)

        target_q = target_pos * alpha + self.start_pos * (1 - alpha)
        target_dq = np.zeros(num_actions)
        tau = pd_control(target_q, q, p_gain, target_dq, dq, d_gain)
        tau = np.clip(tau, -tau_limit, tau_limit)
        data.ctrl[:12] = tau
        plot(target_q, "target_stand_pos")

        return alpha >= 1.0  # 返回是否完成过渡

    def update(self, q, dq, data, pd_control,
               stable_pos, default_pos, crouch_pos,
               pre_stand=False, full_stand=False, pre_crouch=False, crouch=False):
        """
        参数:
        - q, dq: 当前状态
        - data.ctrl: 输出控制命令
        - stable_pos, default_pos, pre_crouch_pos, crouch_pos: 姿态目标
        """
        done = False

        # --- 状态逻辑 ---
        if self.state == StandState.IDLE:
            if pre_stand:
                self.state = StandState.PRE_STAND
                self.timer = 0.0
                self.start_pos = q.copy()

        elif self.state == StandState.PRE_STAND:
            if self._interpolate_motion(stable_pos, q, dq, data, pd_control, speed=1.5):
                if full_stand:
                    self.state = StandState.FULL_STAND
                    self.timer = 0.0
                    self.start_pos = q.copy()

        elif self.state == StandState.FULL_STAND:
            if self._interpolate_motion(default_pos, q, dq, data, pd_control, speed=3.0):
                if pre_crouch:
                    self.state = StandState.PRE_CROUCH
                    self.timer = 0.0
                    self.start_pos = q.copy()

        elif self.state == StandState.PRE_CROUCH:
            if self._interpolate_motion(stable_pos, q, dq, data, pd_control, speed=3.0):
                if crouch:
                    self.state = StandState.CROUCH
                    self.timer = 0.0
                    self.start_pos = q.copy()

        elif self.state == StandState.CROUCH:
            if self._interpolate_motion(crouch_pos, q, dq, data, pd_control, speed=1.5):
                pass

        return self.state, done




# TODO Default_joint
default_joint_angles = {
            'FL_hip_joint': -0.1,   # [rad] LF_HAA
            'RL_hip_joint': -0.1,   # [rad] LH_HAA
            'FR_hip_joint': 0.1 ,  # [rad] RF_HAA
            'RR_hip_joint': 0.1,   # [rad] RH_HAA

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
            'FL_hip_joint': -0.0,   # [rad] LF_HAA
            'RL_hip_joint': -0.0,   # [rad] LH_HAA
            'FR_hip_joint': 0.0 ,  # [rad] RF_HAA
            'RR_hip_joint': 0.0,   # [rad] RH_HAA

            'FL_thigh_joint': 1.3,   # [rad] LF_HFE 
            'RL_thigh_joint': 1.3,   # [rad] LH_HFE
            'FR_thigh_joint': 1.3,   # [rad] RF_HFE
            'RR_thigh_joint': 1.3,   # [rad] RH_HFE

            'FL_calf_joint': -2.64,   # [rad] LF_KFE
            'RL_calf_joint': -2.64,   # [rad] LH_KFE
            'FR_calf_joint': -2.64,   # [rad] RF_KFE
            'RR_calf_joint': -2.64,   # [rad] RH_KFE
        }

lay_down_joint_angles = {
            'FL_hip_joint': 0.635,   # [rad] LF_HAA
            'RL_hip_joint': 0.635,   # [rad] LH_HAA
            'FR_hip_joint': -0.635 ,  # [rad] RF_HAA
            'RR_hip_joint': -0.635,   # [rad] RH_HAA

            'FL_thigh_joint': 1.09,   # [rad] LF_HFE 
            'RL_thigh_joint': 1.09,   # [rad] LH_HFE
            'FR_thigh_joint': 1.09,   # [rad] RF_HFE
            'RR_thigh_joint': 1.09,   # [rad] RH_HFE

            'FL_calf_joint': -2.7,   # [rad] LF_KFE
            'RL_calf_joint': -2.7,   # [rad] LH_KFE
            'FR_calf_joint': -2.7,   # [rad] RF_KFE
            'RR_calf_joint': -2.7,   # [rad] RH_KFE
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
pre_stand = False
full_stand = False
pre_crouch = False
crouch = False
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
        global pre_stand
        pre_stand = not pre_stand

    if keycode == 323:
        global full_stand
        full_stand = not full_stand

    if keycode == 324:
        global pre_crouch
        pre_crouch = not pre_crouch

    if keycode == 325:
        global crouch
        crouch = not crouch

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

    obs_delay = q, dq, quat, v, omega, gvec

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
        tau_limit = 60. * np.ones(12, dtype=np.double)

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

def run_mujoco(policy, cfg: Sim2simCfg):
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

    crouch_pos = np.zeros(cfg.sim_config.num_actions, dtype=np.double)

    # 使用循环进行简化
    for i, leg in enumerate(['FL', 'FR', 'RL', 'RR']):
        default_pos[3*i:3*i+3] = [default_joint_angles[f'{leg}_hip_joint'],
                                default_joint_angles[f'{leg}_thigh_joint'],
                                default_joint_angles[f'{leg}_calf_joint']]
        stable_pos[3*i:3*i+3] = [stable_joint_angles[f'{leg}_hip_joint'],
                                stable_joint_angles[f'{leg}_thigh_joint'],
                                stable_joint_angles[f'{leg}_calf_joint']]
        crouch_pos[3*i:3*i+3] = [lay_down_joint_angles[f'{leg}_hip_joint'],
                                lay_down_joint_angles[f'{leg}_thigh_joint'],
                                lay_down_joint_angles[f'{leg}_calf_joint']]
        

    data.qpos[-12:] = default_pos
    data.qpos[2] = 0.45

    target_q = default_pos_re.copy()

    target_dq = np.zeros((cfg.sim_config.num_actions), dtype=np.double)

    #TODO Observations_list
    observations_list = ["commands", "ang_vel_body", "gravity_vec", "dof_pos", "dof_vel", "actions"]

    count_lowlevel = 0

    obs_history = CircularBuffer(max_len=6, batch_size=1, device=device)

    if (len(observations_list) == 5):
        actor_obs = torch.zeros((1, 42), dtype=torch.float32, device=device)
    if (len(observations_list) == 6):
        actor_obs = torch.zeros((1, 45), dtype=torch.float32, device=device)
    
    command = np.zeros(3, dtype=np.double)
    # 重置 circular buffer 并写入历史帧
    obs_history.reset()
    for _ in range(obs_history.max_length):
        obs_history.append(actor_obs)

    fsm = StandFSM(cfg)
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        # Close the viewer automatically after 30 wall-seconds.
        # Obtain an observation
        (q, dq, quat, v, omega, gvec), foot_contact, foot_forces = get_obs(data, model)
        q = q[-cfg.sim_config.num_actions:]
        dq = dq[-cfg.sim_config.num_actions:]
        while viewer.is_running():
            global paused
            global one_step
            if paused:
                if one_step:
                    one_step = False
                else:
                    continue
            step_start = time.time()
            global reset, pre_stand, full_stand, pre_crouch, crouch
            if reset:
                reset = False
                pre_stand = False
                full_stand = False
                pre_crouch = False
                crouch = False
                fsm.reset()
                action[:] = 0
                # 重置 circular buffer 并写入历史帧
                obs_history.reset()
                for _ in range(obs_history.max_length):
                    obs_history.append(actor_obs)

            # Generate PD control
            tau = pd_control(target_q, q, cfg.robot_config.kps,
                            target_dq, dq, cfg.robot_config.kds)  # Calc torques
            tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)
            data.ctrl[:12] = tau
            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(model, data)

            # Obtain an observation
            (q, dq, quat, v, omega, gvec), foot_contact, foot_forces = get_obs(data, model)
            q = q[-cfg.sim_config.num_actions:]
            dq = dq[-cfg.sim_config.num_actions:]

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
                        command[0] = -joy_data.Axis[1] * 2.0 * 2.0 + cmd.vx * 2.0 * 1.0
                        command[1] = -joy_data.Axis[0]  * 2.0 * 1.0 + cmd.vy * 2.0
                        command[2] = -joy_data.Axis[3] * 0.25 + cmd.dyaw * 0.25 * np.pi / 2
                        obs_list.extend([command[0],
                                        command[1],
                                        command[2]])
                        print(f"vx: {command[0]/2.0:.2f}, vy: {command[1]/2.0:.2f}, dyaw: {command[2]/0.25:.2f}")
                        print(f"real vx: {v[0]:.2f}, real vy: {v[1]:.2f}, real yaw rate: {omega[2]:.2f}")
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
                obs_history.append(to_torch(obs))

                if policy_class == "ac2_roa":
                    action[:] = policy(obs_history.buffer.reshape(1, -1))[0].detach().cpu().numpy()
                else:
                    action[:] = 0
                action = np.clip(action, -100, 100)
                #TODO Hip_reduction
                # action[[0, 3, 6, 9]] *= 0.5
                if action_at:
                    target_q = action * 0.25 + default_pos
                else:
                    target_q = action * 0.25 * 0 + default_pos
           
            # fsm.update(q, dq, data, pd_control, stable_pos, default_pos, crouch_pos,
            #            pre_stand=pre_stand, full_stand=full_stand, pre_crouch=pre_crouch, crouch=crouch)
            
            viewer.sync()

            count_lowlevel += 1
            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
            else:
                # print("Warning: step took {:.3f}s".format(time.time() - step_start))
                pass
        global done
        done = True

joystick = None

def init_joystick():
    global joystick
    pygame.joystick.init()  # 初始化手柄
    joystick_count = pygame.joystick.get_count()

    if joystick_count > 0:
        try:
            joystick = pygame.joystick.Joystick(1)  # 连接第一个手柄
            joystick.init()
            print("Joystick initialized!")
        except pygame.error as e:
            # print(f"Failed to initialize joystick: {e}")
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
            # print("Joystick disconnected, attempting to reconnect...")
            init_joystick()  # 如果手柄断开，尝试重新连接
            time.sleep(1)  # 重连时增加 sleep 时间，给重连过程提供更多时间
        else:
            try:
                # 处理手柄轴输入
                for axis in range(joystick.get_numaxes()):
                    value = joystick.get_axis(axis)
                    joy_data.Axis[axis] = value

                    # 设定阈值，避免小幅度的摇动
                    if abs(joy_data.Axis[axis]) < 0.05:
                        joy_data.Axis[axis] = 0

                # 处理按键输入
                for button in range(joystick.get_numbuttons()):
                    if joystick.get_button(button):
                        joy_data.button_last = joy_data.button
                        joy_data.button = button

            except pygame.error:
                # print("Joystick disconnected during use, attempting to reconnect...")
                joystick = None  # 手柄断开时，清除旧的 joystick 实例
                init_joystick()  # 尝试重新连接手柄

        time.sleep(0.01)  # 控制循环频率

# 这里是main函数
if __name__ == '__main__':
    #TODO Policy path
    path = f'pre-trian_policy/policy.pt'
    if path == f'pre-trian_policy/policy.pt':
        policy_class = "ac2_roa"
    else:
        policy_class = "ac2_amp"
    policy = torch.jit.load(path, map_location=torch.device('cuda:0'))
    
    # 创建两个线程
    thread1 = threading.Thread(target=run_mujoco, args=(policy, Sim2simCfg))
    thread2 = threading.Thread(target=joy_loop, args=())

    # 启动线程
    thread1.start()
    thread2.start()

    # 等待线程完成
    thread1.join()
    thread2.join()