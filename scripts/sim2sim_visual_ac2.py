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
import torch.nn as nn # Added for Vision Network
import os

# ===================================================================
# [新增] 视觉网络结构定义 (必须保留以加载权重)
# ===================================================================
class DepthOnlyFCBackbone58x87(nn.Module):
    def __init__(self, prop_dim, scandots_output_dim, hidden_state_dim, output_activation=None, num_frames=1):
        super().__init__()
        self.num_frames = num_frames
        activation = nn.ELU()
        self.image_compression = nn.Sequential(
            nn.Conv2d(in_channels=self.num_frames, out_channels=32, kernel_size=5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            activation,
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            activation,
            nn.Flatten(),
            nn.Linear(64 * 25 * 39, 128),
            activation,
            nn.Linear(128, scandots_output_dim)
        )
        self.output_activation = nn.Tanh() if output_activation == "tanh" else activation

    def forward(self, images: torch.Tensor):
        images_compressed = self.image_compression(images.unsqueeze(1))
        latent = self.output_activation(images_compressed)
        return latent

class RecurrentDepthBackbone(nn.Module):
    def __init__(self, base_backbone, env_cfg) -> None:
        super().__init__()
        activation = nn.ELU()
        last_activation = nn.Tanh()
        self.base_backbone = base_backbone
        self.combination_mlp = nn.Sequential(
            nn.Linear(32 + env_cfg.env.n_proprio, 128),
            activation,
            nn.Linear(128, 32)
        )
        self.rnn = nn.GRU(input_size=32, hidden_size=512, batch_first=True)
        self.output_mlp = nn.Sequential(
            nn.Linear(512, 32+2), 
            last_activation
        )
        self.hidden_states = None

    def forward(self, depth_image, proprioception):
        depth_latent_cnn = self.base_backbone(depth_image)
        combined_input = torch.cat((depth_latent_cnn, proprioception), dim=-1)
        depth_latent = self.combination_mlp(combined_input)
        
        if self.hidden_states is None or self.hidden_states.shape[1] != depth_latent.shape[0]:
            self.hidden_states = torch.zeros(1, depth_latent.shape[0], 512, device=depth_latent.device)
            
        depth_latent, self.hidden_states = self.rnn(depth_latent[:, None, :], self.hidden_states)
        output = self.output_mlp(depth_latent.squeeze(1))
        return output

    def reset(self):
        self.hidden_states = None

class MockEnvCfg:
    def __init__(self):
        self.env = self.Env()
    class Env:
        def __init__(self):
            self.n_proprio = 53 

# ===================================================================
# 原有代码逻辑
# ===================================================================

# from plot import plot # 如果没有plot文件，可以注释掉
def plot(*args): pass

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
        dt = self.cfg.sim_config.dt
        tau_limit = self.cfg.robot_config.tau_limit
        num_actions = self.cfg.sim_config.num_actions

        self.timer += dt / speed
        alpha = min(self.timer, 1.0)

        target_q = target_pos * alpha + self.start_pos * (1 - alpha)
        target_dq = np.zeros(num_actions)
        tau = pd_control(target_q, q, p_gain, target_dq, dq, d_gain)
        tau = np.clip(tau, -tau_limit, tau_limit)
        # 注意：这里需要配合 Actuator Mapping，稍后在 main loop 统一处理
        # data.ctrl[:12] = tau 
        return alpha >= 1.0

    def update(self, q, dq, data, pd_control,
               stable_pos, default_pos, crouch_pos,
               pre_stand=False, full_stand=False, pre_crouch=False, crouch=False):
        done = False
        # (FSM logic remains same, removed for brevity as it is unused in Policy loop)
        return self.state, done

# [修正] 默认关节角度 (Vision Policy Standard: Hip 0, Thigh 0.9, Calf -1.55)
default_joint_angles = {
    'FL_hip_joint': -0.1, 'RL_hip_joint': -0.1, 'FR_hip_joint': 0.1, 'RR_hip_joint': 0.1,
    'FL_thigh_joint': 0.9, 'RL_thigh_joint': 0.9, 'FR_thigh_joint': 0.9, 'RR_thigh_joint': 0.9,
    'FL_calf_joint': -1.55, 'RL_calf_joint': -1.55, 'FR_calf_joint': -1.55, 'RR_calf_joint': -1.55,
}

stable_joint_angles = default_joint_angles.copy() # Placeholder
lay_down_joint_angles = default_joint_angles.copy() # Placeholder

joint_names = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"
]

np.set_printoptions(suppress=True, precision=3)

def to_torch(x, dtype=torch.float, device='cuda:0', requires_grad=False):
    return torch.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)

device = 'cuda:0'

# Global flags
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

button = { 'X': 2, 'Y': 3, 'B': 1, 'A': 0, 'LB': 4, 'RB': 5, 'SELECT': 6, 'START': 7, 'home': 8, 'return': 11 }

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
    global reset, paused, action_at, one_step
    if keycode == 259: reset = not reset
    if chr(keycode) == ' ': paused = not paused
    if keycode == 320: action_at = not action_at
    if keycode == 334: one_step = not one_step
    if keycode == 265: cmd.vx = 1.0
    elif keycode == 264: cmd.vx = -1.0
    else: cmd.vx = 0.0
    if keycode == 263: cmd.dyaw = 1.0  
    elif keycode == 262: cmd.dyaw = -1.0
    else: cmd.dyaw = 0.0

@torch.jit.script
def quat_rotate_inverse(q, v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
    return a - b + c

def get_foot_contact(data, model):
    foot_forces = np.zeros(4)
    foot_contact = np.zeros(4)
    foot_geom_names = ['FL_foot_fixed', 'FR_foot_fixed', 'RL_foot_fixed', 'RR_foot_fixed']
    foot_idx = {name: i for i, name in enumerate(foot_geom_names)}
    force = np.zeros(6)
    ground_geom_names = ['floor']
    for i in range(data.ncon):
        contact = data.contact[i]
        mujoco.mj_contactForce(model, data, i, force)
        g1 = model.geom(contact.geom1).name
        g2 = model.geom(contact.geom2).name
        if (g1 in ground_geom_names and g2 in foot_geom_names) or (g2 in ground_geom_names and g1 in foot_geom_names):
            foot_name = g1 if g2 in ground_geom_names else g2   
            force_norm = np.linalg.norm(force[0:3])
            foot_forces[foot_idx[foot_name]] = force_norm
            foot_contact[foot_idx[foot_name]] = 1.0
    return foot_contact, foot_forces

def get_obs(data, model):
    # This function extracts raw data, the assembly into 53-dim tensor happens in run_mujoco
    q = data.qpos[7:] # Skip free joint
    dq = data.qvel[6:]
    quat = data.qpos[3:7] # [w, x, y, z]
    
    # Transform Quat to [x, y, z, w] for Scipy/Vision compatibility
    quat_scipy = np.array([quat[1], quat[2], quat[3], quat[0]])
    r = R.from_quat(quat_scipy)
    
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)
    omega = data.qvel[3:6] # Base ang vel
    # Vision code usually projects gravity inverse
    gvec = r.inv().apply(np.array([0., 0., -1.])).astype(np.double)
    
    foot_contact, foot_forces = get_foot_contact(data, model)
    obs_delay = q, dq, quat_scipy, v, omega, gvec # Return quat in x,y,z,w format
    return obs_delay, foot_contact, foot_forces

def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd

class Sim2simCfg():
    class sim_config:
        # [Updated Path]
        mujoco_model_path = "assets/actiger/ac2/urdf/scene_terrain.xml"
        # [Vision Path]
        vision_weight_path = "pre-trian_policy/ac2-visual-robocon_v1.2-15500-vision_weight.pt"
        
        sim_duration = 60.0
        dt = 0.005
        decimation = 4
        num_actions = 12
        
        # [Vision Params]
        lin_vel_scale = 2.0
        ang_vel_scale = 0.25
        dof_pos_scale = 1.0
        dof_vel_scale = 0.05
        action_scale = 0.25
        dim_scan = 132
        dim_priv = 9

    class robot_config:
        # [Updated Gains]
        kps = np.array([40.0]*12, dtype=np.double)
        kds = np.array([1.0]*12, dtype=np.double)
        tau_limit = 60. * np.ones(12, dtype=np.double)

def run_mujoco(policy, cfg: Sim2simCfg):
    global done, reset, action_at
    
    if not os.path.exists(cfg.sim_config.mujoco_model_path):
        print("XML Not Found!")
        return

    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
    data = mujoco.MjData(model)
    model.opt.timestep = cfg.sim_config.dt

    # ---  Init Vision ---
    print("Initializing Vision...")
    try:
        loaded_obj = torch.jit.load(cfg.sim_config.vision_weight_path, map_location=device)
        vision_model = loaded_obj
        print("Vision JIT Loaded")
    except:
        loaded_obj = torch.load(cfg.sim_config.vision_weight_path, map_location=device, weights_only=False)
        mock_env = MockEnvCfg()
        backbone = DepthOnlyFCBackbone58x87(mock_env.env.n_proprio, 32, [512, 256, 128], 'elu')
        vision_model = RecurrentDepthBackbone(backbone, mock_env).to(device)
        if isinstance(loaded_obj, dict):
            if 'depth_encoder_state_dict' in loaded_obj:
                vision_model.load_state_dict(loaded_obj['depth_encoder_state_dict'])
            else:
                vision_model.load_state_dict(loaded_obj)
        else:
            vision_model = loaded_obj
        print("Vision Dict Loaded")
    vision_model.eval()

    # --- Setup Renderer ---
    renderer = mujoco.Renderer(model, height=58, width=87)
    try:
        camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "depth_camera")
    except:
        camera_id = 0

    # ---  Re-indexing Logic ---
    all_joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
    policy_joint_names = ["FL_hip_joint", "FL_thigh_joint", "FL_calf_joint", 
                          "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
                          "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
                          "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"]
    
    joint_indices = []
    for name in policy_joint_names:
        for i, xml_name in enumerate(all_joint_names):
            if name in xml_name: 
                joint_indices.append(i)
                break
    joint_indices = np.array(joint_indices)
    
    actuator_indices = []
    for idx in joint_indices:
        for i in range(model.nu):
            if model.actuator_trnid[i, 0] == idx:
                actuator_indices.append(i)
                break
    actuator_indices = np.array(actuator_indices)
    print(f"Joints Re-indexed: {joint_indices}")

    # Initialize positions
    default_pos = np.zeros(cfg.sim_config.num_actions, dtype=np.double)
    for i, leg in enumerate(['FL', 'FR', 'RL', 'RR']):
        default_pos[3*i:3*i+3] = [default_joint_angles[f'{leg}_hip_joint'],
                                  default_joint_angles[f'{leg}_thigh_joint'],
                                  default_joint_angles[f'{leg}_calf_joint']]

    target_q = default_pos.copy()
    target_dq = np.zeros(cfg.sim_config.num_actions, dtype=np.double)
    last_action = torch.zeros(12, device=device)
    command = np.zeros(3, dtype=np.double)

    # Initial Physics Setup
    def reset_sim():
        data.qpos[:] = 0; data.qvel[:] = 0
        data.qpos[2] = 0.6; data.qpos[3] = 1.0
        q_corr = joint_indices - 1
        for i, idx in enumerate(q_corr): data.qpos[7+idx] = default_pos[i]
        if hasattr(vision_model, "reset"): vision_model.reset()
        mujoco.mj_forward(model, data)
        for _ in range(20): mujoco.mj_step(model, data)
    
    reset_sim()
    count_lowlevel = 0

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        (q_raw, dq_raw, quat, v, omega, gvec), _, _ = get_obs(data, model)
        
        while viewer.is_running():
            global paused
            if paused:
                time.sleep(0.1); viewer.sync(); continue
            
            step_start = time.time()
            if reset:
                reset = False
                reset_sim()
                last_action[:] = 0

            # --- Control Loop (50Hz) ---
            if count_lowlevel % cfg.sim_config.decimation == 0:
                with torch.no_grad():
                    # 1. Update Proprioception (53 dim)
                    obs_list = []
                    # Angular Vel (3)
                    base_ang_vel_body = quat_rotate_inverse(to_torch(quat[None]), to_torch(omega[None])).squeeze(0)
                    obs_list.append(base_ang_vel_body * cfg.sim_config.ang_vel_scale)
                    # Gravity (2)
                    obs_list.append(to_torch(gvec[:2]))
                    # Commands (3)
                    command[0] = -joy_data.Axis[1] * 2.0 * 2.0 + cmd.vx * 2.0
                    command[1] = -joy_data.Axis[0] * 2.0 * 1.0 + cmd.vy * 2.0
                    command[2] = -joy_data.Axis[3] * 0.25 + cmd.dyaw * 0.25 * np.pi/2
                    cmd_tensor = to_torch([
                        command[0] * cfg.sim_config.lin_vel_scale / 2.0, 
                        command[1] * cfg.sim_config.lin_vel_scale / 2.0,
                        command[2] * cfg.sim_config.ang_vel_scale / 0.25
                    ]) # Adjusting for joystick logic
                    obs_list.append(cmd_tensor)
                    # Misc (4)
                    obs_list.append(torch.zeros(4, device=device))
                    # Joint Pos (12)
                    q_corr = joint_indices - 1
                    curr_q = data.qpos[7:][q_corr]
                    curr_dq = data.qvel[6:][q_corr]
                    dof_err = (to_torch(curr_q) - to_torch(default_pos)) * cfg.sim_config.dof_pos_scale
                    obs_list.append(dof_err)
                    # Joint Vel (12)
                    obs_list.append(to_torch(curr_dq) * cfg.sim_config.dof_vel_scale)
                    # Action (12)
                    obs_list.append(last_action)
                    # Feet (5)
                    obs_list.append(torch.zeros(5, device=device))
                    
                    prop_53 = torch.cat(obs_list).view(1, -1)
                    if prop_53.shape[1] < 53: 
                        prop_53 = torch.cat([prop_53, torch.zeros(1, 53-prop_53.shape[1], device=device)], dim=1)

                    # 2. Get Depth
                    renderer.update_scene(data, camera=camera_id)
                    renderer.enable_depth_rendering()
                    depth_img = renderer.render()
                    renderer.disable_depth_rendering()
                    depth_t = torch.from_numpy(depth_img).float().to(device)
                    depth_t = torch.clamp(depth_t, 0.0, 3.0) / 3.0
                    depth_t = depth_t.unsqueeze(0)

                    # 3. Vision Forward
                    if not torch.isnan(prop_53).any():
                        vis_out = vision_model(depth_t, prop_53)
                        latent = vis_out[:, :32]
                        
                        # Full Obs
                        scan_pad = torch.zeros(1, cfg.sim_config.dim_scan, device=device)
                        priv_pad = torch.zeros(1, cfg.sim_config.dim_priv, device=device)
                        hist_pad = torch.zeros(1, 600, device=device)
                        
                        full_obs = torch.cat([prop_53, scan_pad, priv_pad, hist_pad], dim=1)
                        
                        # Policy
                        raw_action = policy(full_obs, latent)[0]
                        last_action = raw_action
                        
                        action_np = raw_action.detach().cpu().numpy()
                        action_np = np.clip(action_np, -100, 100)
                        
                        if action_at:
                            target_q = action_np * cfg.sim_config.action_scale + default_pos
                        else:
                            target_q = default_pos.copy()

            # --- Physics Loop (200Hz) ---
            q_corr = joint_indices - 1
            all_q = data.qpos[7:]
            all_dq = data.qvel[6:]
            
            q_active = all_q[q_corr]
            dq_active = all_dq[q_corr]
            
            tau = cfg.robot_config.kps * (target_q - q_active) - cfg.robot_config.kds * dq_active
            tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)
            
            if not np.isnan(tau).any():
                data.ctrl[actuator_indices] = tau
                
            try:
                mujoco.mj_step(model, data)
            except Exception as e:
                print(f"Sim Error: {e}")
                reset_sim()

            if count_lowlevel % 15 == 0:
                viewer.sync()
            
            count_lowlevel += 1
            time_until_next = model.opt.timestep - (time.time() - step_start)
            if time_until_next > 0: time.sleep(time_until_next)
            
        done = True
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
    # Keep original joystick loop structure as requested
    global done, joystick
    pygame.init()
    init_joystick()
    while not done:
        pygame.event.pump()
        if joystick is None or not joystick.get_init():
            init_joystick()
            time.sleep(1)
        else:
            try:
                for axis in range(joystick.get_numaxes()):
                    val = joystick.get_axis(axis)
                    joy_data.Axis[axis] = val if abs(val) > 0.05 else 0
                for button in range(joystick.get_numbuttons()):
                    if joystick.get_button(button):
                        joy_data.button_last = joy_data.button
                        joy_data.button = button
            except:
                joystick = None
        time.sleep(0.01)

if __name__ == '__main__':
    # Policy Path
    path = "pre-trian_policy/ac2-visual-robocon_v1.2-15500-base_jit.pt"
    print(f"Loading Policy: {path}")
    policy = torch.jit.load(path, map_location=device)
    
    thread1 = threading.Thread(target=run_mujoco, args=(policy, Sim2simCfg))
    thread2 = threading.Thread(target=joy_loop, args=())

    thread1.start()
    thread2.start()
    thread1.join()
    thread2.join()