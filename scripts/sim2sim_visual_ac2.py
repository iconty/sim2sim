import time
import numpy as np
import math
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as R
from collections import deque
import torch.nn.functional as F
import pygame
import threading
import torch
import torch.nn as nn
import os

# ===================================================================
# 1. 视觉网络结构 
# ===================================================================

class DepthOnlyFCBackbone58x87(nn.Module):
    def __init__(self, prop_dim, scandots_output_dim, hidden_state_dim, output_activation=None, num_frames=1):
        super().__init__()

        self.num_frames = num_frames
        activation = nn.ELU()
        self.image_compression = nn.Sequential(
            # [1, 58, 87]
            nn.Conv2d(in_channels=self.num_frames, out_channels=32, kernel_size=5),
            # [32, 54, 83]
            nn.MaxPool2d(kernel_size=2, stride=2),
            # [32, 27, 41]
            activation,
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            activation,
            nn.Flatten(),
            # [32, 25, 39]
            nn.Linear(64 * 25 * 39, 128),
            activation,
            nn.Linear(128, scandots_output_dim)
        )

        if output_activation == "tanh":
            self.output_activation = nn.Tanh()
        else:
            self.output_activation = activation

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
        if env_cfg == None:
            self.combination_mlp = nn.Sequential(
                                    nn.Linear(32 + 53, 128),
                                    activation,
                                    nn.Linear(128, 32)
                                )
        else:
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
        depth_image = self.base_backbone(depth_image)
        depth_latent = self.combination_mlp(torch.cat((depth_image, proprioception), dim=-1))
        # depth_latent = self.base_backbone(depth_image)
        depth_latent, self.hidden_states = self.rnn(depth_latent[:, None, :], self.hidden_states)
        depth_latent = self.output_mlp(depth_latent.squeeze(1))
        
        return depth_latent

    def detach_hidden_states(self):
        self.hidden_states = self.hidden_states.detach().clone()

class MockEnvCfg:
    def __init__(self):
        self.env = self.Env()
    class Env:
        def __init__(self):
            self.n_proprio = 53 

# ===================================================================
# 2. 全局配置与状态变量
# ===================================================================

class Sim2simCfg:
    class sim_config:
        mujoco_model_path = "assets/actiger/ac2/urdf/scene_terrain.xml"
        base_jit_path = "pre-trian_policy/ac2-visual-robocon_v1.2-15500-base_jit.pt"
        vision_weight_path = "pre-trian_policy/ac2-visual-robocon_v1.2-15500-vision_weight.pt"
        
        dt = 0.005
        decimation = 4
        num_actions = 12
        
        # Dimensions
        n_proprio = 53
        history_len = 10
        dim_scan = 132
        dim_priv_explicit = 9
        dim_priv_latent = 29

        
        # Scales
        lin_vel_scale = 2.0
        ang_vel_scale = 0.25
        dof_pos_scale = 1.0
        dof_vel_scale = 0.05
        action_scale = 0.5
        clip_actions = 1.2

        #depth
        near_clip = 0
        far_clip = 2
        camera_name = 'depth_camera'
        camera_width_original = 106
        camera_height_original = 60
    
        camera_width_resized = 87
        camera_height_resized = 58

    class robot_config:
        kps = np.array([40.0]*12, dtype=np.double)
        kds = np.array([1.0]*12, dtype=np.double)
        tau_limit = 60. * np.ones(12, dtype=np.double)
        # 默认站立姿态 (FL, FR, RL, RR)
        default_joint_angles = np.array([
            -0.1, 0.9, -1.55,  # FL
            0.1, 0.9, -1.55,   # FR
            -0.1, 0.9, -1.55,  # RL
            0.1, 0.9, -1.55    # RR
        ], dtype=np.double)

done = False
paused = False
reset = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class cmd:
    vx = 0.0
    vy = 0.0
    dyaw = 0.0

class joy_data:
    Axis = [0] * 8
    button = 0
    button_last = 0

# ===================================================================
# 3. 核心工具函数
# ===================================================================

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

class CircularBuffer:
    def __init__(self, max_len, shape, device):
        self.buffer = deque(maxlen=max_len)
        self.shape = shape
        self.device = device
        self.reset()

    def add(self, obs):
        self.buffer.append(obs)

    def get_stacked(self):
        return torch.stack(list(self.buffer), dim=1)

    def reset(self):
        self.buffer.clear()
        for _ in range(self.buffer.maxlen):
            self.buffer.append(torch.zeros(self.shape, device=self.device))

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

def get_proprio_obs(data, model, cmd_vel, last_action, cfg: Sim2simCfg, joint_qpos_ids, joint_qvel_ids):


    obs_list = []
    
    # 1. Base Angular Velocity (3)
    quat = data.sensor('orientation').data[[1, 2, 3, 0]].astype(np.double)
    omega = data.sensor('angular-velocity').data.astype(np.double)
    r = R.from_quat(quat)
    obs_list.append(torch.tensor(omega * cfg.sim_config.ang_vel_scale, device=device).float())
    
    # 2. IMU RPY (2)
    euler = r.as_euler('xyz', degrees=False)
    obs_list.append(torch.tensor(euler[:2], device=device).float())
    
    # 3. Commands & Nav Placeholders (3 + 2 = 5)
    nav_padding = torch.zeros(3, device=device) # 3个 0: padding, delta_yaw, next_yaw
    obs_list.append(nav_padding)
    
    masked_cmd = torch.zeros(2, device=device)  # 2个 0: masked vx, vy
    obs_list.append(masked_cmd)
    
    # 4. Active Command (1)
    final_vx = -joy_data.Axis[1] * 2.0 + cmd_vel.vx * 2.0
    final_vy = -joy_data.Axis[0] * 2.0 + cmd_vel.vy * 2.0
    final_dyaw = -joy_data.Axis[3] * 1.0 + cmd_vel.dyaw * 1.0
    # raw_axis_1 = joy_data.Axis[1]
    # if abs(raw_axis_1) < 0.1: # 死区阈值
    #     raw_axis_1 = 0.0
        
    # final_vx = -raw_axis_1 * 2.0 + cmd_vel.vx * 2.0
    
    # print(f"Vx Command: {final_vx:.4f}")


    
    cmd_tensor = torch.tensor([
        final_vx * cfg.sim_config.lin_vel_scale
    ], device=device).float() # 只取 vx
    obs_list.append(cmd_tensor)
    
    # 5. Env ID (2)
    env_id = torch.tensor([1.0, 0.0], device=device).float()
    obs_list.append(env_id)
    
    # 6. Dof Pos (12)
    curr_q = data.qpos[joint_qpos_ids]
    dof_pos_err = (torch.tensor(curr_q, device=device).float() - torch.tensor(cfg.robot_config.default_joint_angles, device=device).float()) * cfg.sim_config.dof_pos_scale
    obs_list.append(dof_pos_err)
    
    # 7. Dof Vel (12)
    curr_dq = data.qvel[joint_qvel_ids]
    dof_vel = torch.tensor(curr_dq, device=device).float() * cfg.sim_config.dof_vel_scale
    obs_list.append(dof_vel)
    
    # 8. Last Action (12)
    obs_list.append(last_action)
    
    # 9. Foot Contact (4)
    try:
        fc_binary, _ = get_foot_contact(data, model)
        fc_tensor = torch.tensor(fc_binary, device=device).float() - 0.5
        obs_list.append(fc_tensor)
        
    except Exception as e:
        print(f"Foot Contact Error: {e}, using placeholder.")
        obs_list.append(torch.tensor([-0.5] * 4, device=device).float())
    
    # 10. 拼接
    proprio = torch.cat(obs_list).view(1, -1)
    return proprio

def set_camera_fov(model, camera_name, horizontal_fov_deg, width, height):
    """
    自动将水平FOV转换为垂直FOV并设置给MuJoCo模型
    """
    # 1. 计算长宽比
    aspect_ratio = width / height
    
    # 2. 数学转换 (度 -> 弧度 -> 正切 -> 除以长宽比 -> 反正切 -> 度)
    horizontal_fov_rad = math.radians(horizontal_fov_deg)
    vertical_fov_rad = 2 * math.atan(math.tan(horizontal_fov_rad / 2) / aspect_ratio)
    vertical_fov_deg = math.degrees(vertical_fov_rad)
    
    # 3. 找到相机ID并修改
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if cam_id != -1:
        # MuJoCo 的 model.cam_fovy 存储的是垂直FOV
        model.cam_fovy[cam_id] = vertical_fov_deg
        print(f"📷 [Camera Setup] '{camera_name}': H_FOV={horizontal_fov_deg}° -> V_FOV={vertical_fov_deg:.2f}° (Aspect={aspect_ratio:.2f})")
    else:
        print(f"⚠️ Warning: Camera '{camera_name}' not found!")

def process_depth(renderer, data, cfg):
    try:
        camera_id = mujoco.mj_name2id(renderer._model, mujoco.mjtObj.mjOBJ_CAMERA, cfg.sim_config.camera_name)
    except:
        camera_id = 0
    renderer.update_scene(data, camera=camera_id)
    renderer.enable_depth_rendering()
    depth_img = renderer.render()
    renderer.disable_depth_rendering()
    depth_t = torch.from_numpy(depth_img.copy()).float().to(device)
    depth_t = torch.clip(depth_t, cfg.sim_config.near_clip, cfg.sim_config.far_clip)
    depth_t = depth_t.unsqueeze(0).unsqueeze(0)
    depth_t = F.interpolate(
        depth_t, 
        size=(Sim2simCfg.sim_config.camera_height_resized, Sim2simCfg.sim_config.camera_width_resized), # (58, 87)
        mode='bicubic', 
        align_corners=False
    )
    depth_t = (depth_t - cfg.sim_config.near_clip) / (cfg.sim_config.far_clip - cfg.sim_config.near_clip)
    depth_t = 1.0 - depth_t
    return depth_t.squeeze(1)

# ===================================================================
# 4. 主仿真循环 (Run Mujoco)
# ===================================================================

def run_mujoco(cfg: Sim2simCfg):
    global done, reset, paused
    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
    data = mujoco.MjData(model)
    model.opt.timestep = cfg.sim_config.dt
    renderer = mujoco.Renderer(model, width=cfg.sim_config.camera_width_original, height=cfg.sim_config.camera_height_original)
    set_camera_fov(
        model=model,
        camera_name=cfg.sim_config.camera_name, 
        horizontal_fov_deg=87, 
        width=cfg.sim_config.camera_width_original, # 106
        height=cfg.sim_config.camera_height_original # 60
    )

    print("Loading Models...")
    # Base JIT
    base_model = torch.jit.load(cfg.sim_config.base_jit_path, map_location=device)
    base_model.eval()
    estimator = base_model.estimator.estimator
    hist_encoder = base_model.actor.history_encoder
    actor = base_model.actor.actor_backbone

    mock_env = MockEnvCfg()
    backbone = DepthOnlyFCBackbone58x87(None,32,512) 
    vision_model = RecurrentDepthBackbone(backbone, None).to(device)
    vision_weights = torch.load(cfg.sim_config.vision_weight_path, map_location=device)
    vision_model.load_state_dict(vision_weights.get('depth_encoder_state_dict', vision_weights))
    vision_model.eval()
    print("Models Loaded.")
    target_joint_names = [
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint", 
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"
    ]
    
    print("Mapping Joints...")
    joint_qpos_ids = []
    joint_qvel_ids = []
    
    for name in target_joint_names:
        # 1. 查找关节对象 ID
        j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if j_id == -1:
            print(f"[Error] Joint '{name}' not found in XML! Please check names.")
            return

        # 2. 查找该关节在 qpos 和 qvel 数组中的起始地址
        qpos_addr = model.jnt_qposadr[j_id]
        qvel_addr = model.jnt_dofadr[j_id]
        
        joint_qpos_ids.append(qpos_addr)
        joint_qvel_ids.append(qvel_addr)
        
        print(f"  {name}: qpos_idx={qpos_addr}, qvel_idx={qvel_addr}")
        
    joint_qpos_ids = np.array(joint_qpos_ids)
    joint_qvel_ids = np.array(joint_qvel_ids)
    # -----------------------------------------------------------------------------------

    obs_history = CircularBuffer(cfg.sim_config.history_len, (1, cfg.sim_config.n_proprio), device)
    last_action = torch.zeros(12, device=device)
    target_q = cfg.robot_config.default_joint_angles.copy()
    count_lowlevel = 0
    
    # 预填充
    # zero_obs = torch.zeros(1, cfg.sim_config.n_proprio, device=device)
    # for _ in range(cfg.sim_config.history_len): obs_history.add(zero_obs)

    def reset_sim():
        mujoco.mj_resetData(model, data)
        data.qpos[2] = 0.25
        
        # 使用映射索引来重置关节角度
        # data.qpos[joint_qpos_ids] 只写入这12个位置，跳过 Screw/Motor 等
        data.qpos[joint_qpos_ids] = cfg.robot_config.default_joint_angles
        
        mujoco.mj_forward(model, data)
        for _ in range(20): mujoco.mj_step(model, data)
        
        obs_history.reset()
        if hasattr(vision_model, "reset"): vision_model.reset()
        last_action[:] = 0

    reset_sim()

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        while viewer.is_running() and not done:
            if paused:
                time.sleep(0.1); viewer.sync(); continue
            if reset:
                reset = False; reset_sim()

            step_start = time.time()

            # --- Policy Loop (50Hz) ---
            if count_lowlevel % cfg.sim_config.decimation == 0:
                with torch.no_grad():
                    # 1. Proprioception (传入 indices)
                    proprio = get_proprio_obs(data, model, cmd, last_action, cfg, joint_qpos_ids, joint_qvel_ids)
                    
                    # 2. Vision & Yaw
                    depth_t = process_depth(renderer, data, cfg)
                    vis_out = vision_model(depth_t, proprio)
                    depth_latent = vis_out[:, :-2]
                    visual_yaw = vis_out[:, -2:]
                    proprio[:, 6:8] = visual_yaw * 1.5
                    
                    # 3. History
                    obs_history.add(proprio)
                    hist_input = obs_history.get_stacked()
                    activation = nn.ELU()
                    hist_latent = hist_encoder(activation, hist_input).view(1, -1)

                    # 4. Estimator & Full Obs
                    est_vel = estimator(proprio)
                    full_obs = torch.cat([proprio, depth_latent, est_vel, hist_latent], dim=1)
                    
                    # 5. Actor
                    raw_action = actor(full_obs)[0]
                    last_action = raw_action
                    
                    action_np = raw_action.cpu().numpy()
                    action_np = np.clip(action_np, -cfg.sim_config.clip_actions, cfg.sim_config.clip_actions)
                    target_q = action_np * cfg.sim_config.action_scale + cfg.robot_config.default_joint_angles

            # --- Physics Loop (200Hz) ---
            # [关键] 使用映射索引读取状态
            q_active = data.qpos[joint_qpos_ids]
            dq_active = data.qvel[joint_qvel_ids]
            
            # PD Control
            tau = cfg.robot_config.kps * (target_q - q_active) - cfg.robot_config.kds * dq_active
            tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)
            
            if len(data.ctrl) >= 12:
                data.ctrl[:12] = tau
            
            mujoco.mj_step(model, data)

            if count_lowlevel % 15 == 0:
                viewer.sync()
            
            count_lowlevel += 1
            dt_sleep = cfg.sim_config.dt - (time.time() - step_start)
            if dt_sleep > 0: time.sleep(dt_sleep)

    done = True

# ===================================================================
# 5. 手柄循环
# ===================================================================
def joy_loop():
    global done
    pygame.init()
    pygame.joystick.init()
    js = None
    while not done:
        pygame.event.pump()
        if pygame.joystick.get_count() > 0 and js is None:
            try:
                js = pygame.joystick.Joystick(0)
                js.init()
            except: js = None
        if js:
            try:
                joy_data.Axis[1] = js.get_axis(1)
                joy_data.Axis[0] = js.get_axis(0)
                joy_data.Axis[3] = js.get_axis(3)
            except: js = None
        time.sleep(0.01)

if __name__ == '__main__':
    cfg = Sim2simCfg()
    t1 = threading.Thread(target=run_mujoco, args=(cfg,))
    t2 = threading.Thread(target=joy_loop)
    t1.start(); t2.start()
    t1.join(); t2.join()