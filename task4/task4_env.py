import torch
import numpy as np
import gymnasium as gym
from typing import Tuple, Dict, Any
import sys
import time
import math
import traceback

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass

def probe(msg: str):
    print(f"📍 [精细探针] {msg}", flush=True)
    time.sleep(0.01) 

# ===================================================================
# 1. RL 与环境参数配置类 (Task 4 Marathon & Sim2Real 专属配置)
# ===================================================================
class Task4Config:
    num_envs = 1024
    device = "cuda:0"
    sim_dt = 0.005
    decimation = 4
    max_episode_length = 2000  # 马拉松需要更长的回合来验证长距离稳定性

    num_observations = 310 
    num_actions = 0  
    action_scale = 0.25
    
    target_height = 0.75
    fall_height = 0.45       # 降低死亡高度，允许极限踉跄救场
    
    # ==============================================================
    # 纯粹的 X 轴定速马拉松指令边界
    # ==============================================================
    cmd_vx_range = [0.8, 2.0]    # 巡航速度 0.8 ~ 2.0 m/s
    cmd_vy_range = [0.0, 0.0]    # 封死侧向
    cmd_wz_range = [0.0, 0.0]    # 封死转向
    
    resample_command_steps = 300 
    cmd_smoothing_factor = 0.05  

    ema_alpha_legs = 0.5         
    ema_alpha_arms = 0.2         
    arm_unlock_steps = 1000000   

    # ==============================================================
    # Task 4: 结构化域随机化 (Domain Randomization) 边界
    # ==============================================================
    dr_motor_efficiency_range = [0.85, 1.0] # 电机发热/老化力矩衰减
    dr_mass_offset_range = [0.0, 5.0]       # 躯干附加载荷 (0~5kg)
    dr_friction_range = [0.6, 1.2]          # 地面摩擦系数波动
    dr_push_force_range = [10.0, 50.0]      # 脉冲推力 (N)
    dr_delay_steps_max = 3                  # 动作/观测异步延迟最大步数 (3*20ms = 60ms)
    dr_obs_noise_std = 0.05                 # IMU高频白噪声
    dr_drift_rate = 0.0005                  # 状态估计低频漂移率

    # ==============================================================
    # Omni-622全身协同与抗扰奖励架构 
    # ==============================================================
    # 1. 主任务层
    w_tracking_lin = 0.25    # [上调] 原0.15。拉高主任务权重，重赏马拉松巡航
    w_tracking_yaw = 0.15    
    w_air_time = 0.20        # [上调] 原0.10。鼓励离地迈步，防止贴地死锁
    w_clearance = 0.01       
    w_z_vel_penalty = 0.03   
    
    # 2. 躯干稳定层
    w_base_ang_vel = 0.005   # [巨幅下调] 原0.015。放开躯干封印，允许自然跑动中的重心起伏
    w_upright = 0.05         
    
    # 3. 关节姿态层 
    w_default_pos = 0.04     
    w_arm_cross = 0.05       
    
    # 4. 动力学协同与对称性
    w_arm_leg_sync = 0.05    
    w_symmetry = 0.10       
    
    # 5. 生存、安全与平顺层
    w_alive = 0.01           
    w_joint_limit = 0.05     
    w_action_smooth = 0.005  # [下调] 原0.01。让随机探索期平滑惩罚均值回落到 -0.008 左右
    w_foot_slip = 0.04       
    w_energy = 0.0005        # [断崖级下调] 原0.01。缩减 20 倍彻底解除“省电雕像”魔咒
    w_recovery = 0.05        
    
    # 6. AMP 风格层与静止层
    w_amp_style = 0.05       
    w_stand_still = 0.02
    
    rew_fall = -1.0          

    sigma_v = 4.0
    sigma_w = 4.0

# ===================================================================
# 2. 场景定义
# ===================================================================
@configclass
class G1SceneCfg(InteractiveSceneCfg):
    num_envs: int = Task4Config.num_envs
    env_spacing: float = 2.0
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/lw/IsaacLab/tutorials/03_humanoid_basics/g1.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True, 
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.8),
            joint_pos={".*": 0.0},
        ),
        actuators={
            "legs": ImplicitActuatorCfg(
                joint_names_expr=[".*_hip_.*", ".*_knee_.*", ".*_ankle_.*"],
                stiffness=150.0, damping=5.0,
            ),
            "upper_body": ImplicitActuatorCfg(
                joint_names_expr=["waist_.*", ".*_shoulder_.*", ".*_elbow_.*", ".*_wrist_.*"],
                stiffness=40.0, damping=2.0,
            ),
            "sensors": ImplicitActuatorCfg(
                joint_names_expr=["xl330_joint", "d455_joint"],
                stiffness=10000.0, damping=1000.0, 
            ),
        },
    )
    contact_forces: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_ankle_.*",
        update_period=0.0, history_length=3, debug_vis=False,
    )
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0))

# ===================================================================
# 3. 马拉松专属 AMP 动捕加载器
# ===================================================================
class AMPMotionManager:
    def __init__(self, device: str, target_dim: int):
        self.device = device
        self.target_dim = target_dim
        # 加载 Task 4 专属马拉松动捕数据
        motion_file = "/home/lw/IsaacLab/tutorials/03_humanoid_basics/g1_marathon_cpg.pt"
        
        try:
            self.motion_data = torch.load(motion_file, map_location=device)
            self.amass_pos = self.motion_data["pos"]
            self.amass_vel = self.motion_data["vel"]
            self.amass_cmd = self.motion_data.get("cmd", torch.zeros((self.motion_data["num_frames"], 3), device=device))
            self.num_frames = self.motion_data["num_frames"]
            
            raw_dim = self.amass_pos.shape[1]
            if raw_dim < target_dim:
                pad_size = target_dim - raw_dim
                self.amass_pos = torch.cat([self.amass_pos, torch.zeros((self.num_frames, pad_size), device=device)], dim=-1)
                self.amass_vel = torch.cat([self.amass_vel, torch.zeros((self.num_frames, pad_size), device=device)], dim=-1)
            elif raw_dim > target_dim:
                self.amass_pos = self.amass_pos[:, :target_dim]
                self.amass_vel = self.amass_vel[:, :target_dim]
        except Exception as e:
            traceback.print_exc()
            sys.exit(1)

    def get_rsi_initial_state(self, env_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        random_frames = torch.randint(0, self.num_frames, (len(env_ids),), device=self.device)
        return self.amass_pos[random_frames], self.amass_vel[random_frames], self.amass_cmd[random_frames]

    def compute_style_reward_proxy(self, current_pos: torch.Tensor, current_vel: torch.Tensor) -> torch.Tensor:
        pos_diff = current_pos.unsqueeze(1) - self.amass_pos.unsqueeze(0)
        vel_diff = current_vel.unsqueeze(1) - self.amass_vel.unsqueeze(0)
        dist = torch.norm(pos_diff, dim=-1) + 0.1 * torch.norm(vel_diff, dim=-1)
        min_dist, _ = torch.min(dist, dim=1)
        return torch.exp(-2.0 * min_dist)

# ===================================================================
# 4. G1 马拉松 Sim2Real 环境类 (集成 RMA 教师特权体系)
# ===================================================================
class G1MarathonEnv(gym.Env):
    def __init__(self, cfg: Task4Config):
        print("\n" + "="*60, flush=True)
        probe("🚀 开启 Task 4 马拉松 Sim2Real 极限抗扰环境...")
        try:
            self.cfg = cfg
            self.device = cfg.device
            self.dt = cfg.sim_dt * cfg.decimation
            
            sim_cfg = sim_utils.SimulationCfg(dt=cfg.sim_dt, device=self.device)
            self.sim = sim_utils.SimulationContext(sim_cfg)
            
            scene_cfg = G1SceneCfg()
            scene_cfg.num_envs = cfg.num_envs
            self.scene = InteractiveScene(scene_cfg)
            
            self.sim.reset()
            self.robot: Articulation = self.scene.articulations["robot"]
            self.contact: ContactSensor = self.scene.sensors["contact_forces"]
            
            self.default_joint_pos = self.robot.data.default_joint_pos.clone()
            self.cfg.num_actions = self.default_joint_pos.shape[1]
            
            self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(cfg.num_observations,))
            self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.cfg.num_actions,))
            
            self.foot_body_ids = self.robot.find_bodies(".*_ankle_.*")[0]
            self.leg_joint_ids = self.robot.find_joints(".*_hip_.*|.*_knee_.*|.*_ankle_.*")[0]
            self.arm_joint_ids = self.robot.find_joints(".*_shoulder_.*|.*_elbow_.*|.*_wrist_.*")[0]
            self.waist_joint_ids = self.robot.find_joints("waist_.*")[0]

            self.ema_alpha_tensor = torch.full((self.cfg.num_envs, self.cfg.num_actions), self.cfg.ema_alpha_legs, device=self.device)
            self.ema_alpha_tensor[:, self.arm_joint_ids] = self.cfg.ema_alpha_arms
            self.ema_alpha_tensor[:, self.waist_joint_ids] = self.cfg.ema_alpha_arms
            
            self.joint_slack_weights = torch.ones(self.cfg.num_actions, device=self.device)
            self.joint_slack_weights[self.arm_joint_ids] = 0.05 
            self.joint_slack_weights[self.waist_joint_ids] = 0.1 

            # RMA 域随机化缓存 (特权信息)
            self.dr_motor_efficiency = torch.ones((cfg.num_envs, self.cfg.num_actions), device=self.device)
            self.dr_friction = torch.ones(cfg.num_envs, device=self.device)
            self.dr_mass_offset = torch.zeros(cfg.num_envs, device=self.device)
            
            # 异步延迟队列与状态漂移
            self.action_delay_buffer = torch.zeros((cfg.num_envs, self.cfg.dr_delay_steps_max + 1, self.cfg.num_actions), device=self.device)
            self.obs_drift = torch.zeros((cfg.num_envs, 3), device=self.device) # [vx, vy, wz] 漂移
            
            # 扰动状态位
            self.is_pushed_flag = torch.zeros(cfg.num_envs, dtype=torch.bool, device=self.device)

            self.last_action = torch.zeros((cfg.num_envs, self.cfg.num_actions), device=self.device)
            self.action_history = torch.zeros((cfg.num_envs, 3, self.cfg.num_actions), device=self.device)
            self.episode_length_buf = torch.zeros(cfg.num_envs, dtype=torch.long, device=self.device)
            self.global_step = 0
            
            self.last_base_vel = torch.zeros((cfg.num_envs, 3), device=self.device)
            self.phase = torch.zeros(cfg.num_envs, device=self.device)
            self.step_freq = 1.5 
            
            self.target_cmd = torch.zeros((cfg.num_envs, 3), device=self.device)
            self.smoothed_cmd = torch.zeros((cfg.num_envs, 3), device=self.device)
            
            self.amp_manager = AMPMotionManager(self.device, target_dim=self.cfg.num_actions)
            # (在 __init__ 的张量初始化区域补充)
            # 每个环境在回合初采样固定延迟，代替白噪声抖动
            self.env_delay_steps = torch.zeros(cfg.num_envs, dtype=torch.long, device=self.device)
            # 物理推力张量：(num_envs, num_bodies, 3)
            self.external_force = torch.zeros((cfg.num_envs, self.robot.num_bodies, 3), device=self.device)
            
            print("="*60 + "\n", flush=True)
            
        except Exception as e:
            traceback.print_exc()
            sys.exit(1)

    def _sample_commands(self, num_samples: int) -> torch.Tensor:
        cmds = torch.zeros((num_samples, 3), device=self.device)
        # 马拉松：专注定速前行
        cmds[:, 0] = torch.empty(num_samples, device=self.device).uniform_(*self.cfg.cmd_vx_range)
        return cmds

    def reset(self, env_ids: torch.Tensor = None, options: Dict = None) -> Tuple[torch.Tensor, Dict]:
        if env_ids is None:
            env_ids = torch.arange(self.cfg.num_envs, device=self.device)
            
        try:
            # 结构化域随机化 (仅在 Reset 时刷新，保持一个回合内的物理一致性)
            num_reset = len(env_ids)
            self.dr_motor_efficiency[env_ids] = torch.empty((num_reset, self.cfg.num_actions), device=self.device).uniform_(*self.cfg.dr_motor_efficiency_range)
            self.dr_friction[env_ids] = torch.empty(num_reset, device=self.device).uniform_(*self.cfg.dr_friction_range)
            self.dr_mass_offset[env_ids] = torch.empty(num_reset, device=self.device).uniform_(*self.cfg.dr_mass_offset_range)
            
            # 清零延迟缓冲与漂移累计
            self.action_delay_buffer[env_ids] = 0.0
            self.obs_drift[env_ids] = 0.0
            self.is_pushed_flag[env_ids] = False

            # RSI 初始化
            ref_pos, ref_vel, ref_cmd = self.amp_manager.get_rsi_initial_state(env_ids)
            is_rsi = torch.rand(num_reset, device=self.device) < 0.8
            is_rsi_expanded = is_rsi.unsqueeze(1)
            
            target_pos = torch.where(is_rsi_expanded, self.default_joint_pos[env_ids] + ref_pos, self.default_joint_pos[env_ids])
            target_vel = torch.where(is_rsi_expanded, ref_vel, torch.zeros_like(ref_vel))
            
            root_state = self.robot.data.default_root_state[env_ids].clone()
            root_state[:, 0:2] = self.scene.env_origins[env_ids, 0:2]
            root_state[:, 2] = self.cfg.target_height
            
            root_state[:, 7:9] = torch.where(is_rsi_expanded[:, :2], ref_cmd[:, 0:2], torch.zeros_like(ref_cmd[:, 0:2]))
            root_state[:, 12] = torch.where(is_rsi, ref_cmd[:, 2], torch.zeros_like(ref_cmd[:, 2]))
            
            self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
            self.robot.write_joint_state_to_sim(target_pos, target_vel, env_ids=env_ids)
            
            new_target_cmds = self._sample_commands(num_reset)
            self.target_cmd[env_ids] = torch.where(is_rsi_expanded, ref_cmd, new_target_cmds)
            self.smoothed_cmd[env_ids] = self.target_cmd[env_ids].clone() 

            # 采用 Episode 固定延迟，模拟真实的时间相关性
            self.env_delay_steps[env_ids] = torch.randint(0, self.cfg.dr_delay_steps_max + 1, (len(env_ids),), device=self.device)
            self.action_delay_buffer[env_ids] = 0.0
            self.obs_drift[env_ids] = 0.0
            self.is_pushed_flag[env_ids] = False

            self.last_action[env_ids] = 0.0
            self.action_history[env_ids] = 0.0
            self.last_base_vel[env_ids] = 0.0
            self.phase[env_ids] = 0.0
            self.episode_length_buf[env_ids] = 0
            
            self.scene.update(0.0)
            return self._compute_obs(), {}
            
        except Exception as e:
            traceback.print_exc()
            sys.exit(1)

    def step(self, action: torch.Tensor):
        try:
            self.action_delay_buffer = torch.roll(self.action_delay_buffer, shifts=1, dims=1)
            self.action_delay_buffer[:, 0, :] = action
            
            delayed_action = self.action_delay_buffer[torch.arange(self.cfg.num_envs), self.env_delay_steps, :]

            arm_activation = min(1.0, self.global_step / self.cfg.arm_unlock_steps)
            delayed_action[:, self.arm_joint_ids] *= arm_activation
            delayed_action[:, self.waist_joint_ids] *= arm_activation

            resample_mask = (self.episode_length_buf % self.cfg.resample_command_steps == 0) & (self.episode_length_buf > 0)
            resample_envs = resample_mask.nonzero().squeeze(-1)
            if len(resample_envs) > 0:
                self.target_cmd[resample_envs] = self._sample_commands(len(resample_envs))
            self.smoothed_cmd = self.cfg.cmd_smoothing_factor * self.target_cmd + (1.0 - self.cfg.cmd_smoothing_factor) * self.smoothed_cmd
            
            current_action = self.ema_alpha_tensor * delayed_action + (1.0 - self.ema_alpha_tensor) * self.last_action
            self.last_action = current_action.clone()
            
            self.action_history = torch.roll(self.action_history, shifts=-1, dims=1)
            self.action_history[:, -1, :] = current_action

            degraded_action = current_action * self.dr_motor_efficiency
            target_pos = self.default_joint_pos + degraded_action * self.cfg.action_scale
            
            self.global_step += 1
            self.robot.set_joint_position_target(target_pos)
            
            is_vulnerable_phase = (self.phase < 0.05) | ((self.phase > 0.45) & (self.phase < 0.55))
            push_chance = (torch.rand(self.cfg.num_envs, device=self.device) < 0.05) & is_vulnerable_phase
            
            self.is_pushed_flag = push_chance.clone() 
            self.external_force[:] = 0.0  # 每步清空外力
            
            # 真正施加物理外力 (Wrench)，使用最新 API 替换废弃函数
            if push_chance.any():
                push_envs = push_chance.nonzero().squeeze(-1)
                push_f_x = torch.empty(len(push_envs), device=self.device).uniform_(-200.0, 200.0)
                push_f_y = torch.empty(len(push_envs), device=self.device).uniform_(-200.0, 200.0)
                
                self.external_force[push_envs, 0, 0] = push_f_x
                self.external_force[push_envs, 0, 1] = push_f_y
            
            # 使用 warning 中提示的 permanent_wrench_composer 最新接口
            if hasattr(self.robot, 'permanent_wrench_composer'):
                self.robot.permanent_wrench_composer.set_forces_and_torques(
                    forces=self.external_force, 
                    torques=torch.zeros_like(self.external_force)
                )
            else:
                # 兼容过渡期：如果未注册 composer，直接调用底层 PhysX 视图
                self.robot.root_physx_view.apply_forces_and_torques_at_pos(
                    self.external_force, 
                    torch.zeros_like(self.external_force),
                    is_global=False
                )

            # 物理引擎多步解算
            for _ in range(self.cfg.decimation):
                self.scene.write_data_to_sim()
                self.sim.step()
            
            self.scene.update(self.dt)
            self.phase = (self.phase + self.dt * self.step_freq) % 1.0
            self.episode_length_buf += 1
            
            obs = self._compute_obs()
            rewards, terminated, truncated, info = self._compute_rewards()
            
            resets = terminated | truncated
            reset_env_ids = resets.nonzero(as_tuple=False).squeeze(-1)
            if len(reset_env_ids) > 0:
                self.reset(reset_env_ids)
                
            return obs, rewards, terminated, truncated, info
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            sys.exit(1)

    def _compute_obs(self) -> torch.Tensor:
        v_t = self.robot.data.root_lin_vel_b.clone()
        w_t = self.robot.data.root_ang_vel_b.clone()
        g_t = self.robot.data.projected_gravity_b
        
        # ===============================================
        # 4. 状态估计漂移与观测噪声 (IMU Drift & Noise)
        # ===============================================
        # 累积低频随机游走漂移
        self.obs_drift += torch.randn_like(self.obs_drift) * self.cfg.dr_drift_rate
        
        # 给速度注入漂移和高频白噪声
        v_t[:, 0:2] += self.obs_drift[:, 0:2] + torch.randn_like(v_t[:, 0:2]) * self.cfg.dr_obs_noise_std
        w_t[:, 2] += self.obs_drift[:, 2] + torch.randn_like(w_t[:, 2]) * self.cfg.dr_obs_noise_std
        
        q_pos = self.robot.data.joint_pos 
        q_vel = self.robot.data.joint_vel 
        q_res = q_pos - self.default_joint_pos
        
        contact_forces = self.contact.data.net_forces_w[:, :, 2]
        contact_states = (contact_forces > 10.0).float().view(self.cfg.num_envs, -1)
        
        base_lin_accel = (v_t - self.last_base_vel) / self.dt
        self.last_base_vel = v_t.clone()
        
        sin_phase = torch.sin(2 * math.pi * self.phase).unsqueeze(1)
        cos_phase = torch.cos(2 * math.pi * self.phase).unsqueeze(1)
        
        obs_parts = [
            v_t, w_t, g_t, q_res, q_vel, 
            self.last_action, self.action_history.view(self.cfg.num_envs, -1), 
            contact_states, 
            self.smoothed_cmd, self.target_cmd, 
            base_lin_accel, sin_phase, cos_phase
        ]
        obs = torch.cat(obs_parts, dim=-1)
        
        pad_size = self.cfg.num_observations - obs.shape[1]
        if pad_size > 0:
            obs = torch.cat([obs, torch.zeros((self.cfg.num_envs, pad_size), device=self.device)], dim=-1)
        elif pad_size < 0:
            obs = obs[:, :self.cfg.num_observations]
            
        return torch.tanh(obs)
    
    def _compute_rewards(self):
        v_x = self.robot.data.root_lin_vel_b[:, 0]
        v_y = self.robot.data.root_lin_vel_b[:, 1]
        v_z = self.robot.data.root_lin_vel_b[:, 2]
        w_x = self.robot.data.root_ang_vel_b[:, 0]
        w_y = self.robot.data.root_ang_vel_b[:, 1]
        w_z = self.robot.data.root_ang_vel_b[:, 2]
        g_proj = self.robot.data.projected_gravity_b
        h_torso = self.robot.data.root_pos_w[:, 2]

        in_contact = (self.contact.data.net_forces_w[:, :, 2] > 15.0).float() # 接触判定提高到 15N 防误判
        is_moving = ((torch.norm(self.smoothed_cmd[:, :2], dim=-1) > 0.1)).float()
        is_standing = 1.0 - is_moving

        # ── 1. 全向步态追踪层 ──
        lin_err = torch.square(v_x - self.smoothed_cmd[:, 0]) + torch.square(v_y - self.smoothed_cmd[:, 1])
        r_tracking_lin = torch.exp(-self.cfg.sigma_v * lin_err)
        yaw_err = torch.square(w_z - self.smoothed_cmd[:, 2])
        r_tracking_yaw = torch.exp(-self.cfg.sigma_w * yaw_err)

        p_z_vel = -2.0 * torch.abs(v_z)

        tracking_mul = torch.clamp(r_tracking_lin * r_tracking_yaw, 0.0, 1.0)
        single_contact = (in_contact.sum(dim=-1) > 0).float()
        r_air_time = ((1.0 - in_contact) * self.dt).sum(dim=-1) * single_contact * tracking_mul * is_moving

        foot_z = self.robot.data.body_pos_w[:, self.foot_body_ids, 2] # 移除 -env_origins
        r_clearance = ((1.0 - in_contact) * torch.exp(-10.0 * torch.abs(foot_z - 0.05))).sum(dim=-1) * is_moving

        # ── 2. 躯干稳定层 ──
        roll_tolerance = torch.clamp(0.5 * torch.abs(self.smoothed_cmd[:, 2]) + 1.0 * torch.abs(self.smoothed_cmd[:, 1]), min=0.0, max=1.0)
        # 移除了 boundary_relax_factor，防止机器人通过主动倾斜骗取约束放宽
        p_base_ang = -(torch.square(w_x) * (1.0 - roll_tolerance) + torch.square(w_y))

        # 修复 Upright 逻辑：改用重力投影 XY 分量的指数惩罚
        r_upright = torch.exp(-5.0 * torch.sum(torch.square(g_proj[:, :2]), dim=-1))
        
        base_drift = torch.norm(self.robot.data.root_lin_vel_b[:, :2], dim=-1) + torch.abs(self.robot.data.root_ang_vel_b[:, 2])
        p_stand_still = -base_drift * is_standing

        # 受击恢复奖励 (Recovery Objective)
        # 受到推力后，如果能将速度误差控制在 0.2m/s 以内，给予高额奖励
        speed_error = torch.abs(v_x - self.smoothed_cmd[:, 0])
        r_recovery = torch.exp(-5.0 * speed_error) * self.is_pushed_flag.float()

        # ── 3. 关节解耦与协同 ──
        pos_diff = self.robot.data.joint_pos - self.default_joint_pos
        slack = torch.zeros_like(pos_diff)
        slack[:, self.arm_joint_ids] = 0.3
        slack[:, self.waist_joint_ids] = 0.2
        p_default_pos = -torch.sum(self.joint_slack_weights * torch.square(torch.clamp(torch.abs(pos_diff) - slack, min=0.0)), dim=-1)

        arm_cross_dist = torch.relu(-pos_diff[:, 16] - 0.1) + torch.relu(-pos_diff[:, 20] - 0.1)
        p_arm_cross = -arm_cross_dist

        dq = self.robot.data.joint_vel
        r_arm_leg_sync = (dq[:, 3] * dq[:, 19] + dq[:, 9] * dq[:, 15]) * is_moving
        r_arm_leg_sync = torch.clamp(r_arm_leg_sync * 0.1, min=0.0, max=2.0)

        sym_error = (
            torch.abs(torch.abs(self.robot.data.joint_pos[:, 3]) - torch.abs(self.robot.data.joint_pos[:, 9])) +
            torch.abs(torch.abs(self.robot.data.joint_pos[:, 6]) - torch.abs(self.robot.data.joint_pos[:, 12]))
        )
        p_symmetry = -sym_error * is_moving

        # ── 4. 生存与硬件层 ──
        # 增加存活奖励的时间比重 (模拟随回合长度递增)
        r_alive = 1.0 + (self.episode_length_buf.float() / self.cfg.max_episode_length)

        limits = self.robot.data.soft_joint_pos_limits[0]
        out_of_bounds = (
            torch.maximum(self.robot.data.joint_pos - limits[:, 1], torch.zeros_like(self.robot.data.joint_pos)) +
            torch.maximum(limits[:, 0] - self.robot.data.joint_pos, torch.zeros_like(self.robot.data.joint_pos))
        )
        p_joint_limit = -torch.sum(out_of_bounds, dim=-1)

        p_smooth = -torch.sum(torch.square(self.action_history[:, -1, :] - self.action_history[:, -2, :]), dim=-1)
        foot_vel_xy = self.robot.data.body_lin_vel_w[:, self.foot_body_ids, :2]
        p_slip = -torch.sum(torch.sum(torch.square(foot_vel_xy) * in_contact.unsqueeze(-1), dim=-1), dim=-1)

        # CoT (Cost of Transport) 修正
        # 公式: Power / (mass * g * max(vx, 0.2))
        robot_mass = 47.0 # G1 近似质量
        joint_power = torch.sum(torch.abs(self.robot.data.applied_torque * dq), dim=-1)
        cot = joint_power / (robot_mass * 9.81 * torch.clamp(v_x, min=0.2))
        p_energy = -cot

        # AMP 算力优化：每步仅随机采样 64 帧进行距离比对，防止显存/算力爆炸
        sample_idx = torch.randint(0, self.amp_manager.num_frames, (64,), device=self.device)
        sampled_pos = self.amp_manager.amass_pos[sample_idx]
        sampled_vel = self.amp_manager.amass_vel[sample_idx]
        
        pos_diff_amp = self.robot.data.joint_pos.unsqueeze(1) - sampled_pos.unsqueeze(0)
        vel_diff_amp = self.robot.data.joint_vel.unsqueeze(1) - sampled_vel.unsqueeze(0)
        dist = torch.norm(pos_diff_amp, dim=-1) + 0.1 * torch.norm(vel_diff_amp, dim=-1)
        min_dist, _ = torch.min(dist, dim=1)
        r_style = torch.exp(-2.0 * min_dist)

        # ── 奖励合并 ──
        continuous_rew = (
            self.cfg.w_tracking_lin    * r_tracking_lin +
            self.cfg.w_tracking_yaw    * r_tracking_yaw +
            self.cfg.w_air_time        * r_air_time +
            self.cfg.w_z_vel_penalty   * p_z_vel +
            self.cfg.w_clearance       * r_clearance +

            self.cfg.w_base_ang_vel    * p_base_ang +
            self.cfg.w_upright         * r_upright +
            self.cfg.w_stand_still     * p_stand_still +
            self.cfg.w_recovery        * r_recovery +

            self.cfg.w_default_pos     * p_default_pos +
            self.cfg.w_arm_cross       * p_arm_cross +
            self.cfg.w_arm_leg_sync    * r_arm_leg_sync +
            self.cfg.w_symmetry        * p_symmetry +

            self.cfg.w_alive          * r_alive +
            self.cfg.w_joint_limit    * p_joint_limit +
            self.cfg.w_action_smooth  * p_smooth +
            self.cfg.w_foot_slip      * p_slip +
            self.cfg.w_energy         * p_energy +
            self.cfg.w_amp_style      * r_style
        )

        continuous_rew = torch.clamp(continuous_rew, min=-1.0, max=1.0)

        # 跌倒判定收紧
        tilt_amount = torch.norm(g_proj[:, :2], dim=-1)
        is_fallen = (h_torso < 0.53) | (tilt_amount > 1.0)
        final_reward = torch.where(is_fallen, torch.full_like(continuous_rew, self.cfg.rew_fall), continuous_rew)

        terminated = is_fallen
        truncated = self.episode_length_buf >= self.cfg.max_episode_length

        info = {
            "reward_components": {
                "R_Track_Lin":    (self.cfg.w_tracking_lin    * r_tracking_lin).mean().item(),
                "R_Track_Yaw":    (self.cfg.w_tracking_yaw    * r_tracking_yaw).mean().item(),
                "R_Air_Time":     (self.cfg.w_air_time        * r_air_time).mean().item(),
                "P_Z_Vel":        (self.cfg.w_z_vel_penalty   * p_z_vel).mean().item(),
                "R_Clearance":    (self.cfg.w_clearance       * r_clearance).mean().item(),

                "P_Base_Ang":     (self.cfg.w_base_ang_vel    * p_base_ang).mean().item(),
                "R_Upright":      (self.cfg.w_upright         * r_upright).mean().item(),
                "P_Stand_Still":  (self.cfg.w_stand_still     * p_stand_still).mean().item(),
                "R_Recovery":     (self.cfg.w_recovery        * r_recovery).mean().item(),

                "P_Default_Pos":  (self.cfg.w_default_pos     * p_default_pos).mean().item(),
                "P_Arm_Cross":    (self.cfg.w_arm_cross       * p_arm_cross).mean().item(),
                "R_Arm_Leg_Sync": (self.cfg.w_arm_leg_sync    * r_arm_leg_sync).mean().item(),
                "P_Symmetry":     (self.cfg.w_symmetry        * p_symmetry).mean().item(),

                "R_Alive":        (self.cfg.w_alive          * r_alive).mean().item(),
                "P_Joint_Limit":  (self.cfg.w_joint_limit    * p_joint_limit).mean().item(),
                "P_Action_Smooth":(self.cfg.w_action_smooth  * p_smooth).mean().item(),
                "P_Foot_Slip":    (self.cfg.w_foot_slip      * p_slip).mean().item(),
                "P_Energy":       (self.cfg.w_energy         * p_energy).mean().item(),
                "R_Style":        (self.cfg.w_amp_style      * r_style).mean().item() 
            },
            "telemetry": {
                "actual_vx":       v_x.mean().item(),
                "actual_wz":       w_z.mean().item(),
                "fall_rate":       is_fallen.float().mean().item(),
                "arm_activation":  min(1.0, self.global_step / self.cfg.arm_unlock_steps),
                "global_step":     self.global_step,
            },
            "privileged_obs": {
                "motor_efficiency": self.dr_motor_efficiency.clone(),
                "friction":         self.dr_friction.clone(),
                "mass_offset":      self.dr_mass_offset.clone(),
                "is_pushed":        self.is_pushed_flag.clone(),
                "obs_drift_x":      self.obs_drift[:, 0].clone()
            }
        }

        return final_reward, terminated, truncated, info
