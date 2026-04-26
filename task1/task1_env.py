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
    """极其强硬的底层探针，强制刷新并休眠，防止 C++ 崩溃带走日志"""
    print(f"📍 [精细探针] {msg}", flush=True)
    time.sleep(0.01) # 稍微降低休眠时间加快运行

# ===================================================================
# 1. RL 与环境参数配置类
# ===================================================================
class Task1Config:
    num_envs = 1024
    device = "cuda:0"
    sim_dt = 0.005
    decimation = 4
    max_episode_length = 1000
    
    num_observations = 310 
    num_actions = 0  
    action_scale = 0.25
    ema_alpha = 0.5
    harness_initial_ratio = 0.8
    harness_decay_steps = 5000000
    target_vx = 0.5
    target_height = 0.75
    
    # ==============================================================
    # 工业级 6 层奖励架构权重
    # ==============================================================
    # 1. 步态与运动层 (45%)
    w_air_time = 0.10        # 略微下调：防止一味追求高抬腿
    w_forward = 0.08         
    w_cmd_err = 0.05         # 严杀侧滑和起跳
    w_clearance = 0.02       
    
    # 2. 躯干稳定层 (30%)
    w_base_ang_vel = 0.03    # 死锁躯干，解决走几步就侧倒
    w_upright = 0.03         
    w_com_stability = 0.04   # 逼迫重心始终在支撑脚上方
    
    # 3. 关节姿态层 (10%)
    w_default_pos = 0.12     # 彻底斩断僵尸臂，平举手将付出惨痛代价
    
    # 4. 生存与安全层 (8%)
    w_alive = 0.01           
    w_joint_limit = 0.02    
    
    # 5. 平顺效率层 (5%) 
    w_action_rate = 0.002   
    w_action_smooth = 0.003 
    w_foot_slip = 0.02      
    w_energy = 0.001        
    
    # 6. AMP 风格层 (2%)
    w_amp_style = 0.02       

    rew_fall = -1.0          # 跌倒极刑
    fall_height = 0.52       # 0.52m 安全生存阈值，允许重心下沉

    sigma_v = 4.0
    sigma_w = 4.0
    sigma_z = 10.0
    deadband_height = 0.05

# ===================================================================
# 2. 场景定义
# ===================================================================
@configclass
class G1SceneCfg(InteractiveSceneCfg):
    num_envs: int = Task1Config.num_envs
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
# 3. AMP 动捕数据加载器
# ===================================================================
class AMPMotionManager:
    def __init__(self, device: str, target_dim: int):
        self.device = device
        self.target_dim = target_dim
        motion_file = "/home/lw/IsaacLab/tutorials/03_humanoid_basics/g1_walk.pt"
        
        probe(f"[AMP] 尝试加载动捕序列: {motion_file}")
        try:
            self.motion_data = torch.load(motion_file, map_location=device)
            self.amass_pos = self.motion_data["pos"]
            self.amass_vel = self.motion_data["vel"]
            self.num_frames = self.motion_data["num_frames"]
            
            raw_dim = self.amass_pos.shape[1]
            probe(f"[AMP] 原始维度: {raw_dim} | 目标维度: {target_dim}")
            
            if raw_dim < target_dim:
                pad_size = target_dim - raw_dim
                self.amass_pos = torch.cat([self.amass_pos, torch.zeros((self.num_frames, pad_size), device=device)], dim=-1)
                self.amass_vel = torch.cat([self.amass_vel, torch.zeros((self.num_frames, pad_size), device=device)], dim=-1)
                probe(f"[AMP] 补零对齐完成 -> 形状: {self.amass_pos.shape}")
            elif raw_dim > target_dim:
                self.amass_pos = self.amass_pos[:, :target_dim]
                self.amass_vel = self.amass_vel[:, :target_dim]
                probe(f"[AMP] 截断对齐完成 -> 形状: {self.amass_pos.shape}")
            else:
                probe(f"[AMP] 维度完美匹配 -> 形状: {self.amass_pos.shape}")
        except Exception as e:
            print("\n❌ [AMP 加载崩溃]", flush=True)
            traceback.print_exc()
            sys.exit(1)

    def get_rsi_initial_state(self, env_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        random_frames = torch.randint(0, self.num_frames, (len(env_ids),), device=self.device)
        return self.amass_pos[random_frames], self.amass_vel[random_frames]

    def compute_style_reward_proxy(self, current_pos: torch.Tensor, current_vel: torch.Tensor) -> torch.Tensor:
        diff = current_pos.unsqueeze(1) - self.amass_pos.unsqueeze(0)
        dist = torch.norm(diff, dim=-1)
        min_dist, _ = torch.min(dist, dim=1)
        return torch.exp(-2.0 * min_dist)

# ===================================================================
# 4. G1 核心环境类
# ===================================================================
class G1HarnessEnv(gym.Env):
    def __init__(self, cfg: Task1Config):
        print("\n" + "="*50, flush=True)
        probe("🚀 核心环境开始初始化...")
        try:
            self.cfg = cfg
            self.device = cfg.device
            self.dt = cfg.sim_dt * cfg.decimation
            
            sim_cfg = sim_utils.SimulationCfg(dt=cfg.sim_dt, device=self.device)
            self.sim = sim_utils.SimulationContext(sim_cfg)
            
            scene_cfg = G1SceneCfg()
            scene_cfg.num_envs = cfg.num_envs
            self.scene = InteractiveScene(scene_cfg)
            
            probe("物理引擎实例化成功，执行第一次 Reset...")
            self.sim.reset()
            probe("第一次 Reset 完成！底层内存分配安全。")
            
            self.robot: Articulation = self.scene.articulations["robot"]
            self.contact: ContactSensor = self.scene.sensors["contact_forces"]
            
            probe("尝试读取 USD 关节总数 (num_joints)")
            nj = self.robot.num_joints
            probe(f"-> USD 关节总数 (num_joints) = {nj}")
            
            probe("尝试安全获取 default_joint_pos 张量")
            self.default_joint_pos = self.robot.data.default_joint_pos.clone()
            probe(f"-> 张量成功克隆，形状: {self.default_joint_pos.shape}")
            
            self.cfg.num_actions = self.default_joint_pos.shape[1]
            probe(f"-> ✅ 动作维度安全锁定为: {self.cfg.num_actions}")
            
            probe("尝试获取机器人物理质量")
            self.robot_mass = self.robot.root_physx_view.get_masses()[0].sum().item()
            probe(f"-> 质量获取成功: {self.robot_mass:.2f} kg")
            
            self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(cfg.num_observations,))
            self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.cfg.num_actions,))
            
            probe("初始化 RL 缓存矩阵")
            self.last_action = torch.zeros((cfg.num_envs, self.cfg.num_actions), device=self.device)
            self.action_history = torch.zeros((cfg.num_envs, 3, self.cfg.num_actions), device=self.device)
            self.episode_length_buf = torch.zeros(cfg.num_envs, dtype=torch.long, device=self.device)
            self.global_step = 0
            
            self.last_base_vel = torch.zeros((cfg.num_envs, 3), device=self.device)
            self.phase = torch.zeros(cfg.num_envs, device=self.device)
            self.step_freq = 1.5 
            
            probe("准备缓存脚底碰撞体索引")
            self.foot_body_ids = self.robot.find_bodies(".*_ankle_.*")[0]
            probe(f"-> 脚部 Body 索引获取成功: {self.foot_body_ids}")
            
            probe("准备防僵尸臂动态权重矩阵")
            self.leg_joint_ids = self.robot.find_joints(".*_hip_.*|.*_knee_.*|.*_ankle_.*")[0]
            self.joint_penalty_weights = torch.ones(self.cfg.num_actions, device=self.device)
            self.joint_penalty_weights[self.leg_joint_ids] = 0.05 
            probe(f"-> 成功锁定 {len(self.leg_joint_ids)} 个下肢关节，其余关节惩罚拉满")

            probe("准备初始化 AMP 管理器")
            self.amp_manager = AMPMotionManager(self.device, target_dim=self.cfg.num_actions)
            
            print("="*50 + "\n", flush=True)
            
        except Exception as e:
            print("\n❌❌❌ [初始化致命异常捕获] ❌❌❌", flush=True)
            traceback.print_exc()
            sys.exit(1)

    def reset(self, env_ids: torch.Tensor = None, options: Dict = None) -> Tuple[torch.Tensor, Dict]:
        if env_ids is None:
            env_ids = torch.arange(self.cfg.num_envs, device=self.device)
            
        if self.global_step == 0:
            probe(f"执行首轮环境重置，数量: {len(env_ids)}")
            
        try:
            ref_pos, ref_vel = self.amp_manager.get_rsi_initial_state(env_ids)
            target_pos = self.default_joint_pos[env_ids] + ref_pos
            
            root_state = self.robot.data.default_root_state[env_ids].clone()
            root_state[:, 0:2] = self.scene.env_origins[env_ids, 0:2]
            root_state[:, 2] = self.cfg.target_height
            
            self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
            self.robot.write_joint_state_to_sim(target_pos, ref_vel, env_ids=env_ids)
            
            self.last_action[env_ids] = 0.0
            self.action_history[env_ids] = 0.0
            self.last_base_vel[env_ids] = 0.0
            self.phase[env_ids] = 0.0
            self.episode_length_buf[env_ids] = 0
            
            self.scene.update(0.0)
            return self._compute_obs(), {}
            
        except Exception as e:
            print("\n❌❌❌ [Reset 阶段崩溃] ❌❌❌", flush=True)
            traceback.print_exc()
            sys.exit(1)

    def step(self, action: torch.Tensor):
        if self.global_step == 0:
            probe(f"进入 Step 循环，Action 张量形状: {action.shape}")
            
        try:
            current_action = self.cfg.ema_alpha * action + (1.0 - self.cfg.ema_alpha) * self.last_action
            self.last_action = current_action.clone()
            
            self.action_history = torch.roll(self.action_history, shifts=-1, dims=1)
            self.action_history[:, -1, :] = current_action
            
            target_pos = self.default_joint_pos + current_action * self.cfg.action_scale
            
            self.global_step += 1
            
            mean_vx = self.robot.data.root_lin_vel_b[:, 0].mean().item()
            perf_decay = min(max(mean_vx / self.cfg.target_vx, 0.0), 1.0)
            time_decay = min(1.0, self.global_step / self.cfg.harness_decay_steps)
            decay_progress = 0.8 * perf_decay + 0.2 * time_decay
            
            current_harness_ratio = self.cfg.harness_initial_ratio * (1.0 - decay_progress)
            
            lift_force = torch.zeros((self.cfg.num_envs, 1, 3), device=self.device)
            lift_force[:, 0, 2] = current_harness_ratio * self.robot_mass * 9.81
            self.robot.set_external_force_and_torque(forces=lift_force, torques=torch.zeros_like(lift_force), body_ids=[0])
            
            self.robot.set_joint_position_target(target_pos)
            
            for _ in range(self.cfg.decimation):
                self.scene.write_data_to_sim()
                self.sim.step()
            
            self.scene.update(self.dt)
            
            self.phase = (self.phase + self.dt * self.step_freq) % 1.0
            self.episode_length_buf += 1
            
            obs = self._compute_obs()
            rewards, terminated, truncated, info = self._compute_rewards(current_harness_ratio)
            
            resets = terminated | truncated
            reset_env_ids = resets.nonzero(as_tuple=False).squeeze(-1)
            if len(reset_env_ids) > 0:
                self.reset(reset_env_ids)
                
            return obs, rewards, terminated, truncated, info
            
        except Exception as e:
            print(f"\n❌❌❌ [Step 物理仿真崩溃 | 帧数: {self.global_step}] ❌❌❌", flush=True)
            traceback.print_exc()
            sys.exit(1)

    def _compute_obs(self) -> torch.Tensor:
        v_t = self.robot.data.root_lin_vel_b
        w_t = self.robot.data.root_ang_vel_b
        g_t = self.robot.data.projected_gravity_b
        
        q_pos = self.robot.data.joint_pos 
        q_vel = self.robot.data.joint_vel 
        q_res = q_pos - self.default_joint_pos
        
        contact_forces = self.contact.data.net_forces_w[:, :, 2]
        contact_states = (contact_forces > 10.0).float().view(self.cfg.num_envs, -1)
        
        cmd_vx = torch.full((self.cfg.num_envs, 1), self.cfg.target_vx, device=self.device)
        
        base_lin_accel = (v_t - self.last_base_vel) / self.dt
        self.last_base_vel = v_t.clone()
        
        sin_phase = torch.sin(2 * math.pi * self.phase).unsqueeze(1)
        cos_phase = torch.cos(2 * math.pi * self.phase).unsqueeze(1)
        
        obs_parts = [
            v_t, w_t, g_t, q_res, q_vel, 
            self.last_action, self.action_history.view(self.cfg.num_envs, -1), 
            contact_states, cmd_vx,
            base_lin_accel, sin_phase, cos_phase
        ]
        obs = torch.cat(obs_parts, dim=-1)
        
        pad_size = self.cfg.num_observations - obs.shape[1]
        if pad_size > 0:
            obs = torch.cat([obs, torch.zeros((self.cfg.num_envs, pad_size), device=self.device)], dim=-1)
        elif pad_size < 0:
            obs = obs[:, :self.cfg.num_observations]
            
        return torch.tanh(obs)

    def _compute_rewards(self, harness_ratio: float):
        v_x = self.robot.data.root_lin_vel_b[:, 0]
        v_y = self.robot.data.root_lin_vel_b[:, 1]
        v_z = self.robot.data.root_lin_vel_b[:, 2]
        w_x = self.robot.data.root_ang_vel_b[:, 0]
        w_y = self.robot.data.root_ang_vel_b[:, 1]
        w_z = self.robot.data.root_ang_vel_b[:, 2]
        g_proj = self.robot.data.projected_gravity_b
        h_torso = self.robot.data.root_pos_w[:, 2]
        
        in_contact = (self.contact.data.net_forces_w[:, :, 2] > 5.0).float()
        
        # ── 1. 步态与运动层 ──
        r_vel = torch.exp(-self.cfg.sigma_v * torch.square(v_x - self.cfg.target_vx))
        
        #  Z 轴垂直速度施加双倍严厉惩罚
        p_cmd_err = -(torch.abs(v_y) + 2.0 * torch.abs(v_z))
        
        # 只有单脚接触地面时，才给予另一只脚滞空奖励。双脚同时离地直接记 0 分
        vx_clamp = torch.clamp(v_x / self.cfg.target_vx, 0.0, 1.0).unsqueeze(-1)
        single_contact = (in_contact.sum(dim=-1) > 0).float() # 至少有一只脚踩实
        r_air_time = ((1.0 - in_contact) * self.dt * vx_clamp).sum(dim=-1) * single_contact
        
        foot_z = self.robot.data.body_pos_w[:, self.foot_body_ids, 2] - self.scene.env_origins[:, 2].unsqueeze(-1)
        r_clearance = ((1.0 - in_contact) * torch.exp(-10.0 * torch.abs(foot_z - 0.05))).sum(dim=-1)
        
        # ── 2. 躯干稳定层 ──
        p_base_ang = -(torch.square(w_x) + torch.square(w_y))
        r_heading = torch.exp(-self.cfg.sigma_w * torch.square(w_z))
        r_upright = (1.0 - g_proj[:, 2]) / 2.0 
        
        h_err = torch.clamp(torch.abs(h_torso - self.cfg.target_height) - self.cfg.deadband_height, min=0.0)
        r_height = torch.exp(-self.cfg.sigma_z * torch.square(h_err))
        
        r_com_stability = torch.exp(-torch.abs(v_y)) * (in_contact.sum(dim=-1) > 0).float()

        # ── 3. 关节姿态层 ──
        # 配合 0.12 的暴增权重，直接把僵尸臂打断
        pos_diff = self.robot.data.joint_pos - self.default_joint_pos
        p_default_pos = -torch.sum(self.joint_penalty_weights * torch.square(pos_diff), dim=-1)

        # ── 4. 生存与安全层 ──
        r_alive = self.episode_length_buf.float() / self.cfg.max_episode_length
        p_joint_limit = -torch.sum(torch.maximum(torch.abs(pos_diff) - 1.0, torch.zeros_like(pos_diff)), dim=-1)

        # ── 5. 平顺效率层 ──
        p_action_rate = -torch.sum(torch.square(self.last_action - self.action_history[:, -2, :]), dim=-1)
        p_smooth = -torch.sum(torch.square(self.action_history[:, -1, :] - self.action_history[:, -2, :]), dim=-1)
        
        foot_vel_xy = self.robot.data.body_lin_vel_w[:, self.foot_body_ids, :2]
        p_slip = -torch.sum(torch.sum(torch.square(foot_vel_xy) * in_contact.unsqueeze(-1), dim=-1), dim=-1)
        
        tau = self.robot.data.applied_torque
        dq = self.robot.data.joint_vel
        p_energy = -torch.sum(torch.abs(tau * dq), dim=-1) / 500.0

        # ── 6. AMP 风格层 ──
        r_style = self.amp_manager.compute_style_reward_proxy(self.robot.data.joint_pos, self.robot.data.joint_vel)
        
        # ── 加权总和 ──
        continuous_rew = (
            self.cfg.w_air_time * r_air_time +
            self.cfg.w_forward * r_vel +
            self.cfg.w_cmd_err * p_cmd_err +
            self.cfg.w_clearance * r_clearance +
            
            self.cfg.w_base_ang_vel * p_base_ang +
            self.cfg.w_upright * r_upright +
            self.cfg.w_com_stability * r_com_stability +
            
            self.cfg.w_default_pos * p_default_pos +
            
            self.cfg.w_alive * r_alive +
            self.cfg.w_joint_limit * p_joint_limit +
            
            self.cfg.w_action_rate * p_action_rate +
            self.cfg.w_action_smooth * p_smooth + 
            self.cfg.w_foot_slip * p_slip + 
            self.cfg.w_energy * p_energy +
            
            self.cfg.w_amp_style * r_style
        )
        
        step_reward = torch.clamp(continuous_rew, min=-0.5, max=0.5)
        
        # ── 离散极刑与事件判定 ──
        is_fallen = (h_torso < self.cfg.fall_height) | (torch.norm(g_proj[:, :2], dim=-1) > 0.7)
        final_reward = torch.where(is_fallen, torch.full_like(step_reward, self.cfg.rew_fall), step_reward)
        
        terminated = is_fallen
        truncated = self.episode_length_buf >= self.cfg.max_episode_length
        
        # ── 遥测日志回传 ──
        info = {
            "reward_components": {
                "R_Air_Time": (self.cfg.w_air_time * r_air_time).mean().item(),
                "R_Vel": (self.cfg.w_forward * r_vel).mean().item(),
                "P_Cmd_Err": (self.cfg.w_cmd_err * p_cmd_err).mean().item(),
                "R_Clearance": (self.cfg.w_clearance * r_clearance).mean().item(),
                "P_Base_Ang": (self.cfg.w_base_ang_vel * p_base_ang).mean().item(),
                "R_Upright": (self.cfg.w_upright * r_upright).mean().item(),
                "R_COM_Stab": (self.cfg.w_com_stability * r_com_stability).mean().item(),
                "P_Default_Pos": (self.cfg.w_default_pos * p_default_pos).mean().item(),
                "R_Alive": (self.cfg.w_alive * r_alive).mean().item(),
                "R_Height": (self.cfg.w_height * r_height).mean().item() if hasattr(self.cfg, 'w_height') else 0.0,
                "P_Smooth": (self.cfg.w_action_smooth * p_smooth).mean().item(),
                "P_Slip": (self.cfg.w_foot_slip * p_slip).mean().item(),
                "P_Energy": (self.cfg.w_energy * p_energy).mean().item(),
                "R_AMP_Style": (self.cfg.w_amp_style * r_style).mean().item(),
            },
            "telemetry": {
                "actual_vx": v_x.mean().item(),
                "harness_ratio": harness_ratio,
                "fall_rate": is_fallen.float().mean().item(),
                "torso_height": h_torso.mean().item(),
                "global_step": self.global_step,
            },
        }
        return final_reward, terminated, truncated, info