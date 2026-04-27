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
# 1. RL 与环境参数配置类 (Task 2 Omni-Control 专属配置)
# ===================================================================
class Task2Config:
    num_envs = 1024
    device = "cuda:0"
    sim_dt = 0.005
    decimation = 4
    max_episode_length = 1000

    num_observations = 310 
    num_actions = 0  
    action_scale = 0.25
    ema_alpha = 0.5
    
    target_height = 0.75
    fall_height = 0.52       
    
    # ==============================================================
    # 全向指令边界与动态机制
    # ==============================================================
    # 三维摇杆指令边界：[vx_min, vx_max], [vy_min, vy_max], [wz_min, wz_max]
    cmd_vx_range = [-0.3, 0.5]
    cmd_vy_range = [-0.2, 0.2]
    cmd_wz_range = [-0.3, 0.3]
    
    resample_command_steps = 200 # 每 200 步 (约1.6秒) 强制重采样一次摇杆指令
    cmd_smoothing_factor = 0.05  # 一阶低通滤波系数，模拟真实摇杆推拉的阻尼感

    # ==============================================================
    # Omni-622 全向工业级奖励架构 
    # ==============================================================
    # 1. 主任务层
    w_tracking_lin = 0.15    # [保持] 0.15 的线性追踪非常健康
    w_tracking_yaw = 0.10    # [上调] 从 0.08 提至 0.10，逼迫网络更精准地消除 actual_wz 的漂移
    w_air_time = 0.40        # [上调] 从 0.30 提至 0.40，乘胜追击，让抬脚迈步更加干脆
    w_clearance = 0.01       # [保持] 
    w_z_vel_penalty = 0.03   # [保持] 
    
    # 2. 躯干稳定层
    w_base_ang_vel = 0.015   # [保持]
    w_upright = 0.03         # [保持]
    
    # 3. 关节姿态层
    w_default_pos = 0.08     # [保持] 
    
    # 4. 生存与安全层
    w_alive = 0.015          # [保持]
    w_joint_limit = 0.02     # [保持]
    
    # 5. 平顺效率层  
    w_action_smooth = 0.0015 # [保持]
    w_foot_slip = 0.02       # [保持]
    w_energy = 0.001         # [保持]
    
    # 6. AMP 风格层与静止层
    w_amp_style = 0.05       # [微调] 从 0.04 提至 0.05，用动捕数据帮它纠正转身时可能出现的畸形
    w_stand_still = 0.002    # [保持]

    rew_fall = -1.0          

    sigma_v = 4.0
    sigma_w = 4.0
    sigma_z = 10.0
    deadband_height = 0.05

# ===================================================================
# 2. 场景定义
# ===================================================================
@configclass
class G1SceneCfg(InteractiveSceneCfg):
    num_envs: int = Task2Config.num_envs
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
# 3. 全向 AMP 动捕数据加载器
# ===================================================================
class AMPMotionManager:
    def __init__(self, device: str, target_dim: int):
        self.device = device
        self.target_dim = target_dim
        # 加载任务 2 专属的全向动捕库
        motion_file = "/home/lw/IsaacLab/tutorials/03_humanoid_basics/g1_omni_walk.pt"
        
        probe(f"[AMP] 尝试加载全向动捕序列: {motion_file}")
        try:
            self.motion_data = torch.load(motion_file, map_location=device)
            self.amass_pos = self.motion_data["pos"]
            self.amass_vel = self.motion_data["vel"]
            # 提取附带的指令标签，用于 RSI 初始化对接
            self.amass_cmd = self.motion_data.get("cmd", torch.zeros((self.motion_data["num_frames"], 3), device=device))
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

    def get_rsi_initial_state(self, env_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回：位置, 速度, 对应的预设摇杆指令"""
        random_frames = torch.randint(0, self.num_frames, (len(env_ids),), device=self.device)
        return self.amass_pos[random_frames], self.amass_vel[random_frames], self.amass_cmd[random_frames]

    # === 修改 3: 将速度特征引入 AMP 距离计算 ===
    def compute_style_reward_proxy(self, current_pos: torch.Tensor, current_vel: torch.Tensor) -> torch.Tensor:
        pos_diff = current_pos.unsqueeze(1) - self.amass_pos.unsqueeze(0)
        vel_diff = current_vel.unsqueeze(1) - self.amass_vel.unsqueeze(0)
        
        # 综合位置与速度误差 (使用 0.1 作为速度缩放权重，平衡量纲)
        dist = torch.norm(pos_diff, dim=-1) + 0.1 * torch.norm(vel_diff, dim=-1)
        min_dist, _ = torch.min(dist, dim=1)
        return torch.exp(-2.0 * min_dist)

# ===================================================================
# 4. G1 全向机动环境类
# ===================================================================
class G1OmniEnv(gym.Env):
    def __init__(self, cfg: Task2Config):
        print("\n" + "="*50, flush=True)
        probe("🚀 全向机动核心环境开始初始化...")
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
            
            self.robot: Articulation = self.scene.articulations["robot"]
            self.contact: ContactSensor = self.scene.sensors["contact_forces"]
            
            self.default_joint_pos = self.robot.data.default_joint_pos.clone()
            self.cfg.num_actions = self.default_joint_pos.shape[1]
            probe(f"-> ✅ 动作维度锁定为: {self.cfg.num_actions}")
            
            self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(cfg.num_observations,))
            self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.cfg.num_actions,))
            
            probe("初始化 RL 与全向时序缓存矩阵")
            self.last_action = torch.zeros((cfg.num_envs, self.cfg.num_actions), device=self.device)
            self.action_history = torch.zeros((cfg.num_envs, 3, self.cfg.num_actions), device=self.device)
            self.episode_length_buf = torch.zeros(cfg.num_envs, dtype=torch.long, device=self.device)
            self.global_step = 0
            
            self.last_base_vel = torch.zeros((cfg.num_envs, 3), device=self.device)
            self.phase = torch.zeros(cfg.num_envs, device=self.device)
            self.step_freq = 1.5 
            
            # 指令缓存与平滑队列
            self.target_cmd = torch.zeros((cfg.num_envs, 3), device=self.device)
            self.smoothed_cmd = torch.zeros((cfg.num_envs, 3), device=self.device)
            
            self.foot_body_ids = self.robot.find_bodies(".*_ankle_.*")[0]
            self.leg_joint_ids = self.robot.find_joints(".*_hip_.*|.*_knee_.*|.*_ankle_.*")[0]
            self.joint_penalty_weights = torch.ones(self.cfg.num_actions, device=self.device)
            self.joint_penalty_weights[self.leg_joint_ids] = 0.05 

            self.amp_manager = AMPMotionManager(self.device, target_dim=self.cfg.num_actions)
            
            print("="*50 + "\n", flush=True)
            
        except Exception as e:
            print("\n❌❌❌ [初始化致命异常捕获] ❌❌❌", flush=True)
            traceback.print_exc()
            sys.exit(1)

    def _sample_commands(self, num_samples: int) -> torch.Tensor:
        """从预设边界内均匀采样三维摇杆指令"""
        cmds = torch.zeros((num_samples, 3), device=self.device)
        cmds[:, 0] = torch.empty(num_samples, device=self.device).uniform_(*self.cfg.cmd_vx_range)
        cmds[:, 1] = torch.empty(num_samples, device=self.device).uniform_(*self.cfg.cmd_vy_range)
        cmds[:, 2] = torch.empty(num_samples, device=self.device).uniform_(*self.cfg.cmd_wz_range)
        
        # 10% 概率强行挂机，训练静止站立能力
        is_zero = torch.rand(num_samples, device=self.device) < 0.1
        cmds[is_zero] = 0.0
        return cmds

    def reset(self, env_ids: torch.Tensor = None, options: Dict = None) -> Tuple[torch.Tensor, Dict]:
        if env_ids is None:
            env_ids = torch.arange(self.cfg.num_envs, device=self.device)
            
        if self.global_step == 0:
            probe(f"执行首轮双轨环境重置，数量: {len(env_ids)}")
            
        try:
            # 提取 RSI 初始态
            ref_pos, ref_vel, ref_cmd = self.amp_manager.get_rsi_initial_state(env_ids)
            
            # 80% RSI, 20% Zero-State
            is_rsi = torch.rand(len(env_ids), device=self.device) < 0.8
            is_rsi_expanded = is_rsi.unsqueeze(1)
            
            target_pos = torch.where(is_rsi_expanded, self.default_joint_pos[env_ids] + ref_pos, self.default_joint_pos[env_ids])
            target_vel = torch.where(is_rsi_expanded, ref_vel, torch.zeros_like(ref_vel))
            
            root_state = self.robot.data.default_root_state[env_ids].clone()
            root_state[:, 0:2] = self.scene.env_origins[env_ids, 0:2]
            root_state[:, 2] = self.cfg.target_height
            
            # 赋予 RSI 物理惯性初速度 (根据指令估算)
            root_state[:, 7:9] = torch.where(is_rsi_expanded[:, :2], ref_cmd[:, 0:2], torch.zeros_like(ref_cmd[:, 0:2]))
            root_state[:, 12] = torch.where(is_rsi, ref_cmd[:, 2], torch.zeros_like(ref_cmd[:, 2]))
            
            self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
            self.robot.write_joint_state_to_sim(target_pos, target_vel, env_ids=env_ids)
            
            # 初始化与刷新指令
            new_target_cmds = self._sample_commands(len(env_ids))
            self.target_cmd[env_ids] = torch.where(is_rsi_expanded, ref_cmd, new_target_cmds)
            self.smoothed_cmd[env_ids] = self.target_cmd[env_ids].clone() # 瞬间对齐，防止重置抽搐
            
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

    # ===================================================================
    # 核心物理步进函数 (修复了 RSI 初始化覆盖漏洞)
    # ===================================================================
    def step(self, action: torch.Tensor):
        if self.global_step == 0:
            probe(f"进入全向 Step 循环，Action 张量形状: {action.shape}")
            
        try:
            # 动态指令重采样 (避开第 0 步，保护 RSI 初始状态不被覆盖)
            resample_mask = (self.episode_length_buf % self.cfg.resample_command_steps == 0) & (self.episode_length_buf > 0)
            resample_envs = resample_mask.nonzero().squeeze(-1)
            if len(resample_envs) > 0:
                self.target_cmd[resample_envs] = self._sample_commands(len(resample_envs))
            
            # 指令低通滤波 (平滑摇杆输入)
            self.smoothed_cmd = self.cfg.cmd_smoothing_factor * self.target_cmd + (1.0 - self.cfg.cmd_smoothing_factor) * self.smoothed_cmd
            
            # 动作平滑滤波 (EMA)
            current_action = self.cfg.ema_alpha * action + (1.0 - self.cfg.ema_alpha) * self.last_action
            self.last_action = current_action.clone()
            
            self.action_history = torch.roll(self.action_history, shifts=-1, dims=1)
            self.action_history[:, -1, :] = current_action
            
            target_pos = self.default_joint_pos + current_action * self.cfg.action_scale
            
            self.global_step += 1
            self.robot.set_joint_position_target(target_pos)
            
            # 物理引擎多步解算
            for _ in range(self.cfg.decimation):
                self.scene.write_data_to_sim()
                self.sim.step()
            
            self.scene.update(self.dt)
            
            # 更新相位时钟与生存步数
            self.phase = (self.phase + self.dt * self.step_freq) % 1.0
            self.episode_length_buf += 1
            
            # 计算观测与奖励
            obs = self._compute_obs()
            rewards, terminated, truncated, info = self._compute_rewards()
            
            # 处理越界与超时重置
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
        
        base_lin_accel = (v_t - self.last_base_vel) / self.dt
        self.last_base_vel = v_t.clone()
        
        sin_phase = torch.sin(2 * math.pi * self.phase).unsqueeze(1)
        cos_phase = torch.cos(2 * math.pi * self.phase).unsqueeze(1)
        
        # 塞入平滑指令与目标指令，挤占原有的 Padding 空间
        obs_parts = [
            v_t, w_t, g_t, q_res, q_vel, 
            self.last_action, self.action_history.view(self.cfg.num_envs, -1), 
            contact_states, 
            self.smoothed_cmd, self.target_cmd,  # 3+3 维全向指令感知
            base_lin_accel, sin_phase, cos_phase
        ]
        obs = torch.cat(obs_parts, dim=-1)
        
        # 严格对齐 310 维，确保能对接训练脚本的 VecFrameStack (1550 维)
        pad_size = self.cfg.num_observations - obs.shape[1]
        if pad_size > 0:
            obs = torch.cat([obs, torch.zeros((self.cfg.num_envs, pad_size), device=self.device)], dim=-1)
        elif pad_size < 0:
            obs = obs[:, :self.cfg.num_observations]
            
        return torch.tanh(obs)

    # ===================================================================
    # 全向 622 工业级奖励架构 (彻底解决局部最优骗分)
    # ===================================================================
    def _compute_rewards(self):
        v_x = self.robot.data.root_lin_vel_b[:, 0]
        v_y = self.robot.data.root_lin_vel_b[:, 1]
        v_z = self.robot.data.root_lin_vel_b[:, 2]
        w_x = self.robot.data.root_ang_vel_b[:, 0]
        w_y = self.robot.data.root_ang_vel_b[:, 1]
        w_z = self.robot.data.root_ang_vel_b[:, 2]
        g_proj = self.robot.data.projected_gravity_b
        h_torso = self.robot.data.root_pos_w[:, 2]
        
        in_contact = (self.contact.data.net_forces_w[:, :, 2] > 5.0).float()
        
        # 构建指令状态门控：严格区分“移动”与“静止”
        cmd_norm = torch.norm(self.smoothed_cmd[:, :2], dim=-1)
        is_moving = (cmd_norm > 0.1).float()    # 指令 > 0.1m/s 视为运动相
        is_standing = 1.0 - is_moving           # 否则视为静止相
        
        # ── 1. 全向步态追踪层 (45%) ──
        # X-Y 二维直线追踪
        lin_err = torch.square(v_x - self.smoothed_cmd[:, 0]) + torch.square(v_y - self.smoothed_cmd[:, 1])
        r_tracking_lin = torch.exp(-self.cfg.sigma_v * lin_err)
        
        # Z 轴自转追踪
        yaw_err = torch.square(w_z - self.smoothed_cmd[:, 2])
        r_tracking_yaw = torch.exp(-self.cfg.sigma_w * yaw_err)
        
        p_z_vel = -2.0 * torch.abs(v_z) # 严惩起跳颠簸
        
        # 滞空分与离地间隙受到 is_moving 的绝对门控，断绝原地踏步骗分
        tracking_mul = torch.clamp(r_tracking_lin * r_tracking_yaw, 0.0, 1.0)
        single_contact = (in_contact.sum(dim=-1) > 0).float() 
        r_air_time = ((1.0 - in_contact) * self.dt).sum(dim=-1) * single_contact * tracking_mul * is_moving
        
        foot_z = self.robot.data.body_pos_w[:, self.foot_body_ids, 2] - self.scene.env_origins[:, 2].unsqueeze(-1)
        r_clearance = ((1.0 - in_contact) * torch.exp(-10.0 * torch.abs(foot_z - 0.05))).sum(dim=-1) * is_moving
        
        # ── 2. 躯干稳定与静止约束层 (30%) ──
        p_base_ang = -(torch.square(w_x) + torch.square(w_y)) # 仅惩罚 Roll 和 Pitch，放开 Yaw 以允许自转
        r_upright = (1.0 - g_proj[:, 2]) / 2.0 
        
        # 新增真正的“零指令静止惩罚”，当需要静止时，严厉惩罚所有关节速度
        p_stand_still = -torch.sum(torch.square(self.robot.data.joint_vel), dim=-1) * is_standing

        # ── 3. 关节姿态层 (10%) ──
        pos_diff = self.robot.data.joint_pos - self.default_joint_pos
        p_default_pos = -torch.sum(self.joint_penalty_weights * torch.square(pos_diff), dim=-1)

        # ── 4. 生存与安全层 (8%) ──
        # 移除时间偏置，改为纯粹的常数存活奖励
        r_alive = torch.ones_like(v_x)
        
        # 使用真实的软物理限位进行惩罚
        limits = self.robot.data.soft_joint_pos_limits[0] 
        out_of_bounds = torch.maximum(self.robot.data.joint_pos - limits[:, 1], torch.zeros_like(self.robot.data.joint_pos)) + \
                        torch.maximum(limits[:, 0] - self.robot.data.joint_pos, torch.zeros_like(self.robot.data.joint_pos))
        p_joint_limit = -torch.sum(out_of_bounds, dim=-1)

        # ── 5. 平顺效率层 (5%) ──
        # 移除了高度重复的 p_action_rate，保留纯净的动作差分平滑
        p_smooth = -torch.sum(torch.square(self.action_history[:, -1, :] - self.action_history[:, -2, :]), dim=-1)
        
        foot_vel_xy = self.robot.data.body_lin_vel_w[:, self.foot_body_ids, :2]
        p_slip = -torch.sum(torch.sum(torch.square(foot_vel_xy) * in_contact.unsqueeze(-1), dim=-1), dim=-1)
        
        tau = self.robot.data.applied_torque
        dq = self.robot.data.joint_vel
        p_energy = -torch.sum(torch.abs(tau * dq), dim=-1) / 500.0

        # ── 6. 动态 AMP 风格层 (2%) ──
        r_style = self.amp_manager.compute_style_reward_proxy(self.robot.data.joint_pos, self.robot.data.joint_vel)
        
        # ── 奖励加权总和 (移除硬截断 Clamp，释放全部梯度信号) ──
        continuous_rew = (
            self.cfg.w_tracking_lin * r_tracking_lin +
            self.cfg.w_tracking_yaw * r_tracking_yaw +
            self.cfg.w_air_time * r_air_time +
            self.cfg.w_z_vel_penalty * p_z_vel +
            self.cfg.w_clearance * r_clearance +
            
            self.cfg.w_base_ang_vel * p_base_ang +
            self.cfg.w_upright * r_upright +
            self.cfg.w_stand_still * p_stand_still +  
            
            self.cfg.w_default_pos * p_default_pos +
            
            self.cfg.w_alive * r_alive +
            self.cfg.w_joint_limit * p_joint_limit +
            
            self.cfg.w_action_smooth * p_smooth +  
            self.cfg.w_foot_slip * p_slip + 
            self.cfg.w_energy * p_energy +
            
            self.cfg.w_amp_style * r_style
        )
        
        # ── 离散极刑与事件判定 ──
        # 放宽侧向跌倒的横滚投影阈值 (从 0.7 调至 0.85)，允许大姿态的高速侧步
        is_fallen = (h_torso < self.cfg.fall_height) | (torch.norm(g_proj[:, :2], dim=-1) > 0.85)
        final_reward = torch.where(is_fallen, torch.full_like(continuous_rew, self.cfg.rew_fall), continuous_rew)
        
        terminated = is_fallen
        truncated = self.episode_length_buf >= self.cfg.max_episode_length
        
        # ── 遥测日志回传 (完美对齐当前奖励架构) ──
        info = {
            "reward_components": {
                "R_Track_Lin": (self.cfg.w_tracking_lin * r_tracking_lin).mean().item(),
                "R_Track_Yaw": (self.cfg.w_tracking_yaw * r_tracking_yaw).mean().item(),
                "R_Air_Time": (self.cfg.w_air_time * r_air_time).mean().item(),
                "P_Z_Vel": (self.cfg.w_z_vel_penalty * p_z_vel).mean().item(),
                "R_Clearance": (self.cfg.w_clearance * r_clearance).mean().item(),
                "P_Base_Ang": (self.cfg.w_base_ang_vel * p_base_ang).mean().item(),
                "R_Upright": (self.cfg.w_upright * r_upright).mean().item(),
                "P_Stand_Still": (self.cfg.w_stand_still * p_stand_still).mean().item(),
                "P_Default_Pos": (self.cfg.w_default_pos * p_default_pos).mean().item(),
                "R_Alive": (self.cfg.w_alive * r_alive).mean().item(),
                "P_Joint_Lim": (self.cfg.w_joint_limit * p_joint_limit).mean().item(),
                "P_Smooth": (self.cfg.w_action_smooth * p_smooth).mean().item(),
                "P_Slip": (self.cfg.w_foot_slip * p_slip).mean().item(),
                "P_Energy": (self.cfg.w_energy * p_energy).mean().item(),
                "R_AMP_Style": (self.cfg.w_amp_style * r_style).mean().item(),
            },
            "telemetry": {
                "cmd_vx_target": self.target_cmd[:, 0].mean().item(),
                "cmd_vy_target": self.target_cmd[:, 1].mean().item(),
                "cmd_wz_target": self.target_cmd[:, 2].mean().item(),
                "actual_vx": v_x.mean().item(),
                "actual_vy": v_y.mean().item(),
                "actual_wz": w_z.mean().item(),
                "fall_rate": is_fallen.float().mean().item(),
                "torso_height": h_torso.mean().item(),
                "harness_ratio": 0.0, 
                "global_step": self.global_step,
            },
        }
        return final_reward, terminated, truncated, info