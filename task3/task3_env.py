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
    time.sleep(0.01) 

# ===================================================================
# 1. RL 与环境参数配置类 (Task 3 Whole-Body 专属配置)
# ===================================================================
class Task3Config:
    num_envs = 1024
    device = "cuda:0"
    sim_dt = 0.005
    decimation = 4
    max_episode_length = 1000

    num_observations = 310 
    num_actions = 0  
    action_scale = 0.25
    
    target_height = 0.75
    fall_height = 0.52       
    
    # ==============================================================
    # 极限指令边界与协同机制
    # ==============================================================
    # 扩大边界，激发极限冲刺与急弯，倒逼全身协同
    cmd_vx_range = [-0.3, 1.2]   # 解锁 1.2m/s 冲刺
    cmd_vy_range = [-0.2, 0.2]
    cmd_wz_range = [-0.8, 0.8]   # 解锁 0.8rad/s 急弯
    
    resample_command_steps = 200 
    cmd_smoothing_factor = 0.05  

    # 非对称滤波与软解冻课程
    ema_alpha_legs = 0.5         # 下肢求稳
    ema_alpha_arms = 0.2         # 上肢求快（敏捷代偿）
    arm_unlock_steps = 1000000   # 100万步内逐渐交还上肢控制权

    # ==============================================================
    # Omni-622 全身协同奖励架构
    # ==============================================================
    # 1. 主任务层 
    w_tracking_lin = 0.15    # [回调] 退回 0.15 的稳定值，不逼迫机器人强行冲刺
    w_tracking_yaw = 0.10    # [回调]
    w_air_time = 0.10        # [大砍] 从 0.40 暴降至 0.10！彻底抹杀“踢正步”和“僵尸高跷”
    w_clearance = 0.015      # [恢复] 重新拉高离地间隙奖励，逼迫机器人“弯曲膝盖”把脚抬起来
    w_z_vel_penalty = 0.03   
    
    # 2. 躯干稳定层
    w_base_ang_vel = 0.015   # [回调] 恢复对躯干晃动的适度约束
    w_upright = 0.03         
    
    # 3. 关节姿态层 
    w_default_pos = 0.04     # [微调] 稍微增加一点对默认微曲站姿的偏好，修复杂技步态
    w_arm_cross = 0.05       
    
    # 4. 动力学协同层 
    w_arm_leg_sync = 0.15    # [保持] 继续鼓励真实的手脚反相协同
    
    # 5. 生存、安全与平顺层
    w_alive = 0.01           # [恢复] 适度给予生存奖励，降低高动态下的死亡率
    w_joint_limit = 0.02     
    w_action_smooth = 0.0005 # [大幅下降] 降低平滑惩罚，解放手脚，允许快速交叉步和急转弯
    w_foot_slip = 0.02       
    w_energy = 0.0002        # [断崖下降] 将能耗惩罚降至极低，告诉机器人：“别省力气，大胆用双腿匀称地跑！”
    
    # 6. AMP 风格层与静止层
    w_amp_style = 0.05       
    w_stand_still = 0.02
    rew_fall = -1.0          # 极刑事件

    sigma_v = 4.0
    sigma_w = 4.0
    sigma_z = 10.0
    deadband_height = 0.05

# ===================================================================
# 2. 场景定义
# ===================================================================
@configclass
class G1SceneCfg(InteractiveSceneCfg):
    num_envs: int = Task3Config.num_envs
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
# 3. 极限协同 AMP 动捕数据加载器
# ===================================================================
class AMPMotionManager:
    def __init__(self, device: str, target_dim: int):
        self.device = device
        self.target_dim = target_dim
        # 加载包含 1.2m/s 冲刺与 0.8rad/s 急弯的数据集
        motion_file = "/home/lw/IsaacLab/tutorials/03_humanoid_basics/g1_extreme_omni.pt"
        
        probe(f"[AMP] 尝试加载极限动捕序列: {motion_file}")
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
            probe(f"[AMP] 维度完美匹配 -> 形状: {self.amass_pos.shape}")
        except Exception as e:
            print("\n❌ [AMP 加载崩溃]", flush=True)
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
# 4. G1 全身协同环境类
# ===================================================================
class G1WholeBodyEnv(gym.Env):
    def __init__(self, cfg: Task3Config):
        print("\n" + "="*50, flush=True)
        probe("🚀 全身协同核心环境开始初始化...")
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
            
            # 关节精准分组，支撑上下肢解耦
            self.foot_body_ids = self.robot.find_bodies(".*_ankle_.*")[0]
            self.leg_joint_ids = self.robot.find_joints(".*_hip_.*|.*_knee_.*|.*_ankle_.*")[0]
            self.arm_joint_ids = self.robot.find_joints(".*_shoulder_.*|.*_elbow_.*|.*_wrist_.*")[0]
            self.waist_joint_ids = self.robot.find_joints("waist_.*")[0]

            # 非对称 EMA 滤波矩阵
            self.ema_alpha_tensor = torch.full((self.cfg.num_envs, self.cfg.num_actions), self.cfg.ema_alpha_legs, device=self.device)
            self.ema_alpha_tensor[:, self.arm_joint_ids] = self.cfg.ema_alpha_arms
            self.ema_alpha_tensor[:, self.waist_joint_ids] = self.cfg.ema_alpha_arms
            
            # 松弛惩罚权重 (手臂和腰部允许更大宽容度)
            self.joint_slack_weights = torch.ones(self.cfg.num_actions, device=self.device)
            self.joint_slack_weights[self.arm_joint_ids] = 0.05 
            self.joint_slack_weights[self.waist_joint_ids] = 0.1 

            # 初始化缓存
            self.last_action = torch.zeros((cfg.num_envs, self.cfg.num_actions), device=self.device)
            self.action_history = torch.zeros((cfg.num_envs, 3, self.cfg.num_actions), device=self.device)
            self.episode_length_buf = torch.zeros(cfg.num_envs, dtype=torch.long, device=self.device)
            self.global_step = 0
            
            self.last_base_vel = torch.zeros((cfg.num_envs, 3), device=self.device)
            self.last_w_z = torch.zeros(cfg.num_envs, device=self.device) # 角动量代理缓存
            self.phase = torch.zeros(cfg.num_envs, device=self.device)
            self.step_freq = 1.5 
            
            self.target_cmd = torch.zeros((cfg.num_envs, 3), device=self.device)
            self.smoothed_cmd = torch.zeros((cfg.num_envs, 3), device=self.device)
            
            self.amp_manager = AMPMotionManager(self.device, target_dim=self.cfg.num_actions)
            
            print("="*50 + "\n", flush=True)
            
        except Exception as e:
            print("\n❌❌❌ [初始化致命异常捕获] ❌❌❌", flush=True)
            traceback.print_exc()
            sys.exit(1)

    def _sample_commands(self, num_samples: int) -> torch.Tensor:
        cmds = torch.zeros((num_samples, 3), device=self.device)
        cmds[:, 0] = torch.empty(num_samples, device=self.device).uniform_(*self.cfg.cmd_vx_range)
        cmds[:, 1] = torch.empty(num_samples, device=self.device).uniform_(*self.cfg.cmd_vy_range)
        cmds[:, 2] = torch.empty(num_samples, device=self.device).uniform_(*self.cfg.cmd_wz_range)
        is_zero = torch.rand(num_samples, device=self.device) < 0.1
        cmds[is_zero] = 0.0
        return cmds

    def reset(self, env_ids: torch.Tensor = None, options: Dict = None) -> Tuple[torch.Tensor, Dict]:
        if env_ids is None:
            env_ids = torch.arange(self.cfg.num_envs, device=self.device)
            
        try:
            ref_pos, ref_vel, ref_cmd = self.amp_manager.get_rsi_initial_state(env_ids)
            
            is_rsi = torch.rand(len(env_ids), device=self.device) < 0.8
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
            
            new_target_cmds = self._sample_commands(len(env_ids))
            self.target_cmd[env_ids] = torch.where(is_rsi_expanded, ref_cmd, new_target_cmds)
            self.smoothed_cmd[env_ids] = self.target_cmd[env_ids].clone() 
            
            self.last_action[env_ids] = 0.0
            self.action_history[env_ids] = 0.0
            self.last_base_vel[env_ids] = 0.0
            self.last_w_z[env_ids] = 0.0
            self.phase[env_ids] = 0.0
            self.episode_length_buf[env_ids] = 0
            
            self.scene.update(0.0)
            return self._compute_obs(), {}
            
        except Exception as e:
            traceback.print_exc()
            sys.exit(1)

    def step(self, action: torch.Tensor):
        try:
            # 动作空间软切换 (渐进式解封上半身)
            arm_activation = min(1.0, self.global_step / self.cfg.arm_unlock_steps)
            action[:, self.arm_joint_ids] *= arm_activation
            action[:, self.waist_joint_ids] *= arm_activation

            resample_mask = (self.episode_length_buf % self.cfg.resample_command_steps == 0) & (self.episode_length_buf > 0)
            resample_envs = resample_mask.nonzero().squeeze(-1)
            if len(resample_envs) > 0:
                self.target_cmd[resample_envs] = self._sample_commands(len(resample_envs))
            
            self.smoothed_cmd = self.cfg.cmd_smoothing_factor * self.target_cmd + (1.0 - self.cfg.cmd_smoothing_factor) * self.smoothed_cmd
            
            # 非对称 EMA 动作滤波
            current_action = self.ema_alpha_tensor * action + (1.0 - self.ema_alpha_tensor) * self.last_action
            self.last_action = current_action.clone()
            
            self.action_history = torch.roll(self.action_history, shifts=-1, dims=1)
            self.action_history[:, -1, :] = current_action
            
            target_pos = self.default_joint_pos + current_action * self.cfg.action_scale
            
            self.global_step += 1
            self.robot.set_joint_position_target(target_pos)
            
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
        
        in_contact = (self.contact.data.net_forces_w[:, :, 2] > 5.0).float()
        
        # 综合考虑平面移动与自转，准确识别运动状态
        is_moving = ((torch.norm(self.smoothed_cmd[:, :2], dim=-1) > 0.1) | 
                     (torch.abs(self.smoothed_cmd[:, 2]) > 0.1)).float()
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
        
        foot_z = self.robot.data.body_pos_w[:, self.foot_body_ids, 2] - self.scene.env_origins[:, 2].unsqueeze(-1)
        r_clearance = ((1.0 - in_contact) * torch.exp(-10.0 * torch.abs(foot_z - 0.05))).sum(dim=-1) * is_moving
        
        # ── 2. 躯干稳定与约束层 ──
        # 动态宽容机制全面覆盖：根据当前自转与侧移强度，动态豁免 Roll 惩罚，允许全场景压弯
        roll_tolerance = torch.clamp(0.5 * torch.abs(self.smoothed_cmd[:, 2]) + 1.0 * torch.abs(self.smoothed_cmd[:, 1]), min=0.0, max=1.0)
        p_base_ang = -torch.square(w_x) * (1.0 - roll_tolerance) - torch.square(w_y)
        
        r_upright = (1.0 - g_proj[:, 2]) / 2.0 
        
        # 静止惩罚重构：重点惩罚底盘打滑，允许微小的关节平衡震颤
        base_drift = torch.norm(self.robot.data.root_lin_vel_b[:, :2], dim=-1) + torch.abs(self.robot.data.root_ang_vel_b[:, 2])
        p_stand_still = -base_drift * is_standing

        # ── 3. 关节解耦与动力学协同层 ──
        pos_diff = self.robot.data.joint_pos - self.default_joint_pos
        slack = torch.zeros_like(pos_diff)
        slack[:, self.arm_joint_ids] = 0.3 
        slack[:, self.waist_joint_ids] = 0.2
        p_default_pos = -torch.sum(self.joint_slack_weights * torch.square(torch.clamp(torch.abs(pos_diff) - slack, min=0.0)), dim=-1)

        # 手臂安全软限位：针对手臂向躯干内收（穿模高危区）的 Roll 轴进行定向拦截
        # 假设 16, 20 为左右肩 Roll，向内收的值可能为负数，实施单侧极刑约束
        arm_cross_dist = torch.relu(-pos_diff[:, 16] - 0.1) + torch.relu(-pos_diff[:, 20] - 0.1)
        p_arm_cross = -arm_cross_dist

        # 显式上肢-下肢反相协同
        # 匹配对侧肢体：左髋Pitch(3) vs 右肩Pitch(19) | 右髋Pitch(9) vs 左肩Pitch(15)
        # 当左腿前摆(速度>0)，右臂也应前摆(速度>0)，同向速度乘积为正即给予奖励
        dq = self.robot.data.joint_vel
        r_arm_leg_sync = (dq[:, 3] * dq[:, 19] + dq[:, 9] * dq[:, 15]) * is_moving
        r_arm_leg_sync = torch.clamp(r_arm_leg_sync * 0.1, min=0.0, max=2.0) # 归一化量级

        # ── 4. 生存与硬件层 ──
        # 下调存活常量
        r_alive = torch.ones_like(v_x)
        
        limits = self.robot.data.soft_joint_pos_limits[0] 
        out_of_bounds = torch.maximum(self.robot.data.joint_pos - limits[:, 1], torch.zeros_like(self.robot.data.joint_pos)) + \
                        torch.maximum(limits[:, 0] - self.robot.data.joint_pos, torch.zeros_like(self.robot.data.joint_pos))
        p_joint_limit = -torch.sum(out_of_bounds, dim=-1)

        p_smooth = -torch.sum(torch.square(self.action_history[:, -1, :] - self.action_history[:, -2, :]), dim=-1)
        foot_vel_xy = self.robot.data.body_lin_vel_w[:, self.foot_body_ids, :2]
        p_slip = -torch.sum(torch.sum(torch.square(foot_vel_xy) * in_contact.unsqueeze(-1), dim=-1), dim=-1)
        
        tau = self.robot.data.applied_torque
        p_energy = -torch.sum(torch.abs(tau * dq), dim=-1) / 500.0

        r_style = self.amp_manager.compute_style_reward_proxy(self.robot.data.joint_pos, self.robot.data.joint_vel)
        
        # ── 奖励截断与合并 ──
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
            self.cfg.w_arm_cross * p_arm_cross +
            self.cfg.w_arm_leg_sync * r_arm_leg_sync + 
            
            self.cfg.w_alive * r_alive +
            self.cfg.w_joint_limit * p_joint_limit +
            self.cfg.w_action_smooth * p_smooth +  
            self.cfg.w_foot_slip * p_slip + 
            self.cfg.w_energy * p_energy +
            self.cfg.w_amp_style * r_style
        )
        
        continuous_rew = torch.clamp(continuous_rew, min=-1.0, max=1.0)
        
        is_fallen = (h_torso < self.cfg.fall_height) | (torch.norm(g_proj[:, :2], dim=-1) > 0.85)
        final_reward = torch.where(is_fallen, torch.full_like(continuous_rew, self.cfg.rew_fall), continuous_rew)
        
        terminated = is_fallen
        truncated = self.episode_length_buf >= self.cfg.max_episode_length
        
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
                "P_Arm_Cross": (self.cfg.w_arm_cross * p_arm_cross).mean().item(),
                "R_Arm_Leg_Sync": (self.cfg.w_arm_leg_sync * r_arm_leg_sync).mean().item(),
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
                "arm_activation": min(1.0, self.global_step / self.cfg.arm_unlock_steps),
                "harness_ratio": 0.0, 
                "global_step": self.global_step,
            },
        }
        return final_reward, terminated, truncated, info