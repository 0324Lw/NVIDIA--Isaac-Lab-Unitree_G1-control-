import argparse
import os
import torch
import numpy as np
import logging
from datetime import datetime

# 屏蔽底层物理引擎的冗余日志
logging.getLogger("isaaclab.assets.articulation").setLevel(logging.ERROR)

# ===================================================================
# 0. 启动物理引擎 (必须在所有网络库导入前完成)
# ===================================================================
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train G1 Humanoid AMP Policy (PPO)")
AppLauncher.add_app_launcher_args(parser)
args_cli, unknown_args = parser.parse_known_args()
args_cli.headless = True 
app_launcher = AppLauncher(args_cli)

simulation_app = app_launcher.app

# ===================================================================
# 1. 核心算法库导入
# ===================================================================
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from stable_baselines3.common.vec_env import VecNormalize, VecFrameStack
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed

# 导入最新定稿的环境与配置
from task1_env import Task1Config, G1HarnessEnv

# ===================================================================
# 2. 向量化环境桥接器
# ===================================================================
class CustomSb3VecEnvWrapper(VecEnv):
    def __init__(self, env):
        self.env = env
        super().__init__(env.cfg.num_envs, env.observation_space, env.action_space)
        self.ep_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.ep_lengths = np.zeros(self.num_envs, dtype=np.int32)

    def reset(self):
        obs, _ = self.env.reset()
        self.ep_returns[:] = 0.0
        self.ep_lengths[:] = 0
        return obs.cpu().numpy()

    def step_async(self, actions):
        self.actions = torch.tensor(actions, dtype=torch.float32, device=self.env.device)

    def step_wait(self):
        obs, rewards, terminated, truncated, info = self.env.step(self.actions)
        dones = (terminated | truncated).cpu().numpy()
        rewards_np = rewards.cpu().numpy()
        obs_np = obs.cpu().numpy()
        
        self.ep_returns += rewards_np
        self.ep_lengths += 1
        
        list_infos = [{} for _ in range(self.num_envs)]
        for idx in np.where(dones)[0]:
            list_infos[idx]["episode"] = {"r": self.ep_returns[idx], "l": self.ep_lengths[idx]}
            list_infos[idx]["terminal_observation"] = obs_np[idx]
            self.ep_returns[idx] = 0.0
            self.ep_lengths[idx] = 0

        # 提取底层环境的遥测与奖励日志
        if "telemetry" in info: list_infos[0]["telemetry"] = info["telemetry"]
        if "reward_components" in info: list_infos[0]["reward_components"] = info["reward_components"]
        return obs_np, rewards_np, dones, list_infos

    def close(self): self.env.close()
    def get_attr(self, name, indices=None): return [getattr(self.env, name, None)] * self.num_envs
    def set_attr(self, name, value, indices=None): setattr(self.env, name, value)
    def env_method(self, method_name, *method_args, indices=None, **method_kwargs): return [getattr(self.env, method_name)(*method_args, **method_kwargs)] * self.num_envs
    def env_is_wrapped(self, wrapper_class, indices=None): return [False] * self.num_envs

# ===================================================================
# 3. 强化版自适应 KL 调度器
# ===================================================================
class AdaptiveKLCallback(BaseCallback):
    def __init__(self, target_kl: float = 0.015, min_lr: float = 1e-5, max_lr: float = 1e-3):
        super().__init__()
        self.target_kl = target_kl
        self.min_lr = min_lr
        self.max_lr = max_lr

    def _on_step(self) -> bool: return True

    def _on_rollout_end(self):
        approx_kl = self.logger.name_to_value.get("train/approx_kl")
        if approx_kl is not None:
            current_lr = self.model.learning_rate
            new_lr = current_lr
            
            if approx_kl > self.target_kl * 1.5:
                new_lr = max(current_lr / 1.5, self.min_lr)
            elif approx_kl < self.target_kl / 1.5:
                new_lr = min(current_lr * 1.5, self.max_lr)
                
            if new_lr != current_lr:
                self.model.learning_rate = new_lr
                # 直接强制更新优化器内的参数组，确保下一轮 Rollout 生效
                for param_group in self.model.policy.optimizer.param_groups:
                    param_group["lr"] = new_lr
                
                self.logger.record("train/learning_rate", new_lr)

# ===================================================================
# 4. 工业级集成回调：周期性全量保存 + 遥测监控
# ===================================================================
class G1TrainingCallback(BaseCallback):
    def __init__(self, save_freq: int, save_path: str, verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq  
        self.save_path = save_path
        self.last_save_step = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        if len(infos) > 0:
            if "reward_components" in infos[0]:
                for key, val in infos[0]["reward_components"].items():
                    self.logger.record_mean(f"rewards/{key}", val)
            if "telemetry" in infos[0]:
                for key, val in infos[0]["telemetry"].items():
                    self.logger.record_mean(f"telemetry/{key}", val)
        
        # 每隔指定步数保存一次模型和归一化数据
        if (self.num_timesteps - self.last_save_step) >= self.save_freq:
            self.last_save_step = self.num_timesteps
            
            save_dir = os.path.join(self.save_path, f"checkpoint_{self.num_timesteps}")
            os.makedirs(save_dir, exist_ok=True)
            
            model_path = os.path.join(save_dir, "g1_model.zip")
            self.model.save(model_path)
            
            vec_normalize = self.model.get_vec_normalize_env()
            if vec_normalize is not None:
                vec_normalize.save(os.path.join(save_dir, "vec_normalize.pkl"))
            
            print(f"\n💾 [周期性备份] 步数: {self.num_timesteps} | 数据已保存至: {save_dir}", flush=True)
            
        return True

# ===================================================================
# 5. 主训练流
# ===================================================================
def main():
    set_random_seed(42)
    log_dir = f"./logs/g1_amp_{datetime.now().strftime('%m%d_%H%M')}"
    os.makedirs(log_dir, exist_ok=True)

    env_cfg = Task1Config()
    env_cfg.num_envs = 1024 
    base_env = G1HarnessEnv(env_cfg)
    
    # 1. 基础环境桥接
    env = CustomSb3VecEnvWrapper(base_env)
    
    # 2. 状态空间 5 帧堆叠 (VecFrameStack)
    # 输入维度将从 310 自动扩展为 1550 (310 * 5)
    env = VecFrameStack(env, n_stack=5)
    
    # 3. 状态与奖励滑动归一化 (放在 Stack 之后，确保时序序列被整体平滑)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # 网络架构配置：正交初始化与适配 1550 维输入的宽网络
    policy_kwargs = dict(
        activation_fn=torch.nn.ELU,
        net_arch=dict(pi=[512, 256, 128], vf=[512, 256, 128]),
        ortho_init=True,          # 正交初始化 (Orthogonal Initialization)
        log_std_init=-1.0         
    )

    ppo_kwargs = dict(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=128,              # Rollout buffer size = 1024 * 128 = 131072
        batch_size=32768,         # 梯度更新批大小
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,           # 限制策略更新幅度
        max_grad_norm=1.0,        # 🚨 梯度裁剪 (Gradient Clipping)
        policy_kwargs=policy_kwargs,
        tensorboard_log=log_dir,
        device="cuda:0",
        verbose=1                 
    )

    model = PPO(**ppo_kwargs)

    # 配置周期性保存：每 50,000,000 步
    training_callback = G1TrainingCallback(
        save_freq=50_000_000, 
        save_path=log_dir
    )
    
    # 挂载自适应学习率
    kl_callback = AdaptiveKLCallback(target_kl=0.015)

    print(f"\n🔥 [G1 强化训练启动] 开启 5 帧时序堆叠 | 目标总步数: 10 亿")
    print(f"环境自适应课程学习 (Harness) 已通过底层 global_step 自动触发")
    
    try:
        model.learn(
            total_timesteps=1_000_000_000,
            callback=[training_callback, kl_callback],
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n[WARN] 手动中断，正在保存最终模型...")
    finally:
        # 强制显式释放资源防止死锁
        model.save(os.path.join(log_dir, "final_model.zip"))
        env.save(os.path.join(log_dir, "vec_normalize.pkl"))
        
        del model
        env.close()
        simulation_app.close()
        os._exit(0)

if __name__ == "__main__":
    main()