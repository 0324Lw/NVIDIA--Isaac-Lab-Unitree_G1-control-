import argparse
import os
import sys
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

parser = argparse.ArgumentParser(description="Train G1 Marathon Sim2Real (PPO)")
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
from stable_baselines3.common.logger import configure

# 导入 Task 4 专属的环境与配置
from task4_env import Task4Config, G1MarathonEnv

# ===================================================================
# 2. 向量化环境桥接器 (扩展支持特权信息的日志传递)
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

        if "telemetry" in info: list_infos[0]["telemetry"] = info["telemetry"]
        if "reward_components" in info: list_infos[0]["reward_components"] = info["reward_components"]
        # 提取特权观测信息，供 TensorBoard 监控环境的恶劣程度
        if "privileged_obs" in info: list_infos[0]["privileged_obs"] = info["privileged_obs"]
        
        return obs_np, rewards_np, dones, list_infos

    def close(self): self.env.close()
    def get_attr(self, name, indices=None): return [getattr(self.env, name, None)] * self.num_envs
    def set_attr(self, name, value, indices=None): setattr(self.env, name, value)
    def env_method(self, method_name, *method_args, indices=None, **method_kwargs): return [getattr(self.env, method_name)(*method_args, **method_kwargs)] * self.num_envs
    def env_is_wrapped(self, wrapper_class, indices=None): return [False] * self.num_envs

# ===================================================================
# 3. 自适应 KL 调度器
# ===================================================================
class AdaptiveKLCallback(BaseCallback):
    def __init__(self, target_kl: float = 0.02, min_lr: float = 1e-5, max_lr: float = 5e-4):
        super().__init__()
        self.target_kl = target_kl
        self.min_lr = min_lr
        self.max_lr = max_lr # 允许学习率飙升至 5e-4

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
                for param_group in self.model.policy.optimizer.param_groups:
                    param_group["lr"] = new_lr
                self.logger.record("train/learning_rate", new_lr)

# ===================================================================
# 4. 集成回调：加入域随机化的监控
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
            if "privileged_obs" in infos[0]:
                for key, val in infos[0]["privileged_obs"].items():
                    self.logger.record_mean(f"dr_states/{key}", val.float().mean().item())
        
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
    
    log_dir = f"./logs/g1_marathon_sim2real_{datetime.now().strftime('%m%d_%H%M')}"
    os.makedirs(log_dir, exist_ok=True)

    # 配置预训练模型路径：Task 3 模型
    pretrained_dir = "/home/lw/IsaacLab/logs/g1_whole_body_0427_2205"
    pretrained_model_path = os.path.join(pretrained_dir, "g1_model.zip")
    pretrained_stats_path = os.path.join(pretrained_dir, "vec_normalize.pkl")

    if not os.path.exists(pretrained_model_path):
        print(f"\n❌ 致命错误：未找到预训练模型 {pretrained_model_path}")
        sys.exit(1)

    print("\n" + "="*70)
    print("🚀 G1 Task 4: 马拉松 Sim2Real 破局训练启动 (电击疗法)")
    print("="*70)

    env_cfg = Task4Config()
    env_cfg.num_envs = 1024 
    base_env = G1MarathonEnv(env_cfg)
    env = CustomSb3VecEnvWrapper(base_env)
    
    env = VecFrameStack(env, n_stack=5)
    
    if os.path.exists(pretrained_stats_path):
        print(f"📥 成功接管归一化统计数据: {pretrained_stats_path}")
        env = VecNormalize.load(pretrained_stats_path, env)
        env.training = True
        env.norm_reward = True
    else:
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    print(f"📥 成功注入 6.5 亿步神经网络权重: {pretrained_model_path}")
    
    policy_kwargs = dict(
        activation_fn=torch.nn.ELU,
        net_arch=dict(pi=[512, 256, 128], vf=[512, 256, 128]),
        ortho_init=True,
    )
    
    custom_objects = {
        "learning_rate": 5e-4,     # 初始给 5e-4
        "n_steps": 128,            
        "batch_size": 32768,       
        "n_epochs": 5,
        "clip_range": 0.2,
        "max_grad_norm": 1.0,      
        "policy_kwargs": policy_kwargs
    }
    
    model = PPO.load(
        pretrained_model_path, 
        env=env, 
        custom_objects=custom_objects, 
        device=env_cfg.device,
        tensorboard_log=log_dir  
    )
    
    new_logger = configure(log_dir, ["stdout", "tensorboard"])
    model.set_logger(new_logger)

    training_callback = G1TrainingCallback(save_freq=50_000_000, save_path=log_dir)
    kl_callback = AdaptiveKLCallback(target_kl=0.02, min_lr=1e-5, max_lr=5e-4)

    print(f"\n🔥 [破局训练已开始] 请紧盯 TensorBoard！")
    print(f"💡 预期现象：由于惩罚了对称性和提高了学习率，初期跌倒率可能飙升。")
    print(f"   当您看到 'rewards/P_Symmetry' 逐渐向 0 收敛，且 'actual_vx' 突破 1.0m/s 时，")
    print(f"   这就宣告了局部最优被彻底打破，机器人重新学会了双腿奔跑！")
    
    try:
        model.learn(
            total_timesteps=1_000_000_000,
            callback=[training_callback, kl_callback],
            progress_bar=True,
            reset_num_timesteps=True  
        )
    except KeyboardInterrupt:
        print("\n[WARN] 手动中断，正在保存模型...")
    finally:
        model.save(os.path.join(log_dir, "final_model.zip"))
        env.save(os.path.join(log_dir, "vec_normalize.pkl"))
        
        del model
        env.close()
        simulation_app.close()
        os._exit(0)

if __name__ == "__main__":
    main()