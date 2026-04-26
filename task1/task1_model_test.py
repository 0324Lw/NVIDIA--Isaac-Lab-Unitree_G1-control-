import sys
import os
import argparse

print("\n" + "="*50, flush=True)
print("🚀 [视觉验收] 正在初始化 Isaac Sim 引擎...", flush=True)

# ===================================================================
# 1. 绝对优先：单次启动 Isaac Sim 底层引擎 (开启 GUI 模式)
# ===================================================================
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli, unknown_args = parser.parse_known_args()
args_cli.headless = False  
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

print("🚀 引擎点火成功！准备导入神经网络库...", flush=True)

# ===================================================================
# 2. 导入库 (引擎存活后导入，绝对安全)
# ===================================================================
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, VecFrameStack
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from task1_env import G1HarnessEnv, Task1Config

# ===================================================================
# 3. 本地化 Wrapper，彻底斩断交叉 Import 污染
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
        return obs_np, rewards_np, dones, list_infos

    def close(self): self.env.close()
    def get_attr(self, name, indices=None): return [getattr(self.env, name, None)] * self.num_envs
    def set_attr(self, name, value, indices=None): setattr(self.env, name, value)
    def env_method(self, method_name, *method_args, indices=None, **method_kwargs): return [getattr(self.env, method_name)(*method_args, **method_kwargs)] * self.num_envs
    def env_is_wrapped(self, wrapper_class, indices=None): return [False] * self.num_envs

# ===================================================================
# 4. 主视觉推理流
# ===================================================================
def main():
    # 🚨 精确加载你指定的模型路径
    model_dir = "/home/lw/IsaacLab/logs/g1_amp_0426_1848" 
    model_path = os.path.join(model_dir, "g1_model.zip")
    stats_path = os.path.join(model_dir, "vec_normalize.pkl")

    if not os.path.exists(model_path) or not os.path.exists(stats_path):
        print(f"\n❌ 致命错误: 找不到模型或归一化文件！请检查路径：\n{model_path}\n{stats_path}")
        os._exit(1)

    print("\n🚀 模型路径校验通过，开始实例化物理环境...", flush=True)

    env_cfg = Task1Config()
    # 视觉展示只需要 2 个环境，太多会掉帧
    env_cfg.num_envs = 2  
    # 极限零拉力挑战：看看网络是否真的学会了自我平衡！
    env_cfg.harness_initial_ratio = 0.0 
    
    base_env = G1HarnessEnv(env_cfg)
    env = CustomSb3VecEnvWrapper(base_env)

    # 这样网络在推理时接收到的才是正确的 1550 维张量
    env = VecFrameStack(env, n_stack=5)
    
    # 加载状态观测值分布，并冻结更新
    env = VecNormalize.load(stats_path, env)
    env.training = False
    env.norm_reward = False

    model = PPO.load(model_path, env=env, device=env_cfg.device)

    print("\n🎬 仿真已启动！请在 Isaac Sim 窗口中查看机器人步态。")
    print("💡 提示：在终端按 Ctrl+C 可以安全退出。")
    
    # reset 后吐出的 obs 已经是 (2, 1550) 维度的张量了
    obs = env.reset()
    
    try:
        # 当 Isaac Sim 的图形界面没被关闭时，持续循环
        while simulation_app.is_running():
            # 使用确定性策略以展示最优步态
            action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, infos = env.step(action)
            
            # 在终端实时反馈状态
            if len(infos) > 0 and "telemetry" in infos[0]:
                t = infos[0]["telemetry"]
                print(f"🤖 状态 -> 高度: {t.get('torso_height', 0):.2f}m | 速度: {t.get('actual_vx', 0):.2f}m/s", end='\r')
            
            # 机器人跌倒时自动重置
            if dones[0]:
                print("\n[EVENT] 机器人触发重置...")
                obs = env.reset()

    except KeyboardInterrupt:
        print("\n\n[INFO] 接收到退出指令，正在安全关闭...")
    finally:
        del model
        del env
        del base_env
        simulation_app.close()
        os._exit(0)

if __name__ == "__main__":
    main()