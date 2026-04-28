import sys
import os
import argparse

print("\n" + "="*50, flush=True)
print("🚀 [全身协同视觉验收] 正在初始化 Isaac Sim 引擎...", flush=True)

# ===================================================================
# 1. 絕對優先：單次啟動 Isaac Sim 底層引擎
# ===================================================================
if "--video" not in sys.argv:
    sys.argv.extend([
        "--video", 
        "--video_dir", "task3_video_records", 
        "--video_length", "500",              
        "--video_fps", "30"                   
    ])

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli, unknown_args = parser.parse_known_args()

# 🚨 終極修復：將這裡改回 True！
# 在 Headless 模式下，Isaac Sim 不會彈出視窗，但在背景會以極快的速度把影片算染並存檔出來
args_cli.headless = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

print("🚀 引擎点火与录制模块就绪！准备导入神经网络库...", flush=True)

# ===================================================================
# 2. 导入库 (引擎存活后导入，绝对安全)
# ===================================================================
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, VecFrameStack
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# 🚨 导入 Task 3 专属环境
from task3_env import G1WholeBodyEnv, Task3Config

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
# 4. 主视觉推理与录像流
# ===================================================================
def main():
    # 🚨 请务必将此处修改为你刚刚跑完 Task 3 (6.4亿步) 的最新 logs 目录
    model_dir = "/home/lw/IsaacLab/logs/g1_whole_body_0427_2205" 
    model_path = os.path.join(model_dir, "g1_model.zip")
    stats_path = os.path.join(model_dir, "vec_normalize.pkl")

    if not os.path.exists(model_path) or not os.path.exists(stats_path):
        print(f"\n❌ 致命错误: 找不到模型或归一化文件！请检查路径：\n{model_path}\n{stats_path}")
        os._exit(1)

    print("\n🚀 模型路径校验通过，开始实例化 Task 3 全身协同物理环境...", flush=True)

    env_cfg = Task3Config()
    env_cfg.num_envs = 1  
    
    # 🚨 极其关键：在测试时强制将“手臂解封课程步数”设为 1，直接展现完全体！
    env_cfg.arm_unlock_steps = 1
    
    base_env = G1WholeBodyEnv(env_cfg)
    env = CustomSb3VecEnvWrapper(base_env)

    # 严格保持训练时的 5 帧堆叠结构 (1550维)
    env = VecFrameStack(env, n_stack=5)
    
    # 加载状态观测值分布，并冻结更新
    env = VecNormalize.load(stats_path, env)
    env.training = False
    env.norm_reward = False

    # 加载 PPO 模型
    model = PPO.load(model_path, env=env, device=env_cfg.device)

    print("\n🎬 仿真与录像已启动！正在将机动画面截屏录制为 MP4...")
    print("💡 提示：视频将自动保存在当前工作目录的 'task3_video_records' 文件夹中。")
    print("💡 按 Ctrl+C 可以安全退出并保存视频文件。")
    
    print("\n🎬 仿真与录像已启动！正在将机动画面截屏录制为 MP4...")
    print("💡 提示：程序将自动运行 1500 步（约 30 秒）后自动保存视频，请勿提前按 Ctrl+C！")
    
    obs = env.reset()
    
    # 🚨 设定最大录制步数，对齐启动参数中的 --video_length 1500
    max_steps = 3000
    step_count = 0
    
    try:
        # 改为定长循环，不再使用无限循环
        while simulation_app.is_running() and step_count < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, infos = env.step(action)
            
            step_count += 1
            
            # 终端实时反馈协同状态
            if len(infos) > 0 and "telemetry" in infos[0]:
                t = infos[0]["telemetry"]
                cmd_x, cmd_w = t.get('cmd_vx_target', 0), t.get('cmd_wz_target', 0)
                act_x, act_w = t.get('actual_vx', 0), t.get('actual_wz', 0)
                arm_act = t.get('arm_activation', 0)
                
                sys.stdout.write(
                    f"\r🤖 [录制进度: {step_count}/{max_steps}] 目标 [Vx:{cmd_x:5.2f} | Wz:{cmd_w:5.2f}] >>> "
                    f"实际 [Vx:{act_x:5.2f} | Wz:{act_w:5.2f}]"
                )
                sys.stdout.flush()
            
            if dones[0]:
                obs = env.reset()

        print("\n\n[INFO] 🎥 1500帧录制完毕！正在安全关闭物理引擎并渲染 MP4 视频...")
        print("[INFO] ⏳ 这个过程可能需要几秒到十几秒，请耐心等待程序自行结束...")

    except KeyboardInterrupt:
        print("\n\n[WARN] 警告：被强制中断！视频可能无法正确保存。")
    finally:
        # 优雅清理内存
        del model
        del env
        del base_env
        
        print("\n[INFO] ⏳ 正在通知底层 C++ 引擎压缩 MP4 视频...")
        print("[INFO] 🚨 警告：此时 GUI 界面会卡死（无响应），这是 FFmpeg 在后台编码的正常现象！")
        print("[INFO] 🚨 请双手离开键盘，等待约 30 秒至 1 分钟，直到终端自动回到命令行输入符...")
        
        # 🚨 极其关键：只调用 close，不调用 sys.exit
        simulation_app.close()

if __name__ == "__main__":
    main()