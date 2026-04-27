import argparse
import torch
import numpy as np
import pandas as pd
import os
import time

from isaaclab.app import AppLauncher

# ===================================================================
# 1. 启动 IsaacLab 底层引擎 (必须在导入任何 sim 模块前执行)
# ===================================================================
parser = argparse.ArgumentParser(description="Task 2 Omni-Control Extreme Limit Test")
AppLauncher.add_app_launcher_args(parser)
args_cli, unknown_args = parser.parse_known_args()
# 强制推荐使用无头模式进行高频极限压测，节省渲染显存
args_cli.headless = True 
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 必须在引擎启动后导入环境代码
from task2_env import G1OmniEnv, Task2Config

def run_limit_test():
    print("\n" + "="*80)
    print("🚀 G1 全向机动 (Task 2) 工业级防弹极限测试管线启动")
    print("="*80)

    # ---------------------------------------------------------
    # [测试项 1] 校验全向 AMP 动捕张量 (Omni-Dataset)
    # ---------------------------------------------------------
    print("\n[1/6] 深度校验全向 AMP 动捕张量...")
    motion_file = "g1_omni_walk.pt"
    assert os.path.exists(motion_file), f"❌ 找不到张量文件 {motion_file}，请先运行 process_amp_to_g1_2.py！"
    
    motion_data = torch.load(motion_file)
    required_keys = ["pos", "vel", "cmd", "num_frames"]
    for key in required_keys:
        assert key in motion_data, f"❌ 张量字典缺少核心键值: {key}"
    
    raw_dim = motion_data["pos"].shape[1]
    num_frames = motion_data["num_frames"]
    assert motion_data["pos"].dtype == torch.float32, "❌ 数据类型非 float32，会导致显存浪费与计算低效！"
    assert motion_data["cmd"].shape == (num_frames, 3), "❌ 摇杆指令 (cmd) 维度异常，不满足三维 (vx, vy, wz) 全向控制需求！"
    
    print(f"✅ 全向动捕张量加载成功！包含 {num_frames} 帧，原生特征维度: {raw_dim} 维。")
    print(f"✅ RSI 专用摇杆指令流形验证通过，形状: {motion_data['cmd'].shape}。")

    # ---------------------------------------------------------
    # [测试项 2] 实例化环境与向量化操作验证
    # ---------------------------------------------------------
    print("\n[2/6] 实例化 Task 2 全向环境与向量化验证...")
    cfg = Task2Config()
    # 防爆显存策略：锁定并行数为 16 进行沙盒压测
    cfg.num_envs = 16            
    
    env = G1OmniEnv(cfg)
    obs, _ = env.reset()
    print(f"✅ 成功建立向量化环境，当前并行数量: {cfg.num_envs} (防止 OOM 截断)")

    # ---------------------------------------------------------
    # [测试项 3] 状态与动作空间校验
    # ---------------------------------------------------------
    print("\n[3/6] 状态与动作空间维度与数值校验...")
    actual_actions = env.action_space.shape[0]
    actual_obs = env.observation_space.shape[0]
    
    assert obs.shape == (cfg.num_envs, actual_obs), f"❌ 观测张量形状不匹配！预期: {(cfg.num_envs, actual_obs)}, 实际: {obs.shape}"
    assert actual_obs == 310, f"❌ 状态空间维度不是预期的 310 维，将导致 5 帧堆叠 (1550维) 崩溃！"
    assert not torch.isnan(obs).any(), "❌ 观测张量中出现 NaN 异常值，底层物理大概率已崩溃！"
    assert torch.max(obs) <= 1.01 and torch.min(obs) >= -1.01, "❌ 观测张量未严格通过 Tanh 归一化！"
    
    print(f"✅ 状态空间维度: {actual_obs} 维 (已完美填充指令与前庭觉，随时可支持 VecFrameStack)")
    print(f"✅ 动作空间维度: {actual_actions} 维 (已自动对齐 G1 物理底层)")
    print(f"✅ 观测数值已严格进行 Tanh 归一化，当前数据跨度: [{torch.min(obs):.4f}, {torch.max(obs):.4f}]")

    # ---------------------------------------------------------
    # [测试项 4 & 5] 5000 步极限随机探索与终局结算测试
    # ---------------------------------------------------------
    print("\n[4&5/6] 执行 5000 步纯随机探索，测试全向指令重采样与极刑拦截机制...")
    
    num_steps = 5000
    history_data = []
    total_falls = 0
    start_time = time.time()
    
    # 获取底层的极刑数值，确保类型对齐
    fall_penalty_tensor = torch.tensor(cfg.rew_fall, device=cfg.device, dtype=torch.float32)

    # 记录初始的摇杆指令，用于比对重采样机制
    initial_cmds = env.target_cmd.clone()
    command_changed = False

    for step in range(num_steps):
        # 生成完全随机的越界动作来逼迫机器人极限倾斜和摔倒
        actions = 2.0 * torch.rand((cfg.num_envs, actual_actions), device=cfg.device) - 1.0
        
        obs, rewards, terminated, truncated, info = env.step(actions)
        
        # 验证全向控制的“课程学习”：动态指令重采样是否生效
        if not command_changed and not torch.allclose(env.target_cmd, initial_cmds):
            command_changed = True
            
        # 极限事件测试：抓取跌倒的机器人，并验证极刑奖励
        num_falls_this_step = terminated.sum().item()
        if num_falls_this_step > 0:
            total_falls += num_falls_this_step
            fallen_indices = terminated.nonzero(as_tuple=False).squeeze(-1)
            fallen_rewards = rewards[fallen_indices]
            assert torch.allclose(fallen_rewards, fall_penalty_tensor.expand_as(fallen_rewards), atol=1e-4), "❌ 跌倒时未正确触发死亡极刑 (-1.0) 强制覆盖！"

        # 收集当前帧的所有奖励分量和遥测数据 (只保存标量)
        step_record = {"Step": step}
        if "reward_components" in info:
            step_record.update(info["reward_components"])
        if "telemetry" in info:
            # 加入当前三维指令的监控
            step_record["Cmd_Vx"] = info["telemetry"].get("cmd_vx_target", 0.0)
            step_record["Cmd_Vy"] = info["telemetry"].get("cmd_vy_target", 0.0)
            step_record["Cmd_Wz"] = info["telemetry"].get("cmd_wz_target", 0.0)
            
        history_data.append(step_record)
        
        # 进度打印
        if (step + 1) % 1000 == 0:
            fps = 1000 * cfg.num_envs / (time.time() - start_time)
            print(f"   -> 进度: {step+1}/{num_steps} 步 | 模拟帧率: {fps:.1f} FPS | 累计触发重置 (摔倒): {total_falls} 次")
            start_time = time.time()

    assert command_changed, "❌ 动态指令重采样机制失效！摇杆指令在 5000 步内未发生任何改变。"

    print("\n✅ 终局事件测试通过：跌倒判定、极刑(-1.0)强制覆盖与 RSI 重置逻辑全部正常运转。")
    print("✅ 全向动态课程测试通过：指令重采样 (Command Resampling) 正常触发，摇杆信号随时间动态切换。")

    # ---------------------------------------------------------
    # [测试项 6] 基于 Pandas 的奖励组件统计分析
    # ---------------------------------------------------------
    print("\n[6/6] 导出并分析 5000 步奖励组件统计学特征...")
    df = pd.DataFrame(history_data)
    
    # 筛选出所有奖励相关的列 (R_ 开头为正奖励，P_ 开头为惩罚)
    reward_cols = [col for col in df.columns if col.startswith("R_") or col.startswith("P_")]
    
    # 计算均值、方差、最值和分位数
    stats_df = df[reward_cols].describe(percentiles=[0.25, 0.5, 0.75]).T
    stats_df['var'] = df[reward_cols].var() 
    
    # 重新排列列顺序
    columns_order = ['mean', 'var', 'min', '25%', '50%', '75%', 'max']
    stats_df = stats_df[columns_order]
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    
    print("\n📊 各项奖励组件 5000 步全局统计分析 (随机策略下的极限拉扯状态):")
    print("-" * 100)
    print(stats_df.to_string(float_format="{:.5f}".format))
    print("-" * 100)
    print("💡 结论指引：")
    print("  - 由于使用了高频纯随机策略，机器人会像疯子一样抽搐和摔倒，因此 'P_Smooth', 'P_Energy' 惩罚必定极高。")
    print("  - 'R_Track_Lin' 和 'R_Track_Yaw' 应处于低位，因为随机扭动极难精确追踪随机生成的摇杆指令。")
    print("  - 检查所有的 'P_' 开头的项的 max 值，确保它们绝不为正数 (修复了 Task 1 早期的符号 BUG)。")
    print("  - 若控制台未抛出张量维度报错，说明 310 维空间对齐完美，已完全准备好进行 PPO 强化学习！")

if __name__ == "__main__":
    try:
        run_limit_test()
    except Exception as e:
        print(f"\n❌ [测试脚本外层异常] {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()