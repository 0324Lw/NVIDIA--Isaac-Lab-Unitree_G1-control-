import argparse
import torch
import numpy as np
import pandas as pd
import os
import time

from isaaclab.app import AppLauncher

# ===================================================================
# 1. 启动 IsaacLab 底层引擎
# ===================================================================
parser = argparse.ArgumentParser(description="Task 1 Reward Functions Sandbox Test")
AppLauncher.add_app_launcher_args(parser)
args_cli, unknown_args = parser.parse_known_args()
args_cli.headless = True 
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 必须在引擎启动后导入环境代码
from task1_env import G1HarnessEnv, Task1Config

# ===================================================================
# 2. 场景压测核心函数
# ===================================================================
def test_scenario(env, scenario_name, action_mode, target_harness, steps=50000):
    print("\n" + "━"*90)
    print(f"▶️ 开始测试场景: {scenario_name} | 设定拉力比: {target_harness:.1f}")
    print("━"*90)

    env.cfg.harness_initial_ratio = target_harness
    env.global_step = 0 
    obs, _ = env.reset()

    history_data = []
    actual_actions = env.action_space.shape[0]

    for step in range(steps):
        # --- 动作注入逻辑 ---
        if action_mode == "zombie":
            # 僵尸臂：下肢不动 (0.0)，强行将上肢/躯干关节打满 (1.0)
            actions = torch.zeros((env.cfg.num_envs, actual_actions), device=env.device)
            # 假设后 13 个关节为上肢
            actions[:, 12:] = 1.0  
        elif action_mode == "jitter":
            # 高频抽搐：每一帧都在 1.0 和 -1.0 之间疯狂横跳
            val = 1.0 if step % 2 == 0 else -1.0
            actions = torch.full((env.cfg.num_envs, actual_actions), val, device=env.device)
        elif action_mode == "zero":
            # 完美挂机：输出绝对的 0 动作
            actions = torch.zeros((env.cfg.num_envs, actual_actions), device=env.device)
        elif action_mode == "random":
            # 随机乱扭
            actions = 2.0 * torch.rand((env.cfg.num_envs, actual_actions), device=env.device) - 1.0

        # 环境步进
        obs, rewards, terminated, truncated, info = env.step(actions)

        # 收集数据
        step_record = {"Step": step}
        if "reward_components" in info:
            step_record.update(info["reward_components"])
            
        # 顺便记录一下真实的拉力，验证混合衰减机制是否正常
        if "telemetry" in info and "harness_ratio" in info["telemetry"]:
            step_record["Actual_Harness"] = info["telemetry"]["harness_ratio"]
            
        history_data.append(step_record)

    # --- 数据统计与 Pandas 输出 ---
    df = pd.DataFrame(history_data)
    reward_cols = [col for col in df.columns if col.startswith("R_") or col.startswith("P_")]
    
    # 计算统计量
    stats_df = df[reward_cols].describe(percentiles=[0.25, 0.5, 0.75]).T
    stats_df['var'] = df[reward_cols].var() 
    
    columns_order = ['mean', 'var', 'min', '25%', '50%', '75%', 'max']
    stats_df = stats_df[columns_order]
    
    print(f"\n📊 [{scenario_name}] 500步核心奖励指标剖析:")
    print("-" * 90)
    print(stats_df.to_string(float_format="{:.5f}".format))
    print("-" * 90)

# ===================================================================
# 3. 主压测管线
# ===================================================================
def run_limit_test():
    print("\n" + "="*90)
    print("🚀 G1 工业级奖励函数 (6层架构) 白盒沙盘压测启动")
    print("="*90)

    cfg = Task1Config()
    cfg.num_envs = 16  # 锁定16个环境防爆显存
    env = G1HarnessEnv(cfg)
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    # ---------------------------------------------------------
    # 压测执行区
    # ---------------------------------------------------------
    test_scenario(env, "场景 1 - 完美挂机 (0动作输出)", "zero", target_harness=1.0, steps=500)
    test_scenario(env, "场景 2 - 僵尸臂局部最优 (仅上肢满输出)", "zombie", target_harness=1.0, steps=500)
    test_scenario(env, "场景 3 - 极限高频抽搐 (+1/-1交替)", "jitter", target_harness=1.0, steps=500)
    test_scenario(env, "场景 4 - 无保护伞坠落 (0拉力随机动作)", "random", target_harness=0.0, steps=500)

    print("\n✅ 所有奖励压测管线执行完毕！请对照预期分析惩罚项的均值与方差。")

if __name__ == "__main__":
    try:
        run_limit_test()
    except Exception as e:
        print(f"\n❌ [测试脚本外层异常] {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()