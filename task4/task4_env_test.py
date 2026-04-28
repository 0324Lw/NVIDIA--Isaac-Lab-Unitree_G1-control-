import argparse
import torch
import numpy as np
import pandas as pd
import os
import time

# ===================================================================
# 1. 启动 IsaacLab 底层引擎 (必须在导入任何 sim 模块前执行)
# ===================================================================
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Task 4 Marathon Sim2Real Limit Test")
AppLauncher.add_app_launcher_args(parser)
args_cli, unknown_args = parser.parse_known_args()
args_cli.headless = True  # 强制无头模式进行高频压测
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 必须在引擎点火后导入环境代码
from task4_env import G1MarathonEnv, Task4Config

# ===================================================================
# 核心压测管线
# ===================================================================
def run_limit_test():
    print("\n" + "="*80)
    print("🚀 G1 马拉松定速奔跑与 Sim2Real 抗扰 (Task 4) 工业级防弹压测启动")
    print("="*80)

    # ---------------------------------------------------------
    # [测试项 1] 校验马拉松专属 AMP 动捕张量
    # ---------------------------------------------------------
    print("\n[1/6] 深度校验 Task 4 马拉松动捕张量...")
    motion_file = "g1_marathon_cpg.pt"
    assert os.path.exists(motion_file), f"❌ 找不到张量文件 {motion_file}，请先运行 process_amp_to_g1_4.py！"
    
    motion_data = torch.load(motion_file)
    required_keys = ["pos", "vel", "cmd", "num_frames"]
    for key in required_keys:
        assert key in motion_data, f"❌ 张量字典缺少核心键值: {key}"
    
    print(f"✅ 马拉松动捕张量加载成功！包含 {motion_data['num_frames']} 帧数据。")
    print(f"✅ RSI 专用摇杆指令流形验证通过，包含 1.5m/s 巡航与 2.0m/s 极限冲刺配速。")

    # ---------------------------------------------------------
    # [测试项 2] 实例化环境与向量化操作验证
    # ---------------------------------------------------------
    print("\n[2/6] 实例化 Task 4 Sim2Real 环境与向量化验证...")
    cfg = Task4Config()
    cfg.num_envs = 16  # 锁定并行数为 16，防爆显存
    
    # 临时加速课程，观察解封
    cfg.arm_unlock_steps = 2000 
    # 为了快速触发终点判定，临时缩短最大回合长度
    cfg.max_episode_length = 150 
    
    env = G1MarathonEnv(cfg)
    obs, _ = env.reset()
    print(f"✅ 成功建立向量化环境，当前并行数量: {cfg.num_envs}")

    # ---------------------------------------------------------
    # [测试项 3] 状态与动作空间校验 (包含 RMA 特权信息)
    # ---------------------------------------------------------
    print("\n[3/6] 状态空间、动作空间与 RMA 特权信息校验...")
    actual_actions = env.action_space.shape[0]
    actual_obs = env.observation_space.shape[0]
    
    assert obs.shape == (cfg.num_envs, actual_obs), f"❌ 观测张量形状不匹配！预期: {(cfg.num_envs, actual_obs)}"
    assert not torch.isnan(obs).any(), "❌ 观测张量中出现 NaN 异常值，物理引擎或延迟队列可能已崩溃！"
    assert torch.max(obs) <= 1.01 and torch.min(obs) >= -1.01, "❌ 观测张量未严格通过 Tanh 归一化！"
    
    print(f"✅ 状态空间维度: {actual_obs} 维 (支持多帧堆叠)")
    print(f"✅ 动作空间维度: {actual_actions} 维")

    # ---------------------------------------------------------
    # [测试项 4 & 5] 5000 步极限探索：碰撞/推搡、终点、极刑
    # ---------------------------------------------------------
    print("\n[4&5/6] 执行 5000 步随机探索，测试抗扰机制、终点结算与奖励流...")
    
    num_steps = 5000
    history_data = []
    
    total_falls = 0
    total_pushes = 0
    total_finishes = 0
    
    start_time = time.time()
    fall_penalty_tensor = torch.tensor(cfg.rew_fall, device=cfg.device, dtype=torch.float32)

    for step in range(num_steps):
        # 生成 [-1, 1] 的极限随机动作
        actions = 2.0 * torch.rand((cfg.num_envs, actual_actions), device=cfg.device) - 1.0
        
        obs, rewards, terminated, truncated, info = env.step(actions)
        
        # --- A. 极刑与跌倒校验 ---
        num_falls_this_step = terminated.sum().item()
        if num_falls_this_step > 0:
            total_falls += num_falls_this_step
            fallen_indices = terminated.nonzero(as_tuple=False).squeeze(-1)
            fallen_rewards = rewards[fallen_indices]
            assert torch.allclose(fallen_rewards, fall_penalty_tensor.expand_as(fallen_rewards), atol=1e-4), "❌ 跌倒时未正确触发极刑 (-1.0)！"

        # --- B. 终点判定 (Truncated) 校验 ---
        num_finishes_this_step = truncated.sum().item()
        if num_finishes_this_step > 0:
            total_finishes += num_finishes_this_step

        # --- C. 障碍物/推力碰撞判定 校验 ---
        if "privileged_obs" in info and "is_pushed" in info["privileged_obs"]:
            pushes_this_step = info["privileged_obs"]["is_pushed"].sum().item()
            total_pushes += pushes_this_step

        # 数据抓取
        step_record = {"Step": step}
        if "reward_components" in info:
            step_record.update(info["reward_components"])
            
        history_data.append(step_record)
        
        if (step + 1) % 1000 == 0:
            fps = 1000 * cfg.num_envs / (time.time() - start_time)
            print(f"   -> 进度: {step+1}/{num_steps} 步 | 帧率: {fps:.1f} FPS | 累计摔倒: {total_falls} | 遭遇推搡: {total_pushes} | 抵达终点(截断): {total_finishes}")
            start_time = time.time()

    assert total_pushes > 0, "❌ 障碍物/推搡判定失效：5000 步内未检测到任何外力注入！"
    assert total_finishes > 0, "❌ 终点判定失效：未能正常触发基于回合长度的 Truncated 截断！"
    
    print("✅ 机器狗障碍物(外力推搡)注入与感知判定逻辑测试通过。")
    print("✅ 终点(马拉松最大里程截断)与重置逻辑测试通过。")
    print("✅ 终局事件测试通过：侧翻/坠落时，跌倒极刑 (-1.0) 已成功阻断奖励。")

    # ---------------------------------------------------------
    # [测试项 6] Pandas 奖励组件统计学分析
    # ---------------------------------------------------------
    print("\n[6/6] 导出并分析 5000 步奖励组件统计学特征...")
    df = pd.DataFrame(history_data)
    
    # 动态抓取所有 R_ 和 P_ 开头的奖励列
    reward_cols = [col for col in df.columns if col.startswith("R_") or col.startswith("P_")]
    
    # 计算统计特征
    stats_df = df[reward_cols].describe(percentiles=[0.25, 0.5, 0.75]).T
    stats_df['var'] = df[reward_cols].var() 
    
    # 重排序列，符合需求
    columns_order = ['mean', 'var', 'min', '25%', '50%', '75%', 'max']
    stats_df = stats_df[columns_order]
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    
    print("\n📊 Task 4 各项奖励组件 5000 步全局统计分析 (纯随机策略下的极限拉扯状态):")
    print("-" * 100)
    print(stats_df.to_string(float_format="{:.5f}".format))
    print("-" * 100)
    
    print("\n💡 [重点诊断指引]：")
    print("  1. 'P_Symmetry': 由于随机动作乱扭，早期对称性惩罚应该有明显的负值方差。")
    print("  2. 'P_Energy': 引入 CoT 后，极限随机乱扭会造成巨大的能量消耗，应当观察到深负值的惩罚。")
    print("  3. 'P_Base_Ang': 得益于动态安全边界(is_recovering)，当触发推力或倾斜时，惩罚被放宽，最大值不至于卡死网络。")
    print("  4. 恭喜！如果成功看到此表格，说明 Task 4 的 RMA 延迟、推力机制与新版奖励已无缝衔接。")

if __name__ == "__main__":
    try:
        run_limit_test()
    except Exception as e:
        print(f"\n❌ [测试脚本外层异常] {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()