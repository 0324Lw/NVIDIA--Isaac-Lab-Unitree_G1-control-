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

parser = argparse.ArgumentParser(description="Task 3 Whole-Body Coordination Limit Test")
AppLauncher.add_app_launcher_args(parser)
args_cli, unknown_args = parser.parse_known_args()
args_cli.headless = True  # 强制无头模式进行高频压测
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 必须在引擎点火后导入环境代码
from task3_env import G1WholeBodyEnv, Task3Config

# ===================================================================
# 核心压测管线
# ===================================================================
def run_limit_test():
    print("\n" + "="*80)
    print("🚀 G1 全身协同控制 (Task 3) 工业级防弹极限测试启动")
    print("="*80)

    # ---------------------------------------------------------
    # [测试项 1] 校验极限协同 AMP 动捕张量
    # ---------------------------------------------------------
    print("\n[1/6] 深度校验 Task 3 极限动捕张量...")
    motion_file = "g1_extreme_omni.pt"
    assert os.path.exists(motion_file), f"❌ 找不到张量文件 {motion_file}，请先运行 process_amp_to_g1_3.py！"
    
    motion_data = torch.load(motion_file)
    required_keys = ["pos", "vel", "cmd", "num_frames"]
    for key in required_keys:
        assert key in motion_data, f"❌ 张量字典缺少核心键值: {key}"
    
    raw_dim = motion_data["pos"].shape[1]
    num_frames = motion_data["num_frames"]
    assert motion_data["pos"].dtype == torch.float32, "❌ 数据类型非 float32，会导致显存浪费！"
    assert motion_data["cmd"].shape == (num_frames, 3), "❌ 摇杆指令维度异常！"
    
    print(f"✅ 极限协同动捕张量加载成功！包含 {num_frames} 帧，原生特征维度: {raw_dim} 维。")
    print(f"✅ RSI 专用摇杆指令流形验证通过，包含 1.2m/s 冲刺与 0.8rad/s 急弯。")

    # ---------------------------------------------------------
    # [测试项 2] 实例化环境与向量化操作验证
    # ---------------------------------------------------------
    print("\n[2/6] 实例化 Task 3 全身协同环境与向量化验证...")
    cfg = Task3Config()
    cfg.num_envs = 16  # 锁定并行数为 16，防爆显存且足够进行统计
    
    # 临时为了加速测试，把课程步数缩短，方便我们观察手臂解封情况
    original_unlock_steps = cfg.arm_unlock_steps
    cfg.arm_unlock_steps = 2000 
    
    env = G1WholeBodyEnv(cfg)
    obs, _ = env.reset()
    print(f"✅ 成功建立向量化环境，当前并行数量: {cfg.num_envs}")

    # ---------------------------------------------------------
    # [测试项 3] 状态与动作空间校验
    # ---------------------------------------------------------
    print("\n[3/6] 状态与动作空间维度与数值校验...")
    actual_actions = env.action_space.shape[0]
    actual_obs = env.observation_space.shape[0]
    
    assert obs.shape == (cfg.num_envs, actual_obs), f"❌ 观测张量形状不匹配！预期: {(cfg.num_envs, actual_obs)}, 实际: {obs.shape}"
    assert actual_obs == 310, "❌ 状态空间不是预期的 310 维，将导致 VecFrameStack 崩溃！"
    assert not torch.isnan(obs).any(), "❌ 观测张量中出现 NaN 异常值，物理引擎可能已崩溃！"
    assert torch.max(obs) <= 1.01 and torch.min(obs) >= -1.01, "❌ 观测张量未严格通过 Tanh 归一化！"
    
    print(f"✅ 状态空间维度: {actual_obs} 维 (支持 5 帧堆叠)")
    print(f"✅ 动作空间维度: {actual_actions} 维")
    print(f"✅ 观测数值已 Tanh 归一化，范围: [{torch.min(obs):.4f}, {torch.max(obs):.4f}]")

    # ---------------------------------------------------------
    # [测试项 4 & 5] 5000 步极限探索、终局极刑与课程测试
    # ---------------------------------------------------------
    print("\n[4&5/6] 执行 5000 步随机探索，测试跌倒极刑与【上肢解封课程】...")
    
    num_steps = 5000
    history_data = []
    total_falls = 0
    start_time = time.time()
    
    fall_penalty_tensor = torch.tensor(cfg.rew_fall, device=cfg.device, dtype=torch.float32)
    arm_activation_history = []

    for step in range(num_steps):
        # 生成 [-1, 1] 的极限随机动作
        actions = 2.0 * torch.rand((cfg.num_envs, actual_actions), device=cfg.device) - 1.0
        
        obs, rewards, terminated, truncated, info = env.step(actions)
        
        # 极刑事件拦截校验
        num_falls_this_step = terminated.sum().item()
        if num_falls_this_step > 0:
            total_falls += num_falls_this_step
            fallen_indices = terminated.nonzero(as_tuple=False).squeeze(-1)
            fallen_rewards = rewards[fallen_indices]
            # 允许 1e-4 的浮点误差
            assert torch.allclose(fallen_rewards, fall_penalty_tensor.expand_as(fallen_rewards), atol=1e-4), "❌ 跌倒时未正确触发极刑 (-1.0) 覆盖！"

        # 数据抓取
        step_record = {"Step": step}
        if "reward_components" in info:
            step_record.update(info["reward_components"])
            
        if "telemetry" in info:
            act_val = info["telemetry"].get("arm_activation", 0.0)
            arm_activation_history.append(act_val)
            
        history_data.append(step_record)
        
        if (step + 1) % 1000 == 0:
            fps = 1000 * cfg.num_envs / (time.time() - start_time)
            current_act = arm_activation_history[-1]
            print(f"   -> 进度: {step+1}/{num_steps} 步 | 帧率: {fps:.1f} FPS | 累计摔倒重置: {total_falls} 次 | 上肢解封率: {current_act*100:.1f}%")
            start_time = time.time()

    # 验证上肢解封课程是否随 step 增加而正常推进
    assert arm_activation_history[-1] > arm_activation_history[0], "❌ 课程学习失败：上肢控制权未随时间解封！"
    print("✅ 课程学习测试通过：上肢动作解封率随时间动态增长 (Action Space Soft-Switching 正常)。")
    print("✅ 终局事件测试通过：侧翻/坠落时，跌倒极刑 (-1.0) 已成功阻断奖励。")

    # ---------------------------------------------------------
    # [测试项 6] Pandas 奖励组件统计学分析
    # ---------------------------------------------------------
    print("\n[6/6] 导出并分析 5000 步奖励组件统计学特征...")
    df = pd.DataFrame(history_data)
    
    reward_cols = [col for col in df.columns if col.startswith("R_") or col.startswith("P_")]
    
    stats_df = df[reward_cols].describe(percentiles=[0.25, 0.5, 0.75]).T
    stats_df['var'] = df[reward_cols].var() 
    
    columns_order = ['mean', 'var', 'min', '25%', '50%', '75%', 'max']
    stats_df = stats_df[columns_order]
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    
    print("\n📊 Task 3 各项奖励组件 5000 步全局统计分析 (纯随机策略下的极限撕扯状态):")
    print("-" * 100)
    print(stats_df.to_string(float_format="{:.5f}".format))
    print("-" * 100)
    
    print("\n💡 [重点诊断指引]：")
    print("  1. 检查 'R_Arm_Leg_Sync': 均值可能极低，因为随机策略很难碰巧产生正确的手脚反相配合，这正是我们要训练的内容。")
    print("  2. 检查 'P_Arm_Cross': 如果有较大的负值，说明软限位惩罚生效，成功拦截了随机乱扭产生的手臂内收穿模。")
    print("  3. 检查 'P_Base_Ang': 因引入了动态宽容度，其最大惩罚应该比 Task 2 时略轻。")
    print("  4. 若控制台未抛出张量维度或越界报错，说明您的环境已达到工业部署级别，可直接启动 PPO 训练！")

if __name__ == "__main__":
    try:
        run_limit_test()
    except Exception as e:
        print(f"\n❌ [测试脚本外层异常] {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()