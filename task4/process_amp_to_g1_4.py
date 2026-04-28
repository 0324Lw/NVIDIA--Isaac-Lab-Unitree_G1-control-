import numpy as np
import torch
import math

# ===================================================================
# G1 马拉松定速奔跑运动学合成引擎 (Task 4 Sim2Real 专属)
# ===================================================================
def generate_marathon_data_v4():
    output_file = "g1_marathon_cpg.pt"
    print("\n" + "="*80)
    print("🚀 正在启动 G1 马拉松专属步态合成引擎 (Marathon CPG v4)...")
    print("="*80)

    # 基础设定
    fps = 30.0
    dt = 1.0 / fps
    frames_per_mode = 400  # 增加每种模式的帧数，提供更长的稳态参考
    num_joints = 25        

    # 🚨 Task 4 核心特化：纯粹的 X 轴定速巡航，阶梯式配速
    modes = [
        {"name": "Stand",          "cmd": [0.0, 0.0, 0.0]},   # 静止待命
        {"name": "Walk",           "cmd": [0.5, 0.0, 0.0]},   # 基础行走
        {"name": "Jog",            "cmd": [1.0, 0.0, 0.0]},   # 慢跑
        {"name": "Marathon_Pace",  "cmd": [1.5, 0.0, 0.0]},   # 🚨 马拉松巡航配速
        {"name": "Sprint_Max",     "cmd": [2.0, 0.0, 0.0]},   # 🚨 极限冲刺
        {"name": "Backward",       "cmd": [-0.5, 0.0, 0.0]},  # 基础后退防摔
    ]
    
    total_frames = frames_per_mode * len(modes)
    g1_pos = np.zeros((total_frames, num_joints), dtype=np.float32)
    g1_vel = np.zeros((total_frames, num_joints), dtype=np.float32)
    g1_cmd = np.zeros((total_frames, 3), dtype=np.float32) 

    print(f"📦 预分配内存: 总帧数 {total_frames}, 关节维数 {num_joints}")

    # =========================================================
    # 注入马拉松高级动力学特征 (Forward Leaning & High-Speed Kinematics)
    # =========================================================
    for idx, mode in enumerate(modes):
        start_idx = idx * frames_per_mode
        end_idx = start_idx + frames_per_mode
        time_steps = np.linspace(0, frames_per_mode * dt, frames_per_mode)
        
        v_x, v_y, w_z = mode["cmd"]
        g1_cmd[start_idx:end_idx] = mode["cmd"]
        
        # 🚨 步频优化：高速时步频不宜过快，主要靠步幅（Amplitude）提速
        base_freq = 5.0 + 1.0 * abs(v_x) 
        phase_L = time_steps * base_freq
        phase_R = time_steps * base_freq + math.pi
        
        # ==========================================
        # 1. 躯干主动前倾 (Task 4 马拉松核心)
        # ==========================================
        if v_x > 0.5:
            # 速度越快，身体前倾角越大 (例如 2.0m/s 时前倾约 0.3 rad)
            forward_lean = 0.15 * v_x 
            # 假设关节 14 为腰部 Pitch 轴 (需根据实际 G1 URDF 确认)
            g1_pos[start_idx:end_idx, 14] = forward_lean

        # ==========================================
        # 2. 下肢奔跑运动学 (高抬腿与大跨步)
        # ==========================================
        if abs(v_x) > 0:
            direction = 1.0 if v_x > 0 else -1.0
            
            # 腿部振幅解封上限：1.5m/s~2.0m/s 需要更大的髋部摆幅
            leg_amp_hip = min(0.6 * (abs(v_x) / 0.5 + 0.1), 1.2)  # 上限提至 1.2 rad
            leg_amp_knee = min(0.4 * (abs(v_x) / 0.5 + 0.1), 1.0) # 膝盖提至 1.0 rad (增加离地间隙)
            
            # 注入相位差，模拟蹬地与抬腿
            g1_pos[start_idx:end_idx, 3] = np.sin(phase_L) * leg_amp_hip * direction          # 左髋 Pitch
            g1_pos[start_idx:end_idx, 6] = np.sin(phase_L + 0.5) * leg_amp_knee * direction   # 左膝
            g1_pos[start_idx:end_idx, 9] = np.sin(phase_R) * leg_amp_hip * direction          # 右髋 Pitch
            g1_pos[start_idx:end_idx, 12] = np.sin(phase_R + 0.5) * leg_amp_knee * direction  # 右膝

        # ==========================================
        # 3. 上肢极限反相代偿 (抵抗 2.0m/s 偏航角动量)
        # ==========================================
        arm_amp = min(0.4 * (abs(v_x) / 0.5), 1.2) if v_x != 0 else 0.0
        if arm_amp > 0:
            direction = 1.0 if v_x > 0 else -1.0
            # 严格对侧反相挥臂：左腿前迈(phase_L)，左臂后摆；右臂前摆
            g1_pos[start_idx:end_idx, 15] = np.sin(phase_R) * arm_amp * direction  # 左肩 Pitch
            g1_pos[start_idx:end_idx, 19] = np.sin(phase_L) * arm_amp * direction  # 右肩 Pitch
            
            # 肘关节微曲，像跑步运动员一样收紧双臂，降低转动惯量
            g1_pos[start_idx:end_idx, 17] = 0.5  # 左肘曲屈 (根据实际方向调整符号)
            g1_pos[start_idx:end_idx, 21] = 0.5  # 右肘曲屈

    # =========================================================
    # 物理求导与序列化
    # =========================================================
    g1_vel[:-1, :] = (g1_pos[1:, :] - g1_pos[:-1, :]) / dt
    
    # 消除分段边界的速度突变
    for idx in range(1, len(modes)):
        boundary = idx * frames_per_mode
        g1_vel[boundary-1] = 0.0 
        g1_vel[boundary] = 0.0

    g1_motion_dict = {
        "pos": torch.tensor(g1_pos, dtype=torch.float32),
        "vel": torch.tensor(g1_vel, dtype=torch.float32),
        "cmd": torch.tensor(g1_cmd, dtype=torch.float32), 
        "num_frames": total_frames
    }

    torch.save(g1_motion_dict, output_file)
    print(f"🎉 马拉松专属数据集生成完毕！文件保存为: {output_file}\n")
    
    return output_file, g1_motion_dict

# ===================================================================
# 自动化质检管线
# ===================================================================
def check_marathon_dataset(file_path, data_dict):
    print("🔍 [数据体验报告] 正在进行张量级合法性校验...")
    pos, vel, cmd = data_dict["pos"], data_dict["vel"], data_dict["cmd"]
    num_frames = data_dict["num_frames"]

    assert pos.shape == vel.shape, "❌ 维度不一致！"
    assert cmd.shape == (num_frames, 3), "❌ 摇杆指令形状异常！"
    if torch.isnan(pos).any() or torch.isnan(vel).any():
        print("❌ 存在 NaN 数据！")
        return

    print(f"✅ 维度校验通过 | 总帧数: {num_frames} | 动作维度: {pos.shape[1]}")
    print(f"✅ 边界校验通过 | 最大角度: {torch.max(pos):.2f} rad | 最小角度: {torch.min(pos):.2f} rad")
    
    unique_cmds = torch.unique(cmd, dim=0)
    print(f"✅ 提取到 {len(unique_cmds)} 种马拉松阶梯配速 (包含 1.5m/s 巡航与 2.0m/s 冲刺)。")
    print("="*80 + "\n")

if __name__ == "__main__":
    out_file, motion_dict = generate_marathon_data_v4()
    check_marathon_dataset(out_file, motion_dict)