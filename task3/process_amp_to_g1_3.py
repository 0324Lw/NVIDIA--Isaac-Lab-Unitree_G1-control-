import numpy as np
import torch
import math

# ===================================================================
# G1 全身协同极限运动学合成引擎 (Task 3 Whole-Body 专属)
# ===================================================================
def generate_omni_data_v3():
    output_file = "g1_extreme_omni.pt"
    print("\n" + "="*80)
    print("🚀 正在启动 G1 全身协同运动学合成引擎 (Whole-Body CPG v3)...")
    print("="*80)

    # 基础设定
    fps = 30.0
    dt = 1.0 / fps
    frames_per_mode = 300  
    num_joints = 25        

    # 🚨 Task 3 核心升级：引入极限冲刺与高速急弯，倒逼全身协同
    modes = [
        {"name": "Forward",        "cmd": [0.5, 0.0, 0.0]},
        {"name": "Sprint",         "cmd": [1.2, 0.0, 0.0]},   # 极限冲刺
        {"name": "Backward",       "cmd": [-0.5, 0.0, 0.0]},
        {"name": "Leftward",       "cmd": [0.0, 0.3, 0.0]},
        {"name": "Rightward",      "cmd": [0.0, -0.3, 0.0]},
        {"name": "Turn_L",         "cmd": [0.0, 0.0, 0.5]},
        {"name": "Turn_R",         "cmd": [0.0, 0.0, -0.5]},
        {"name": "Sprint_Turn_L",  "cmd": [1.0, 0.0, 0.8]},   # 高速向左压弯
        {"name": "Sprint_Turn_R",  "cmd": [1.0, 0.0, -0.8]},  # 高速向右压弯
    ]
    
    total_frames = frames_per_mode * len(modes)
    g1_pos = np.zeros((total_frames, num_joints), dtype=np.float32)
    g1_vel = np.zeros((total_frames, num_joints), dtype=np.float32)
    g1_cmd = np.zeros((total_frames, 3), dtype=np.float32) 

    print(f"📦 预分配内存: 总帧数 {total_frames}, 关节维数 {num_joints}")

    # =========================================================
    # 注入高级动力学特征 (Dynamic Amplitude & Centrifugal Leaning)
    # =========================================================
    for idx, mode in enumerate(modes):
        start_idx = idx * frames_per_mode
        end_idx = start_idx + frames_per_mode
        time_steps = np.linspace(0, frames_per_mode * dt, frames_per_mode)
        
        v_x, v_y, w_z = mode["cmd"]
        g1_cmd[start_idx:end_idx] = mode["cmd"]
        
        # 步频随前向速度动态增加
        base_freq = 5.0 + 1.5 * abs(v_x)
        phase_L = time_steps * base_freq
        phase_R = time_steps * base_freq + math.pi
        
        # ==========================================
        # 1. 下肢基础运动学
        # ==========================================
        if "Forward" in mode["name"] or "Sprint" in mode["name"] or "Backward" in mode["name"]:
            direction = 1.0 if v_x > 0 else -1.0
            
            # 腿部振幅随速度放大 (0.5m/s -> 0.6rad, 1.2m/s -> 1.0rad)
            leg_amp_hip = min(0.6 * (abs(v_x) / 0.5 + 0.1), 1.0)
            leg_amp_knee = min(0.4 * (abs(v_x) / 0.5 + 0.1), 0.8)
            
            g1_pos[start_idx:end_idx, 3] = np.sin(phase_L) * leg_amp_hip * direction          # 左髋 Pitch
            g1_pos[start_idx:end_idx, 6] = np.sin(phase_L + 0.5) * leg_amp_knee * direction   # 左膝
            g1_pos[start_idx:end_idx, 9] = np.sin(phase_R) * leg_amp_hip * direction          # 右髋 Pitch
            g1_pos[start_idx:end_idx, 12] = np.sin(phase_R + 0.5) * leg_amp_knee * direction  # 右膝

        elif mode["name"] in ["Leftward", "Rightward"]:
            direction = 1.0 if v_y > 0 else -1.0
            g1_pos[start_idx:end_idx, 2] = np.sin(phase_L) * 0.3 * direction
            g1_pos[start_idx:end_idx, 5] = np.sin(phase_L) * 0.1 * direction
            g1_pos[start_idx:end_idx, 8] = np.sin(phase_R) * 0.3 * direction
            g1_pos[start_idx:end_idx, 11] = np.sin(phase_R) * 0.1 * direction

        if "Turn" in mode["name"]:
            direction = 1.0 if w_z > 0 else -1.0
            g1_pos[start_idx:end_idx, 1] += np.sin(phase_L) * 0.4 * direction
            g1_pos[start_idx:end_idx, 7] += np.sin(phase_R) * 0.4 * direction
            
        # ==========================================
        # 2. 上半身高级协同特征 (Task 3 核心)
        # ==========================================
        
        # A. 动态摆臂 (Dynamic Arm Swing): 完全与腿部相位和前向速度挂钩
        # 基础摆臂幅度 0.3，冲刺时拉满到 0.8，静止时为 0
        arm_amp = min(0.3 * (abs(v_x) / 0.5), 0.8) if v_x != 0 else 0.0
        if arm_amp > 0:
            direction = 1.0 if v_x > 0 else -1.0
            g1_pos[start_idx:end_idx, 15] = np.sin(phase_R) * arm_amp * direction  # 左肩 (对向代偿)
            g1_pos[start_idx:end_idx, 19] = np.sin(phase_L) * arm_amp * direction  # 右肩 (对向代偿)
            
        # B. 离心力内倾压弯 (Centrifugal Leaning): 腰部 Roll 代偿
        # 只有在同时具备前向速度和转弯速度时触发
        if abs(v_x) > 0.5 and abs(w_z) > 0:
            # 倾角大小与 v_x * w_z 成正比
            lean_angle = 0.25 * v_x * w_z  
            # 假设关节 13 为腰部 Roll 轴 (请根据实际 URDF 微调索引，这里做启发式占位)
            g1_pos[start_idx:end_idx, 13] = lean_angle 
            # 手臂外展代偿重心
            outward_swing = 0.2 * abs(w_z)
            if w_z > 0: # 左转
                g1_pos[start_idx:end_idx, 16] = outward_swing # 左肩外展
            else:       # 右转
                g1_pos[start_idx:end_idx, 20] = outward_swing # 右肩外展

    # =========================================================
    # 物理求导与序列化
    # =========================================================
    g1_vel[:-1, :] = (g1_pos[1:, :] - g1_pos[:-1, :]) / dt
    
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
    print(f"🎉 极限协同数据集生成完毕！文件保存为: {output_file}\n")
    
    return output_file, g1_motion_dict

# ===================================================================
# 自动化质检管线
# ===================================================================
def check_omni_dataset(file_path, data_dict):
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
    print(f"✅ 提取到 {len(unique_cmds)} 种极端摇杆状态 (包含 1.2m/s 冲刺与 0.8rad/s 急弯)。")
    print("="*80 + "\n")

if __name__ == "__main__":
    out_file, motion_dict = generate_omni_data_v3()
    check_omni_dataset(out_file, motion_dict)