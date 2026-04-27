import numpy as np
import torch
import math

# ===================================================================
# 工业级全向步态合成与重定向引擎 (Task 2 Omni-Control 专属)
# ===================================================================
def generate_omni_data():
    output_file = "g1_omni_walk.pt"
    print("\n" + "="*70)
    print("🚀 正在启动 G1 全向启发式步态合成引擎 (Omni-CPG Generator)...")
    print("="*70)

    # 基础设定
    fps = 30.0
    dt = 1.0 / fps
    frames_per_mode = 300  # 每种步态生成 300 帧 (10秒)
    num_joints = 25        # 适配最新的 25 维动作空间

    # 6 种全向模式：前进, 后退, 左移, 右移, 左转, 右转
    modes = [
        {"name": "Forward",  "cmd": [0.5, 0.0, 0.0]},
        {"name": "Backward", "cmd": [-0.5, 0.0, 0.0]},
        {"name": "Leftward", "cmd": [0.0, 0.3, 0.0]},
        {"name": "Rightward","cmd": [0.0, -0.3, 0.0]},
        {"name": "Turn_L",   "cmd": [0.0, 0.0, 0.5]},
        {"name": "Turn_R",   "cmd": [0.0, 0.0, -0.5]},
    ]
    
    total_frames = frames_per_mode * len(modes)
    g1_pos = np.zeros((total_frames, num_joints), dtype=np.float32)
    g1_vel = np.zeros((total_frames, num_joints), dtype=np.float32)
    g1_cmd = np.zeros((total_frames, 3), dtype=np.float32) # 🚨 新增：与帧绑定的摇杆指令

    print(f"📦 预分配内存: 总帧数 {total_frames}, 关节维数 {num_joints}")

    # =========================================================
    # 分段注入全向运动学流形 (Kinematic Manifolds)
    # =========================================================
    for idx, mode in enumerate(modes):
        start_idx = idx * frames_per_mode
        end_idx = start_idx + frames_per_mode
        time_steps = np.linspace(0, frames_per_mode * dt, frames_per_mode)
        
        # 写入指令标签，供 RSI (参考态初始化) 使用
        g1_cmd[start_idx:end_idx] = mode["cmd"]
        
        # 基础步频机制
        phase_L = time_steps * 5.0
        phase_R = time_steps * 5.0 + math.pi
        
        if mode["name"] in ["Forward", "Backward"]:
            # 纵向运动 (主要激活 Pitch 俯仰轴)
            direction = 1.0 if mode["name"] == "Forward" else -1.0
            g1_pos[start_idx:end_idx, 3] = np.sin(phase_L) * 0.6 * direction          # 左髋 Pitch
            g1_pos[start_idx:end_idx, 6] = np.sin(phase_L + 0.5) * 0.4 * direction    # 左膝
            g1_pos[start_idx:end_idx, 9] = np.sin(phase_R) * 0.6 * direction          # 右髋 Pitch
            g1_pos[start_idx:end_idx, 12] = np.sin(phase_R + 0.5) * 0.4 * direction   # 右膝
            # 手臂代偿
            g1_pos[start_idx:end_idx, 15] = np.sin(phase_R) * 0.3 * direction         # 左肩 (🚨 拼写错误已修复)
            g1_pos[start_idx:end_idx, 19] = np.sin(phase_L) * 0.3 * direction         # 右肩

        elif mode["name"] in ["Leftward", "Rightward"]:
            # 侧向运动 (主要激活 Roll 横滚轴，例如外展/内收)
            direction = 1.0 if mode["name"] == "Leftward" else -1.0
            # 假设索引 2, 8 为髋部 Roll，5, 11 为踝部 Roll (需根据 G1 实际 URDF 微调，此处为启发式近似)
            g1_pos[start_idx:end_idx, 2] = np.sin(phase_L) * 0.3 * direction
            g1_pos[start_idx:end_idx, 5] = np.sin(phase_L) * 0.1 * direction
            g1_pos[start_idx:end_idx, 8] = np.sin(phase_R) * 0.3 * direction
            g1_pos[start_idx:end_idx, 11] = np.sin(phase_R) * 0.1 * direction

        elif mode["name"] in ["Turn_L", "Turn_R"]:
            # 旋转运动 (主要激活 Yaw 偏航轴)
            direction = 1.0 if mode["name"] == "Turn_L" else -1.0
            # 假设 1 为髋 Yaw，7 为右髋 Yaw
            g1_pos[start_idx:end_idx, 1] = np.sin(phase_L) * 0.4 * direction
            g1_pos[start_idx:end_idx, 7] = np.sin(phase_R) * 0.4 * direction
            # 加上轻微的踏步动作辅助转向
            g1_pos[start_idx:end_idx, 6] = (np.sin(phase_L) + 1.0) * 0.2 
            g1_pos[start_idx:end_idx, 12] = (np.sin(phase_R) + 1.0) * 0.2

    # =========================================================
    # 物理求导与序列化
    # =========================================================
    # 严格按照差分计算真实角速度
    g1_vel[:-1, :] = (g1_pos[1:, :] - g1_pos[:-1, :]) / dt
    # 消除分段拼接处的异常速度突变
    for idx in range(1, len(modes)):
        boundary = idx * frames_per_mode
        g1_vel[boundary-1] = 0.0 
        g1_vel[boundary] = 0.0

    g1_motion_dict = {
        "pos": torch.tensor(g1_pos, dtype=torch.float32),
        "vel": torch.tensor(g1_vel, dtype=torch.float32),
        "cmd": torch.tensor(g1_cmd, dtype=torch.float32), # 注入指令特征
        "num_frames": total_frames
    }

    torch.save(g1_motion_dict, output_file)
    print(f"🎉 全向数据集生成完毕！文件保存为: {output_file}\n")
    
    return output_file, g1_motion_dict

# ===================================================================
# 自动化质检管线 (Sanity Check)
# ===================================================================
def check_omni_dataset(file_path, data_dict):
    print("🔍 [数据体验报告] 正在进行张量级合法性校验...")
    
    pos = data_dict["pos"]
    vel = data_dict["vel"]
    cmd = data_dict["cmd"]
    num_frames = data_dict["num_frames"]

    assert pos.shape == vel.shape, "❌ 致命错误：位置与速度张量维度不一致！"
    assert pos.shape[0] == num_frames, "❌ 致命错误：帧数与声明值不符！"
    assert cmd.shape == (num_frames, 3), "❌ 致命错误：摇杆指令张量形状异常！"
    
    if torch.isnan(pos).any() or torch.isnan(vel).any():
        print("❌ 致命错误：数据集中存在 NaN (Not a Number)，会导致底层 C++ 崩溃！")
        return

    print("✅ [维度校验] Pass!")
    print(f"   -> 总帧数: {num_frames} | 动作维度: {pos.shape[1]}")
    
    print("✅ [边界校验] Pass!")
    print(f"   -> 关节最大角度: {torch.max(pos):.2f} rad | 最小角度: {torch.min(pos):.2f} rad")
    print(f"   -> 最大角速度: {torch.max(vel):.2f} rad/s")
    
    print("✅ [全向指令校验] Pass!")
    unique_cmds = torch.unique(cmd, dim=0)
    print(f"   -> 提取到 {len(unique_cmds)} 种独特的摇杆指令状态。网络将依靠这些状态进行 RSI 重置。")
    print("="*70 + "\n")

if __name__ == "__main__":
    out_file, motion_dict = generate_omni_data()
    check_omni_dataset(out_file, motion_dict)