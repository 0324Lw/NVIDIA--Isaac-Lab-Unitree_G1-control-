import argparse
import torch
import numpy as np

from isaaclab.app import AppLauncher

# 1. 启动配置
parser = argparse.ArgumentParser(description="Unitree G1 Humanoid Posture Test")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = False 
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass

# ===================================================================
# 2. G1 场景配置：针对 35kg 倒立摆的动力学设置
# ===================================================================
@configclass
class G1SceneCfg(InteractiveSceneCfg):
    num_envs: int = 1
    env_spacing: float = 2.0

    # 加载转换好的 G1 USD
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/lw/IsaacLab/tutorials/03_humanoid_basics/g1.usd", # 绝对路径防丢失
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True, 
                solver_position_iteration_count=8, 
                solver_velocity_iteration_count=4
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.75), # G1 髋关节初始离地高度
            joint_pos={".*": 0.0}, # 全部关节初始化为 0
        ),
        actuators={
            # 腿部：高刚度支撑自重
            "legs": ImplicitActuatorCfg(
                joint_names_expr=[".*_hip_.*", ".*_knee_.*", ".*_ankle_.*"],
                stiffness=150.0,
                damping=5.0,
            ),
            # 上半身：低刚度柔顺控制
            "upper_body": ImplicitActuatorCfg(
                joint_names_expr=["waist_.*", ".*_shoulder_.*", ".*_elbow_.*"],
                stiffness=40.0,
                damping=2.0,
            ),
        },
    )

    # 将生成器包装为标准的场景资产
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg()
    )

    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0)
    )

# ===================================================================
# 3. 运行测试循环
# ===================================================================
def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.01)
    sim = sim_utils.SimulationContext(sim_cfg)
    
    scene_cfg = G1SceneCfg()
    scene = InteractiveScene(scene_cfg)
    
    sim.reset()
    
    robot = scene.articulations["robot"]
    sim_dt = sim.get_physics_dt()
    count = 0
    
    print("\n[✔] 宇树 G1 模型加载成功！正在执行底层 PD 站立维持...")
    print("👉 提示：你可以按住 Shift 键用鼠标左键拖拽机器人的躯干，测试其抗扰恢复能力。")
    
    while simulation_app.is_running():
        targets = robot.data.default_joint_pos.clone()
        robot.set_joint_position_target(targets)
        
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        
        count += 1
        if count % 100 == 0:
            pos = robot.data.root_pos_w[0]
            print(f"帧数: {count} | 质心高度: {pos[2]:.3f}m | 状态: 伺服锁定中")

if __name__ == "__main__":
    main()
    simulation_app.close()