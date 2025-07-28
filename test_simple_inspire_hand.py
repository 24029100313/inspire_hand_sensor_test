#!/usr/bin/env python3

import argparse
import os
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Simple Inspire Hand test.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(headless=args_cli.headless)
simulation_app = app_launcher.app

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

@configclass
class SimpleSceneCfg(InteractiveSceneCfg):
    """Simple scene configuration."""
    
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    
    inspire_hand: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/InspireHand",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/larry/NVIDIA_DEV/isaac_grasp_ws/cabinet_sensor_project/inspire_hand_with_sensors/usd/inspire_hand_processed_with_pads.usd",
            activate_contact_sensors=True,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.3),
            joint_pos={".*": 0.0},
        ),
        actuators={
            "inspire_hand_actuators": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                effort_limit=50.0,
                velocity_limit=1.0,
                stiffness=200.0,
                damping=20.0,
            ),
        },
    )

@configclass
class SimpleEnvCfg(ManagerBasedEnvCfg):
    """Minimal environment configuration."""
    
    scene: SimpleSceneCfg = SimpleSceneCfg(num_envs=1, env_spacing=2.0)
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(dt=1.0/240.0, render_interval=4)
    episode_length_s = 10.0
    decimation = 4
    
    # Empty managers to satisfy validation
    actions = {}
    observations = {"policy": {"joint_pos": {"func": lambda: torch.zeros(6)}}}

def main():
    """Test the basic Inspire Hand setup."""
    print("�� Testing basic Inspire Hand setup...")
    
    # Create environment
    env_cfg = SimpleEnvCfg()
    print(f"✅ Environment configuration created")
    
    try:
        env = ManagerBasedEnv(cfg=env_cfg)
        print(f"✅ Environment created successfully!")
        
        # Reset and test
        obs, _ = env.reset()
        print(f"✅ Environment reset successful!")
        print(f"📊 Scene objects: {list(env.scene.keys())}")
        
        # Run a few steps
        for i in range(10):
            action = torch.zeros((env.num_envs, 0), device=env.device)  # Empty action
            obs, _, _, _, _ = env.step(action)
            if i % 5 == 0:
                print(f"✅ Step {i} completed")
        
        print("🎉 Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()
        simulation_app.close()

if __name__ == "__main__":
    main()
