#!/usr/bin/env python3

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Inspire Hand - Isaac Lab Integration with MediaPipe Control

This script creates an Isaac Lab environment for the Inspire Hand with optional MediaPipe control
using the non-sensor USD file for memory optimization on 8GB VRAM systems.

.. code-block:: bash

    ./isaaclab.sh -p lift_cube_inspire_hand_thumb_index.py --num_envs 1

"""

import argparse
import time
import threading
from typing import Dict, List

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Inspire Hand with MediaPipe control - Memory optimized.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--disable_mediapipe", action="store_true", default=False, help="Disable MediaPipe control.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(headless=args_cli.headless)
simulation_app = app_launcher.app

"""Rest everything else."""

import torch
import numpy as np

# Isaac Lab imports
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

# Import MediaPipe modules (optional)
MEDIAPIPE_AVAILABLE = False
if not args_cli.disable_mediapipe:
    try:
        from mp_read_hand import HandDetector
        import cv2
        MEDIAPIPE_AVAILABLE = True
        print("✅ MediaPipe modules loaded successfully")
    except ImportError as e:
        print(f"⚠️ MediaPipe not available: {e}")
        print("   Will use automatic demo motion instead")

##
# Scene Configuration
##

@configclass
class InspireHandSceneCfg(InteractiveSceneCfg):
    """Configuration for the Inspire Hand scene (memory optimized)."""

    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    # Inspire Hand (using non-sensor USD for memory optimization)
    inspire_hand: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/InspireHand",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/larry/NVIDIA_DEV/isaac_grasp_ws/cabinet_sensor_project/inspire_hand_with_sensors/usd/inspire_hand_processed.usd",
            activate_contact_sensors=False,  # Explicitly disable sensors for memory optimization
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.3),
            joint_pos={
                "right_index_1_joint": 0.0,
                "right_little_1_joint": 0.0,
                "right_middle_1_joint": 0.0,
                "right_ring_1_joint": 0.0,
                "right_thumb_1_joint": 0.0,
                "right_thumb_2_joint": 0.0,
            },
        ),
        actuators={
            "inspire_hand_actuators": ImplicitActuatorCfg(
                joint_names_expr=["right_.*_joint"],
                effort_limit=50.0,
                velocity_limit=1.0,
                stiffness=200.0,
                damping=20.0,
            ),
        },
    )

    # Cube to grasp
    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.1, 0.35), rot=(1.0, 0.0, 0.0, 0.0)),
    )


##
# Environment Configuration
##

@configclass
class InspireHandEnvCfg(DirectRLEnvCfg):
    """Configuration for the Inspire Hand environment."""

    # Scene settings
    scene: InspireHandSceneCfg = InspireHandSceneCfg(num_envs=1, env_spacing=2.0)
    
    # Simulation settings  
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=1.0 / 240.0,
        render_interval=4,
    )

    # Basic settings
    episode_length_s = 20.0  # Longer episodes for manual control
    decimation = 4
    action_space = 12  # 12 joints
    observation_space = 24  # 12 joint positions + 12 joint velocities
    state_space = 0


##
# MediaPipe Controller
##

class MediaPipeController:
    """MediaPipe hand control with automatic fallback."""
    
    def __init__(self, enable_mediapipe: bool = True):
        self.enable_mediapipe = enable_mediapipe and MEDIAPIPE_AVAILABLE
        self.current_command = torch.zeros(12)  # 12 joints
        self.running = False
        self.camera_thread = None
        
        if self.enable_mediapipe:
            try:
                self.detector = HandDetector()
                print("✅ MediaPipe controller initialized")
                print("🎥 Camera controls: Press 'q' to quit MediaPipe window")
            except Exception as e:
                print(f"⚠️ MediaPipe initialization failed: {e}")
                self.enable_mediapipe = False
        
        if not self.enable_mediapipe:
            print("🎮 Using automatic demo motion")
    
    def start(self):
        """Start MediaPipe control or demo mode."""
        self.running = True
        if self.enable_mediapipe:
            self.camera_thread = threading.Thread(target=self._camera_loop, daemon=True)
            self.camera_thread.start()
            print("📹 MediaPipe camera started - use your hand to control the robot!")
        else:
            print("🔄 Automatic demo motion started")
    
    def stop(self):
        """Stop control."""
        self.running = False
        if self.camera_thread and self.camera_thread.is_alive():
            self.camera_thread.join(timeout=1.0)
    
    def _camera_loop(self):
        """MediaPipe camera processing loop."""
        if not self.enable_mediapipe:
            return
            
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("📷 Camera opened successfully")
        
        while self.running:
            success, frame = cap.read()
            if not success:
                continue
                
            try:
                landmarks, annotated_frame = self.detector.detect(frame)
                cv2.imshow('MediaPipe Hand Control - Inspire Hand', annotated_frame)
                
                if landmarks and len(landmarks) > 0:
                    inspire_values = self.detector.convert_fingure_to_inspire(landmarks[0])
                    
                    # Convert MediaPipe values to joint commands (12 joints)
                    command = torch.tensor([
                        # Index finger (joints 0-2)
                        (inspire_values['index_finger'] - 500) / 500.0,
                        0.0,  # index_2
                        0.0,  # index_3
                        # Little finger (joints 3-5)
                        (inspire_values['little_finger'] - 500) / 500.0,
                        0.0,  # little_2
                        0.0,  # little_3
                        # Middle finger (joints 6-8)
                        (inspire_values['middle_finger'] - 500) / 500.0,
                        0.0,  # middle_2
                        0.0,  # middle_3
                        # Ring finger (joints 9-11)
                        (inspire_values['ring_finger'] - 500) / 500.0,
                        # Thumb (joints 10-11)
                        (inspire_values['thumb'] - 500) / 500.0,
                        0.0,  # thumb_2
                    ])
                    
                    self.current_command = torch.clamp(command, -1.0, 1.0)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
                    break
                    
            except Exception as e:
                print(f"MediaPipe error: {e}")
                continue
        
        cap.release()
        cv2.destroyAllWindows()
        print("📷 Camera released")
    
    def get_current_command(self) -> torch.Tensor:
        """Get current hand command."""
        if not self.enable_mediapipe:
            # Automatic demo motion
            t = time.time()
            return torch.tensor([
                0.3 * np.sin(t * 1.0),      # joint 0
                0.0,                        # joint 1
                0.0,                        # joint 2
                0.3 * np.sin(t * 1.2),      # joint 3
                0.0,                        # joint 4
                0.0,                        # joint 5
                0.3 * np.sin(t * 1.1),      # joint 6
                0.0,                        # joint 7
                0.0,                        # joint 8
                0.3 * np.sin(t * 1.3),      # joint 9
                0.4 * np.sin(t * 0.8),      # joint 10
                0.2 * np.sin(t * 0.9),      # joint 11
            ], dtype=torch.float32)
        
        return self.current_command.clone()


##
# Environment Implementation
##

class InspireHandEnv(DirectRLEnv):
    """Inspire Hand environment with MediaPipe control."""
    
    cfg: InspireHandEnvCfg

    def __init__(self, cfg: InspireHandEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        # Get joint indices
        self._hand_joint_indices, _ = self.scene["inspire_hand"].find_joints("right_.*_joint")
        print(f"🎯 Found hand joints: {len(self._hand_joint_indices)} joints")
        print(f"📋 Joint names: {[name for name in self.scene['inspire_hand'].data.joint_names if 'right_' in name]}")

    def _setup_scene(self):
        """Setup the scene with Inspire Hand and cube."""
        # Get the inspire hand articulation
        self.inspire_hand = self.scene["inspire_hand"]
        
        # Add ground plane using standard Isaac Lab method
        from isaaclab.sim.spawners import spawn_ground_plane
        ground_cfg = sim_utils.GroundPlaneCfg()
        spawn_ground_plane(prim_path="/World/ground", cfg=ground_cfg)
        
        # Clone environments  
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=["/World/ground"])
        
        # Add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Process actions before physics step."""
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        """Apply actions to the robot."""
        # Apply joint position commands
        self.scene["inspire_hand"].set_joint_position_target(
            self.actions, joint_ids=self._hand_joint_indices
        )

    def _get_observations(self) -> dict:
        """Get observations."""
        # Get joint states
        joint_pos = self.scene["inspire_hand"].data.joint_pos[:, self._hand_joint_indices]
        joint_vel = self.scene["inspire_hand"].data.joint_vel[:, self._hand_joint_indices]
        
        obs = torch.cat([joint_pos, joint_vel], dim=-1)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """Get rewards."""
        # Simple reward: stay alive + smooth motion
        joint_vel = self.scene["inspire_hand"].data.joint_vel[:, self._hand_joint_indices]
        smooth_penalty = -0.01 * torch.sum(torch.abs(joint_vel), dim=-1)
        return torch.ones(self.num_envs, device=self.device) + smooth_penalty

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get done flags."""
        # No termination conditions for manual control
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        time_outs = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_outs

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset environments."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        super()._reset_idx(env_ids)
        
        # Reset joint positions to default
        default_joint_pos = self.scene["inspire_hand"].data.default_joint_pos[env_ids]
        default_joint_vel = self.scene["inspire_hand"].data.default_joint_vel[env_ids]
        
        self.scene["inspire_hand"].write_joint_state_to_sim(
            default_joint_pos, default_joint_vel, None, env_ids
        )


##
# Main Function
##

def main():
    """Main function to run the Inspire Hand simulation with MediaPipe control."""
    
    # Print configuration
    print("=" * 60)
    print("🚀 Inspire Hand Isaac Lab Environment")
    print("=" * 60)
    print(f"💾 Memory optimization: Using non-sensor USD file")
    print(f"🎮 Control mode: {'MediaPipe + Demo' if MEDIAPIPE_AVAILABLE else 'Demo only'}")
    print(f"📊 Environments: {args_cli.num_envs}")
    print(f"🖥️ Headless mode: {args_cli.headless}")
    print("=" * 60)
    
    # Create environment configuration
    env_cfg = InspireHandEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    
    # Create environment using standard DirectRLEnv approach
    env = InspireHandEnv(cfg=env_cfg)
    
    print("✅ Environment created successfully!")
    
    # Initialize MediaPipe controller
    mp_controller = MediaPipeController(enable_mediapipe=not args_cli.disable_mediapipe)
    mp_controller.start()
    
    # Print control instructions
    control_dt = env_cfg.sim.dt * env_cfg.decimation
    print(f"\n📋 Simulation Configuration:")
    print(f"   Control frequency: {1/control_dt:.1f} Hz")
    print(f"   Physics step: {env_cfg.sim.dt:.4f}s")
    print(f"   Episode length: {env_cfg.episode_length_s}s")
    
    if MEDIAPIPE_AVAILABLE and not args_cli.disable_mediapipe:
        print(f"\n🎮 MediaPipe Controls:")
        print(f"   - Show your hand to the camera to control the robot")
        print(f"   - Press 'q' in camera window to quit")
        print(f"   - Fallback to demo motion if no hand detected")
    else:
        print(f"\n🎮 Demo Mode:")
        print(f"   - Automatic sinusoidal motion")
        print(f"   - Press Ctrl+C to quit")
    
    print(f"\n🚀 Starting simulation...")
    
    try:
        # Reset environment
        obs, _ = env.reset()
        print(f"📊 Observation shape: {obs['policy'].shape}")
        
        # Main simulation loop
        step_count = 0
        last_print_time = time.time()
        
        while simulation_app.is_running():
            # Get control command
            command = mp_controller.get_current_command()
            actions = command.unsqueeze(0).repeat(env.num_envs, 1).to(env.device)
            
            # Step simulation
            obs, reward, terminated, truncated, info = env.step(actions)
            
            # Print progress every 5 seconds
            current_time = time.time()
            if current_time - last_print_time >= 5.0:
                print(f"⏱️ Step {step_count}: Simulation running smoothly")
                print(f"   Average reward: {reward.mean().item():.3f}")
                print(f"   Joint range: [{obs['policy'][0, :12].min().item():.2f}, {obs['policy'][0, :12].max().item():.2f}]")
                last_print_time = current_time
            
            step_count += 1
                
    except KeyboardInterrupt:
        print("\n🛑 Simulation interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during simulation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        print("\n🧹 Cleaning up...")
        mp_controller.stop()
        env.close()
        simulation_app.close()
        print("✅ Simulation ended successfully")


if __name__ == "__main__":
    main() 