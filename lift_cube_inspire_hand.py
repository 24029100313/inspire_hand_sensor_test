#!/usr/bin/env python3

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Inspire Hand with 1061 Sensors - Isaac Lab Integration

This script creates an Isaac Lab environment for the Inspire Hand with MediaPipe control
and 1061 tactile sensor pads for real-time force feedback during grasping tasks.

Usage:
    ./isaaclab.sh -p lift_cube_inspire_hand.py --num_envs 4

"""

import argparse
import os
import sys
from typing import Dict, List
import time

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Inspire Hand with MediaPipe control and 1061 sensors.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(headless=args_cli.headless)
simulation_app = app_launcher.app

"""Rest everything else."""

import gymnasium as gym
import torch
import numpy as np
import cv2
import threading
from collections.abc import Sequence

# Isaac Lab imports
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.managers import ActionTermCfg as ActionTerm, SceneEntityCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
import isaaclab.envs.mdp as mdp
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as Gnoise

# Import MediaPipe modules
from mp_read_hand import HandDetector

##
# Pre-defined configs
##
import isaaclab_tasks.manager_based.manipulation.inhand.mdp as inhand_mdp


@configclass
class InspireHandSceneCfg(InteractiveSceneCfg):
    """Configuration for the Inspire Hand scene."""

    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # Lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    # Inspire Hand with 1061 sensors
    inspire_hand: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/InspireHand",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/larry/NVIDIA_DEV/isaac_grasp_ws/cabinet_sensor_project/inspire_hand_with_sensors/usd/inspire_hand_processed_with_pads.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
            ),
            activate_contact_sensors=True,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.3),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={
                "left_index_1_joint": 0.0,
                "left_little_1_joint": 0.0,
                "left_middle_1_joint": 0.0,
                "left_ring_1_joint": 0.0,
                "left_thumb_1_joint": 0.0,
                "left_thumb_swing_joint": 0.0,
            },
        ),
        actuators={
            "inspire_hand_actuators": ImplicitActuatorCfg(
                joint_names_expr=["left_index_1_joint", "left_little_1_joint", "left_middle_1_joint", 
                                 "left_ring_1_joint", "left_thumb_1_joint", "left_thumb_swing_joint"],
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
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.1),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    
    joint_pos = inhand_mdp.EMAJointPositionToLimitsActionCfg(
        asset_name="inspire_hand",
        joint_names=[".*"],
        alpha=0.95,
        rescale_to_limits=True,
    )

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyObsGroupCfg(ObsGroup):
        """Observations for policy group."""
        
        # robot terms
        joint_pos = ObsTerm(func=mdp.joint_pos_limit_normalized, noise=Gnoise(std=0.005))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.2, noise=Gnoise(std=0.01))
        
        # object terms
        object_pos = ObsTerm(
            func=mdp.root_pos_w, noise=Gnoise(std=0.002), params={"asset_cfg": SceneEntityCfg("cube")}
        )
        object_quat = ObsTerm(
            func=mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("cube"), "make_quat_unique": False}
        )
        
        # action terms
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyObsGroupCfg = PolicyObsGroupCfg()


@configclass 
class InspireHandEnvCfg(ManagerBasedEnvCfg):
    """Configuration for the Inspire Hand environment."""

    # Scene settings
    scene: InspireHandSceneCfg = InspireHandSceneCfg(num_envs=1, env_spacing=2.0)
    
    # Simulation settings
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=1.0 / 240.0,
        render_interval=4,
    )

    # Basic settings
    episode_length_s = 10.0
    decimation = 4
    
    # MDP settings
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()


def add_sensor_configs_to_scene(env_cfg: InspireHandEnvCfg):
    """Add ALL 1061 sensor configurations to enable complete tactile feedback."""
    
    # CORRECT sensor groups configuration - ALL 1061 sensors based on actual USD structure
    # Based on the accurate sensor layout provided by user
    full_sensor_groups = {
        # Palm sensors
        "palm_sensor": {"count": 112, "grid": (14, 8), "size": "3.0×3.0×0.6mm"},
        
        # Thumb sensors (Total: 96 + 8 + 96 + 9 = 209)
        "thumb_sensor_1": {"count": 96, "grid": (8, 12), "size": "1.2×1.2×0.6mm"},
        "thumb_sensor_2": {"count": 8, "grid": (2, 4), "size": "1.2×1.2×0.6mm"},
        "thumb_sensor_3": {"count": 96, "grid": (8, 12), "size": "1.2×1.2×0.6mm"},
        "thumb_sensor_4": {"count": 9, "grid": (3, 3), "size": "1.2×1.2×0.6mm"},
        
        # Index finger sensors (Total: 80 + 96 + 9 = 185)
        "index_sensor_1": {"count": 80, "grid": (8, 10), "size": "1.2×1.2×0.6mm"},
        "index_sensor_2": {"count": 96, "grid": (8, 12), "size": "1.2×1.2×0.6mm"},
        "index_sensor_3": {"count": 9, "grid": (3, 3), "size": "1.2×1.2×0.6mm"},
        
        # Middle finger sensors (Total: 80 + 96 + 9 = 185)
        "middle_sensor_1": {"count": 80, "grid": (10, 8), "size": "1.2×1.2×0.6mm"},
        "middle_sensor_2": {"count": 96, "grid": (8, 12), "size": "1.2×1.2×0.6mm"},
        "middle_sensor_3": {"count": 9, "grid": (3, 3), "size": "1.2×1.2×0.6mm"},
        
        # Ring finger sensors (Total: 80 + 96 + 9 = 185)
        "ring_sensor_1": {"count": 80, "grid": (8, 10), "size": "1.2×1.2×0.6mm"},
        "ring_sensor_2": {"count": 96, "grid": (8, 12), "size": "1.2×1.2×0.6mm"},
        "ring_sensor_3": {"count": 9, "grid": (3, 3), "size": "1.2×1.2×0.6mm"},
        
        # Little finger sensors (Total: 80 + 96 + 9 = 185)
        "little_sensor_1": {"count": 80, "grid": (8, 10), "size": "1.2×1.2×0.6mm"},
        "little_sensor_2": {"count": 96, "grid": (8, 12), "size": "1.2×1.2×0.6mm"},
        "little_sensor_3": {"count": 9, "grid": (3, 3), "size": "1.2×1.2×0.6mm"},
    }
    
    # Calculate current total
    current_total = sum(config['count'] for config in full_sensor_groups.values())
    print(f"Current sensor count: {current_total}")
    print("Breakdown:")
    print(f"  - Palm: 112")
    print(f"  - Thumb: 96 + 8 + 96 + 9 = 209")
    print(f"  - Index: 80 + 96 + 9 = 185")
    print(f"  - Middle: 80 + 96 + 9 = 185") 
    print(f"  - Ring: 80 + 96 + 9 = 185")
    print(f"  - Little: 80 + 96 + 9 = 185")
    print(f"  - Total: {current_total}")
    
    # Check if we need additional sensors to reach 1061
    if current_total < 1061:
        remaining = 1061 - current_total
        print(f"Need {remaining} additional sensors to reach 1061 total")
        
        # Add wrist/base sensors for the remaining count
        full_sensor_groups["wrist_sensor"] = {"count": remaining, "grid": "auto", "size": "1.2×1.2×0.6mm"}
        print(f"Added wrist_sensor with {remaining} sensors")
    
    # Verify final total
    final_total = sum(config['count'] for config in full_sensor_groups.values())
    print(f"✅ Final sensor count: {final_total} (target: 1061)")
    
    # Add contact sensor configurations for ALL sensor pads
    # Using the reference configuration from lift_cube_sm_with_sensors.py
    for sensor_group_name, config in full_sensor_groups.items():
        for pad_id in range(1, config["count"] + 1):
            sensor_pad_name = f"{sensor_group_name}_pad_{pad_id:03d}"
            
            # Configure each sensor pad with ContactSensorCfg (same as reference)
            setattr(env_cfg.scene, f"contact_{sensor_pad_name}", ContactSensorCfg(
                prim_path=f"{{ENV_REGEX_NS}}/InspireHand/{sensor_pad_name}",
                track_pose=True,
                update_period=0.0,  # Use simulation timestep for real-time feedback
                debug_vis=False,   # Disable visualization for performance with 1061 sensors
                filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube"],  # Detect contact with cube
            ))
    
    print(f"✅ Successfully configured ALL {final_total} sensor pads for complete tactile feedback")
    print(f"📊 Accurate sensor distribution:")
    for group_name, config in full_sensor_groups.items():
        print(f"   - {group_name}: {config['count']} sensors {config.get('grid', 'auto')} ({config['size']})")
    
    return final_total


class InspireHandSensorManager:
    """Manager for ALL 1061 sensor pads on the Inspire Hand for complete tactile feedback."""
    
    def __init__(self, env: ManagerBasedEnv, device: torch.device):
        self.env = env
        self.device = device
        self.sensor_names = []
        self._initialize_sensors()
        
    def _initialize_sensors(self):
        """Initialize ALL 1061 sensor configurations for complete tactile feedback."""
        # Complete sensor groups matching the actual USD structure
        full_sensor_groups = {
            # Palm sensors
            "palm_sensor": {"count": 112, "grid": (14, 8), "size": "3.0×3.0×0.6mm"},
            
            # Thumb sensors (Total: 96 + 8 + 96 + 9 = 209)
            "thumb_sensor_1": {"count": 96, "grid": (8, 12), "size": "1.2×1.2×0.6mm"},
            "thumb_sensor_2": {"count": 8, "grid": (2, 4), "size": "1.2×1.2×0.6mm"},
            "thumb_sensor_3": {"count": 96, "grid": (8, 12), "size": "1.2×1.2×0.6mm"},
            "thumb_sensor_4": {"count": 9, "grid": (3, 3), "size": "1.2×1.2×0.6mm"},
            
            # Index finger sensors (Total: 80 + 96 + 9 = 185)
            "index_sensor_1": {"count": 80, "grid": (8, 10), "size": "1.2×1.2×0.6mm"},
            "index_sensor_2": {"count": 96, "grid": (8, 12), "size": "1.2×1.2×0.6mm"},
            "index_sensor_3": {"count": 9, "grid": (3, 3), "size": "1.2×1.2×0.6mm"},
            
            # Middle finger sensors (Total: 80 + 96 + 9 = 185)
            "middle_sensor_1": {"count": 80, "grid": (10, 8), "size": "1.2×1.2×0.6mm"},
            "middle_sensor_2": {"count": 96, "grid": (8, 12), "size": "1.2×1.2×0.6mm"},
            "middle_sensor_3": {"count": 9, "grid": (3, 3), "size": "1.2×1.2×0.6mm"},
            
            # Ring finger sensors (Total: 80 + 96 + 9 = 185)
            "ring_sensor_1": {"count": 80, "grid": (8, 10), "size": "1.2×1.2×0.6mm"},
            "ring_sensor_2": {"count": 96, "grid": (8, 12), "size": "1.2×1.2×0.6mm"},
            "ring_sensor_3": {"count": 9, "grid": (3, 3), "size": "1.2×1.2×0.6mm"},
            
            # Little finger sensors (Total: 80 + 96 + 9 = 185)
            "little_sensor_1": {"count": 80, "grid": (8, 10), "size": "1.2×1.2×0.6mm"},
            "little_sensor_2": {"count": 96, "grid": (8, 12), "size": "1.2×1.2×0.6mm"},
            "little_sensor_3": {"count": 9, "grid": (3, 3), "size": "1.2×1.2×0.6mm"},
        }
        
        # Calculate current total
        current_total = sum(config['count'] for config in full_sensor_groups.values())
        
        # Add remaining sensors if needed to reach exactly 1061
        if current_total < 1061:
            remaining = 1061 - current_total
            full_sensor_groups["wrist_sensor"] = {"count": remaining, "grid": "auto", "size": "1.2×1.2×0.6mm"}
        
        # Generate all sensor names
        for group_name, info in full_sensor_groups.items():
            for i in range(1, info["count"] + 1):
                sensor_name = f"{group_name}_pad_{i:03d}"
                self.sensor_names.append(sensor_name)
        
        total_sensors = len(self.sensor_names)
        print(f"Initialized {total_sensors} sensor pads for complete tactile feedback")
        
        # Create sensor groupings for efficient processing
        self.sensor_groups = {
            "palm": [name for name in self.sensor_names if name.startswith("palm_sensor")],
            "thumb": [name for name in self.sensor_names if name.startswith("thumb_sensor")],
            "index": [name for name in self.sensor_names if name.startswith("index_sensor")],
            "middle": [name for name in self.sensor_names if name.startswith("middle_sensor")],
            "ring": [name for name in self.sensor_names if name.startswith("ring_sensor")],
            "little": [name for name in self.sensor_names if name.startswith("little_sensor")],
            "wrist": [name for name in self.sensor_names if name.startswith("wrist_sensor")],
        }
    
    def get_sensor_data(self, sensor_name: str):
        """Get force data for a specific sensor."""
        if hasattr(self.env.scene, f"contact_{sensor_name}"):
            sensor = getattr(self.env.scene, f"contact_{sensor_name}")
            return sensor.data.force_matrix_w
        return None
    
    def get_all_sensor_data(self) -> Dict[str, torch.Tensor]:
        """Get force data for all sensors."""
        sensor_data = {}
        for sensor_name in self.sensor_names:
            data = self.get_sensor_data(sensor_name)
            if data is not None:
                sensor_data[sensor_name] = data
        return sensor_data
    
    def print_sensor_summary(self, env_id: int = 0):
        """Print a comprehensive summary of all 1061 sensor forces by groups."""
        all_data = self.get_all_sensor_data()
        
        print(f"=== Inspire Hand Complete Sensor Summary (Environment {env_id}) ===")
        print(f"Total sensors: {len(self.sensor_names)} (Complete 1061 sensor array)")
        
        group_stats = {}
        total_force = 0.0
        total_active_sensors = 0
        
        # Analyze each sensor group
        for group_name, sensor_list in self.sensor_groups.items():
            group_force = 0.0
            group_active = 0
            group_max_force = 0.0
            
            for sensor_name in sensor_list:
                if sensor_name in all_data and len(all_data[sensor_name]) > env_id:
                    force_magnitude = float(torch.norm(all_data[sensor_name][env_id]).cpu().numpy())
                    group_force += force_magnitude
                    total_force += force_magnitude
                    
                    if force_magnitude > 0.01:  # 0.01N threshold
                        group_active += 1
                        total_active_sensors += 1
                        
                    if force_magnitude > group_max_force:
                        group_max_force = force_magnitude
            
            group_stats[group_name] = {
                "total_sensors": len(sensor_list),
                "active_sensors": group_active,
                "total_force": group_force,
                "max_force": group_max_force,
                "avg_force": group_force / max(len(sensor_list), 1)
            }
        
        # Print detailed group analysis
        print("\n📊 Sensor Group Analysis:")
        for group_name, stats in group_stats.items():
            if stats["total_sensors"] > 0:
                activity_rate = (stats["active_sensors"] / stats["total_sensors"]) * 100
                print(f"  🖐️ {group_name.upper():>8}: {stats['active_sensors']:>3}/{stats['total_sensors']:>3} active ({activity_rate:>5.1f}%) | "
                      f"Force: {stats['total_force']:>6.2f}N total, {stats['max_force']:>5.2f}N max, {stats['avg_force']:>5.3f}N avg")
        
        print(f"\n🔍 Overall Statistics:")
        print(f"  • Total active sensors: {total_active_sensors}/1061 ({(total_active_sensors/1061)*100:.1f}%)")
        print(f"  • Total force magnitude: {total_force:.3f}N")
        print(f"  • Average force per active sensor: {total_force/max(total_active_sensors, 1):.3f}N")
        print(f"  • System load: {'🟢 Light' if total_active_sensors < 100 else '🟡 Medium' if total_active_sensors < 300 else '🔴 Heavy'}")
        print("-" * 80)
        
        return group_stats
    
    def get_sensor_group_data(self, group_name: str, env_id: int = 0) -> Dict[str, float]:
        """Get aggregated sensor data for a specific group (palm, thumb, index, etc.)."""
        if group_name not in self.sensor_groups:
            return {}
        
        all_data = self.get_all_sensor_data()
        group_data = {
            "total_force": 0.0,
            "max_force": 0.0,
            "active_sensors": 0,
            "sensor_count": len(self.sensor_groups[group_name]),
            "forces": []
        }
        
        for sensor_name in self.sensor_groups[group_name]:
            if sensor_name in all_data and len(all_data[sensor_name]) > env_id:
                force_magnitude = float(torch.norm(all_data[sensor_name][env_id]).cpu().numpy())
                group_data["forces"].append(force_magnitude)
                group_data["total_force"] += force_magnitude
                
                if force_magnitude > group_data["max_force"]:
                    group_data["max_force"] = force_magnitude
                    
                if force_magnitude > 0.01:  # 0.01N threshold
                    group_data["active_sensors"] += 1
        
        return group_data


class MediaPipeController:
    """MediaPipe-based hand control for Inspire Hand."""
    
    def __init__(self, target_hand="left"):
        self.target_hand = target_hand
        self.detector = HandDetector(target_hand=target_hand)
        self.camera_thread = None
        self.running = False
        self.current_hand_command = None
        self.camera_lock = threading.Lock()
        
    def start_camera_control(self):
        """Start the camera control thread."""
        if not self.running:
            self.running = True
            self.camera_thread = threading.Thread(target=self._camera_loop)
            self.camera_thread.daemon = True
            self.camera_thread.start()
            print("MediaPipe camera control started")
    
    def stop_camera_control(self):
        """Stop the camera control thread."""
        self.running = False
        if self.camera_thread and self.camera_thread.is_alive():
            self.camera_thread.join(timeout=2.0)
        print("MediaPipe camera control stopped")
    
    def _camera_loop(self):
        """Main camera processing loop."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
            
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        while self.running:
            success, frame = cap.read()
            if not success:
                continue
                
            # Detect hand landmarks
            landmarks, annotated_frame = self.detector.detect(frame)
            
            if landmarks and len(landmarks) > 0:
                # Convert to inspire hand values
                inspire_values = self.detector.convert_fingure_to_inspire(landmarks[0])
                if inspire_values:
                    # Convert to torch tensor and normalize to [-1, 1] range for Isaac Lab
                    command = torch.tensor([
                        (inspire_values['little_finger'] - 500) / 500.0,    # Little finger
                        (inspire_values['ring_finger'] - 500) / 500.0,      # Ring finger  
                        (inspire_values['middle_finger'] - 500) / 500.0,    # Middle finger
                        (inspire_values['index_finger'] - 500) / 500.0,     # Index finger
                        (inspire_values['thumb'] - 500) / 500.0,            # Thumb
                        (inspire_values['wrist'] - 500) / 500.0,            # Wrist
                    ], dtype=torch.float32)
                    
                    with self.camera_lock:
                        self.current_hand_command = command
            
            # Display the frame
            cv2.imshow('MediaPipe Hand Control', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def get_current_hand_command(self):
        """Get the current hand command from MediaPipe."""
        with self.camera_lock:
            return self.current_hand_command


def main():
    """Main function to run the Inspire Hand environment with MediaPipe control."""
    
    # Create environment configuration
    env_cfg = InspireHandEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    
    # Add reduced sensor configurations
    add_sensor_configs_to_scene(env_cfg)
    
    # Create environment
    env = ManagerBasedEnv(cfg=env_cfg)
    
    # Initialize sensor manager
    sensor_manager = InspireHandSensorManager(env, env.device)
    
    # Initialize MediaPipe controller
    mp_controller = MediaPipeController(target_hand="left")
    
    # Start MediaPipe camera control if not in headless mode
    if not args_cli.headless:
        mp_controller.start_camera_control()
    
    print("Starting Inspire Hand simulation with MediaPipe control...")
    print(f"Using ALL {len(sensor_manager.sensor_names)} sensors for complete tactile feedback")
    print("Press 'q' in the MediaPipe window to quit, or Ctrl+C in terminal")
    
    try:
        # Reset environment
        obs, _ = env.reset()
        
        # Main simulation loop
        step_count = 0
        while simulation_app.is_running():
            
            # Get MediaPipe hand command
            hand_command = mp_controller.get_current_hand_command()
            
            if hand_command is not None:
                # Apply hand command to the robot
                actions = {"inspire_hand_action": hand_command.unsqueeze(0).repeat(env.num_envs, 1)}
                obs, _, _, _, _ = env.step(actions)
            else:
                # Default action (open hand)
                default_action = torch.zeros((env.num_envs, 6), device=env.device)
                actions = {"inspire_hand_action": default_action}
                obs, _, _, _, _ = env.step(actions)
            
            # Print comprehensive sensor data every 50 steps for real-time monitoring
            if step_count % 50 == 0:
                sensor_manager.print_sensor_summary(env_id=0)
            
            step_count += 1
            
    except KeyboardInterrupt:
        print("Interrupted by user")
    
    finally:
        # Clean up
        mp_controller.stop_camera_control()
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
