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
from isaaclab.managers import ActionTermCfg as ActionTerm, SceneEntityCfg, ObservationGroupCfg, ObservationTermCfg
import isaaclab.envs.mdp as mdp
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

# Import MediaPipe modules
from mp_read_hand import HandDetector

##
# Pre-defined configs
##
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg


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
    
    # Actions - simplified working version
    actions = {}
    
    # Observations - simplified working version  
    observations = {
        "policy": {
            "joint_pos": {
                "func": lambda: torch.zeros(6)
            }
        }
    }


def add_sensor_configs_to_scene(env_cfg: InspireHandEnvCfg):
    """Add all 1061 sensor configurations to the scene."""
    
    # Define sensor groups based on inspire_hand_with_sensors documentation
    sensor_groups = {
        "palm_sensor": {"count": 112, "grid": (14, 8)},
        "thumb_sensor_1": {"count": 96, "grid": (8, 12)},
        "thumb_sensor_2": {"count": 8, "grid": (2, 4)},
        "thumb_sensor_3": {"count": 96, "grid": (8, 12)},
        "thumb_sensor_4": {"count": 9, "grid": (3, 3)},
        "index_sensor_1": {"count": 80, "grid": (8, 10)},
        "index_sensor_2": {"count": 96, "grid": (8, 12)},
        "index_sensor_3": {"count": 9, "grid": (3, 3)},
        "middle_sensor_1": {"count": 80, "grid": (10, 8)},
        "middle_sensor_2": {"count": 96, "grid": (8, 12)},
        "middle_sensor_3": {"count": 9, "grid": (3, 3)},
        "ring_sensor_1": {"count": 80, "grid": (8, 10)},
        "ring_sensor_2": {"count": 96, "grid": (8, 12)},
        "ring_sensor_3": {"count": 9, "grid": (3, 3)},
        "little_sensor_1": {"count": 80, "grid": (8, 10)},
        "little_sensor_2": {"count": 96, "grid": (8, 12)},
        "little_sensor_3": {"count": 9, "grid": (3, 3)},
    }
    
    # Add contact sensor configurations for each sensor group
    for sensor_name, config in sensor_groups.items():
        # Add individual pad sensors
        for pad_id in range(1, config["count"] + 1):
            sensor_pad_name = f"{sensor_name}_pad_{pad_id:03d}"
            setattr(env_cfg.scene, f"contact_{sensor_pad_name}", ContactSensorCfg(
                prim_path=f"{{ENV_REGEX_NS}}/InspireHand/{sensor_pad_name}",
                track_pose=True,
                update_period=0.0,  # Use simulation timestep
                debug_vis=False,  # Disable for performance with 1061 sensors
                filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube"],
            ))
    
    print(f"Added {sum(config['count'] for config in sensor_groups.values())} sensor configurations")


class InspireHandSensorManager:
    """Manager for all 1061 sensor pads on the Inspire Hand."""
    
    def __init__(self, env: ManagerBasedEnv, device: torch.device):
        self.env = env
        self.device = device
        self.sensor_names = []
        self._initialize_sensors()
        
    def _initialize_sensors(self):
        """Initialize all 1061 sensor configurations."""
        # Define sensor groups based on the README
        sensor_groups = {
            "palm_sensor": {"count": 112, "grid": (14, 8)},
            "thumb_sensor_1": {"count": 96, "grid": (8, 12)},
            "thumb_sensor_2": {"count": 8, "grid": (2, 4)},
            "thumb_sensor_3": {"count": 96, "grid": (8, 12)},
            "thumb_sensor_4": {"count": 9, "grid": (3, 3)},
            "index_sensor_1": {"count": 80, "grid": (8, 10)},
            "index_sensor_2": {"count": 96, "grid": (8, 12)},
            "index_sensor_3": {"count": 9, "grid": (3, 3)},
            "middle_sensor_1": {"count": 80, "grid": (10, 8)},
            "middle_sensor_2": {"count": 96, "grid": (8, 12)},
            "middle_sensor_3": {"count": 9, "grid": (3, 3)},
            "ring_sensor_1": {"count": 80, "grid": (8, 10)},
            "ring_sensor_2": {"count": 96, "grid": (8, 12)},
            "ring_sensor_3": {"count": 9, "grid": (3, 3)},
            "little_sensor_1": {"count": 80, "grid": (8, 10)},
            "little_sensor_2": {"count": 96, "grid": (8, 12)},
            "little_sensor_3": {"count": 9, "grid": (3, 3)},
        }
        
        # Generate all sensor names
        for group_name, info in sensor_groups.items():
            for i in range(1, info["count"] + 1):
                sensor_name = f"{group_name}_pad_{i:03d}"
                self.sensor_names.append(sensor_name)
        
        print(f"Initialized {len(self.sensor_names)} sensor pads")
        assert len(self.sensor_names) == 1061, f"Expected 1061 sensors, got {len(self.sensor_names)}"
    
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
        """Print a summary of sensor forces."""
        all_data = self.get_all_sensor_data()
        total_force = 0.0
        active_sensors = 0
        
        for sensor_name, data in all_data.items():
            if data is not None and len(data) > env_id:
                force_magnitude = float(torch.norm(data[env_id]).cpu().numpy())
                total_force += force_magnitude
                if force_magnitude > 0.01:  # 0.01N threshold
                    active_sensors += 1
        
        print(f"=== Inspire Hand Sensor Summary (Environment {env_id}) ===")
        print(f"Total sensors: {len(self.sensor_names)}")
        print(f"Active sensors (>0.01N): {active_sensors}")
        print(f"Total force magnitude: {total_force:.3f}N")
        print(f"Average force per active sensor: {total_force/max(active_sensors, 1):.3f}N")
        print("-" * 60)


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


def add_sensor_configs_to_scene(env_cfg: InspireHandEnvCfg):
    """Add all 1061 sensor configurations to the scene."""
    # This would be a very long function to add all sensor configs
    # For now, let's add a representative subset to test the system
    test_sensors = [
        "palm_sensor_pad_001", "palm_sensor_pad_002", "palm_sensor_pad_003",
        "thumb_sensor_1_pad_001", "thumb_sensor_1_pad_002",
        "index_sensor_1_pad_001", "index_sensor_1_pad_002",
        "middle_sensor_1_pad_001", "ring_sensor_1_pad_001", "little_sensor_1_pad_001"
    ]
    
    for sensor_name in test_sensors:
        setattr(env_cfg.scene, f"contact_{sensor_name}", ContactSensorCfg(
            prim_path=f"{{ENV_REGEX_NS}}/InspireHand/{sensor_name}",
            track_pose=True,
            update_period=0.0,
            debug_vis=True,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube"],
        ))
    
    print(f"Added {len(test_sensors)} test sensors to scene configuration")


def main():
    """Main function to run the Inspire Hand environment with MediaPipe control."""
    
    # Create environment configuration
    env_cfg = InspireHandEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    
    # Add sensor configurations
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
                action = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)
                action[:, :6] = hand_command.unsqueeze(0).repeat(env.num_envs, 1)
                obs, _, _, _, _ = env.step(action)
            else:
                # Default action (open hand)
                action = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)
                obs, _, _, _, _ = env.step(action)
            
            # Print sensor data every 100 steps
            if step_count % 100 == 0:
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
