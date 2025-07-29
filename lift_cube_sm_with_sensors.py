# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to run an environment with a pick and lift state machine with contact sensors.

This script extends the lift_cube_sm.py with contact sensors using the modified URDF file.
The sensors are integrated directly into the URDF and provide real-time force feedback.

.. code-block:: bash

    ./isaaclab.sh -p lift_cube_sm_with_sensors.py --num_envs 32

"""

"""Launch Omniverse Toolkit first."""

import argparse
import os
from typing import Dict, List

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Pick and lift state machine with contact sensors for lift environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
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
from collections.abc import Sequence
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches

import warp as wp

from isaaclab.assets.rigid_object.rigid_object_data import RigidObjectData
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
import isaaclab.sim as sim_utils

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from isaaclab.markers.config import FRAME_MARKER_CFG

# initialize warp
wp.init()


class GripperState:
    """States for the gripper."""

    OPEN = wp.constant(1.0)
    CLOSE = wp.constant(-1.0)


class PickSmState:
    """States for the pick state machine."""

    REST = wp.constant(0)
    APPROACH_ABOVE_OBJECT = wp.constant(1)
    APPROACH_OBJECT = wp.constant(2)
    GRASP_OBJECT = wp.constant(3)
    LIFT_OBJECT = wp.constant(4)


class PickSmWaitTime:
    """Additional wait times (in s) for states for before switching."""

    REST = wp.constant(0.2)
    APPROACH_ABOVE_OBJECT = wp.constant(0.5)
    APPROACH_OBJECT = wp.constant(0.6)
    GRASP_OBJECT = wp.constant(0.3)
    LIFT_OBJECT = wp.constant(1.0)


@wp.func
def distance_below_threshold(current_pos: wp.vec3, desired_pos: wp.vec3, threshold: float) -> bool:
    return wp.length(current_pos - desired_pos) < threshold


@wp.kernel
def infer_state_machine(
    dt: wp.array(dtype=float),
    sm_state: wp.array(dtype=int),
    sm_wait_time: wp.array(dtype=float),
    ee_pose: wp.array(dtype=wp.transform),
    object_pose: wp.array(dtype=wp.transform),
    des_object_pose: wp.array(dtype=wp.transform),
    des_ee_pose: wp.array(dtype=wp.transform),
    gripper_state: wp.array(dtype=float),
    offset: wp.array(dtype=wp.transform),
    position_threshold: float,
):
    # retrieve thread id
    tid = wp.tid()
    # retrieve state machine state
    state = sm_state[tid]
    # decide next state
    if state == PickSmState.REST:
        des_ee_pose[tid] = ee_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        # wait for a while
        if sm_wait_time[tid] >= PickSmWaitTime.REST:
            # move to next state and reset wait time
            sm_state[tid] = PickSmState.APPROACH_ABOVE_OBJECT
            sm_wait_time[tid] = 0.0
    elif state == PickSmState.APPROACH_ABOVE_OBJECT:
        des_ee_pose[tid] = wp.transform_multiply(offset[tid], object_pose[tid])
        gripper_state[tid] = GripperState.OPEN
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        ):
            # wait for a while
            if sm_wait_time[tid] >= PickSmWaitTime.APPROACH_ABOVE_OBJECT:
                # move to next state and reset wait time
                sm_state[tid] = PickSmState.APPROACH_OBJECT
                sm_wait_time[tid] = 0.0
    elif state == PickSmState.APPROACH_OBJECT:
        des_ee_pose[tid] = object_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        ):
            if sm_wait_time[tid] >= PickSmWaitTime.APPROACH_OBJECT:
                # move to next state and reset wait time
                sm_state[tid] = PickSmState.GRASP_OBJECT
                sm_wait_time[tid] = 0.0
    elif state == PickSmState.GRASP_OBJECT:
        des_ee_pose[tid] = object_pose[tid]
        gripper_state[tid] = GripperState.CLOSE
        # wait for a while
        if sm_wait_time[tid] >= PickSmWaitTime.GRASP_OBJECT:
            # move to next state and reset wait time
            sm_state[tid] = PickSmState.LIFT_OBJECT
            sm_wait_time[tid] = 0.0
    elif state == PickSmState.LIFT_OBJECT:
        des_ee_pose[tid] = des_object_pose[tid]
        gripper_state[tid] = GripperState.CLOSE
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        ):
            # wait for a while
            if sm_wait_time[tid] >= PickSmWaitTime.LIFT_OBJECT:
                # move to next state and reset wait time
                sm_state[tid] = PickSmState.LIFT_OBJECT
                sm_wait_time[tid] = 0.0
    # increment wait time
    sm_wait_time[tid] = sm_wait_time[tid] + dt[tid]


class PickAndLiftSm:
    """A simple state machine in a robot's task space to pick and lift an object."""

    def __init__(self, dt: float, num_envs: int, device: torch.device, position_threshold: float = 0.01):
        """Initialize the state machine.

        Args:
            dt: The environment time step.
            num_envs: The number of environments to simulate.
            device: The device to run the state machine on.
            position_threshold: The position threshold for the state machine.
        """
        # save parameters
        self.dt = float(dt)
        self.num_envs = num_envs
        self.device = device
        self.position_threshold = position_threshold

        # initialize state machine
        self.sm_dt = torch.full((self.num_envs,), self.dt, device=self.device)
        self.sm_state = torch.full((self.num_envs,), 0, dtype=torch.int32, device=self.device)
        self.sm_wait_time = torch.zeros((self.num_envs,), device=self.device)

        # desired state
        self.des_ee_pose = torch.zeros((self.num_envs, 7), device=self.device)
        self.des_gripper_state = torch.full((self.num_envs,), 0.0, device=self.device)

        # approach above object offset
        self.offset = torch.zeros((self.num_envs, 7), device=self.device)
        self.offset[:, 2] = 0.1
        self.offset[:, -1] = 1.0  # warp expects quaternion as (x, y, z, w)

        # convert to warp
        self.sm_dt_wp = wp.from_torch(self.sm_dt, wp.float32)
        self.sm_state_wp = wp.from_torch(self.sm_state, wp.int32)
        self.sm_wait_time_wp = wp.from_torch(self.sm_wait_time, wp.float32)
        self.des_ee_pose_wp = wp.from_torch(self.des_ee_pose, wp.transform)
        self.des_gripper_state_wp = wp.from_torch(self.des_gripper_state, wp.float32)
        self.offset_wp = wp.from_torch(self.offset, wp.transform)

    def reset_idx(self, env_ids: Sequence[int]):
        """Reset the state machine."""
        self.sm_state[env_ids] = 0
        self.sm_wait_time[env_ids] = 0.0

    def compute(self, ee_pose: torch.Tensor, object_pose: torch.Tensor, des_object_pose: torch.Tensor):
        """Compute the desired state of the robot's end-effector and the gripper."""
        # convert transformations from (w, x, y, z) to (x, y, z, w)
        ee_pose = ee_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        object_pose = object_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        des_object_pose = des_object_pose[:, [0, 1, 2, 4, 5, 6, 3]]

        # convert to warp (ensuring correct data type)
        ee_pose_wp = wp.from_torch(ee_pose.contiguous(), dtype=wp.transform)
        object_pose_wp = wp.from_torch(object_pose.contiguous(), dtype=wp.transform)
        des_object_pose_wp = wp.from_torch(des_object_pose.contiguous(), dtype=wp.transform)

        # run state machine
        wp.launch(
            kernel=infer_state_machine,
            dim=self.num_envs,
            inputs=[
                self.sm_dt_wp,
                self.sm_state_wp,
                self.sm_wait_time_wp,
                ee_pose_wp,
                object_pose_wp,
                des_object_pose_wp,
                self.des_ee_pose_wp,
                self.des_gripper_state_wp,
                self.offset_wp,
                self.position_threshold,
            ],
            device=self.device,
        )

        # convert transformations back to (w, x, y, z)
        des_ee_pose = self.des_ee_pose[:, [0, 1, 2, 6, 3, 4, 5]]
        # convert to torch
        return torch.cat([des_ee_pose, self.des_gripper_state.unsqueeze(-1)], dim=-1)


class ContactSensorManager:
    """Manager for contact sensors on the gripper."""

    def __init__(self, num_envs: int, device: torch.device):
        self.num_envs = num_envs
        self.device = device
        self.sensors: Dict[str, ContactSensor] = {}

    def add_sensor(self, sensor_name: str, sensor: ContactSensor):
        """Add a contact sensor to the manager."""
        self.sensors[sensor_name] = sensor

    def get_sensor_data(self, sensor_name: str) -> torch.Tensor:
        """Get contact force data from a specific sensor."""
        if sensor_name in self.sensors:
            return self.sensors[sensor_name].data.net_forces_w
        return torch.zeros((self.num_envs, 3), device=self.device)

    def get_all_sensor_data(self) -> Dict[str, torch.Tensor]:
        """Get contact force data from all sensors."""
        return {name: sensor.data.net_forces_w for name, sensor in self.sensors.items()}

    def print_sensor_data(self, env_id: int = 0):
        """Print contact force data for debugging."""
        print(f"\n=== Contact Sensor Data (Environment {env_id}) - Step-wise Average ===")
        total_force = 0.0
        left_total = 0.0
        right_total = 0.0
        
        print(f" Detailed Force Analysis (Object mass: 1.0kg, Expected gravity: ~9.8N):")
        print("-" * 80)
        
        for name, sensor in self.sensors.items():
            forces = sensor.data.net_forces_w[env_id]
            force_magnitude = torch.norm(forces).item()
            
            # 正确提取力分量 - 参考精确代码的实现
            if forces.numel() >= 3:  # 确保有足够的元素
                forces_np = forces.cpu().numpy().flatten()
                fx, fy, fz = float(forces_np[0]), float(forces_np[1]), float(forces_np[2])
            else:  # 如果数据不完整，使用0填充
                fx, fy, fz = 0.0, 0.0, 0.0
            
            total_force += force_magnitude
            
            # Separate statistics for left and right fingers
            if "leftfinger" in name:
                left_total += force_magnitude
            elif "rightfinger" in name:
                right_total += force_magnitude
            
            # Force anomaly detection
            warning = ""
            if force_magnitude > 10.0:  # High force warning
                warning = "  HIGH FORCE!"
            elif force_magnitude > 20.0:  # Very high force
                warning = "  EXCESSIVE FORCE!"
                
            print(f"{name}:")
            print(f"  Force components: fx={fx:7.3f}N, fy={fy:7.3f}N, fz={fz:7.3f}N")
            print(f"  Total magnitude:  {force_magnitude:7.3f}N{warning}")
            print(f"  Dominant axis:    {'X' if abs(fx) > abs(fy) and abs(fx) > abs(fz) else 'Y' if abs(fy) > abs(fz) else 'Z'}")
            print()
        
        print(f" Statistics Summary:")
        print(f"   Left finger total force: {left_total:.3f}N")
        print(f"   Right finger total force: {right_total:.3f}N") 
        print(f"   Gripper total force: {total_force:.3f}N")
        
        # Anomaly analysis
        if total_force > 40.0:  # Adjusted threshold for 1kg object
            print(f" FORCE ANOMALY DETECTED!")
            print(f"   Current total: {total_force:.1f}N")
            print(f"   Expected for 1kg object: ~15-30N (including safety factor)")
            print(f"   Ratio: {total_force/9.8:.1f}x theoretical gravity")
            print(f" Possible causes:")
            print(f"   - Excessive gripper stiffness")
            print(f"   - Contact material properties too rigid")
            print(f"   - Sensor configuration issues")
            print(f"   - Simulation step size problems")
        
        print(f"   Data type: Step-wise average over control period")
        print("-" * 60)


class SensorForceVisualizer:
    """Real-time visualization of sensor force data using matplotlib."""
    
    def __init__(self, sensor_manager: ContactSensorManager):
        self.sensor_manager = sensor_manager
        self.has_display = True
        
        # Set matplotlib backend to ensure proper display
        import matplotlib
        
        print(" 配置matplotlib后端...")
        
        # Set DISPLAY if not set properly
        import os
        if not os.environ.get('DISPLAY'):
            os.environ['DISPLAY'] = ':1'
            print("设置DISPLAY=:1")
        
        # Only use Qt5Agg backend (most stable for Isaac Sim)
        try:
            import matplotlib.pyplot as plt
            plt.switch_backend('Qt5Agg')
            matplotlib.use('Qt5Agg', force=True)
            
            # Test if the backend works
            test_fig = plt.figure()
            plt.close(test_fig)
            
            print(" 成功设置Qt5Agg后端")
            backend_set = True
            
        except ImportError as e:
            print(f" Qt5Agg后端缺少依赖: {e}")
            print(" 请在Isaac Sim Python环境中安装PyQt5:")
            print("   ./isaaclab.sh -p -m pip install PyQt5")
            backend_set = False
            
        except Exception as e:
            print(f" Qt5Agg后端失败: {e}")
            backend_set = False
        
        if not backend_set:
            print(" 切换到文本输出模式...")
            self.has_display = False
        
        if self.has_display:
            try:
                # Setup matplotlib for interactive plotting
                plt.ion()  # Turn on interactive mode
                self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(16, 8))
                self.fig.suptitle('Gripper Sensor Force Visualization (Step-wise Average)', 
                                 fontsize=16, fontweight='bold')
                
                # Configure subplots with clearer titles
                self.ax1.set_title('Left Finger Sensors - Average Force\n(Inner side faces object)', fontsize=14, fontweight='bold', pad=20)
                self.ax2.set_title('Right Finger Sensors - Average Force\n(Inner side faces object)', fontsize=14, fontweight='bold', pad=20)
                
                # Set up 2x2 grids for each subplot with better labeling
                for ax in [self.ax1, self.ax2]:
                    ax.set_xlim(-0.1, 2.1)
                    ax.set_ylim(-0.1, 2.1)
                    ax.set_aspect('equal')
                    # Remove default ticks, use custom labels
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.grid(True, alpha=0.3)
                
                # Define sensor layout - arranged by physical position
                # Assuming sensors layout on fingers:
                # Sensors 1 and 2 on upper part, sensors 3 and 4 on lower part
                # Sensors 1 and 3 on outer side, sensors 2 and 4 on inner side
                sensor_layout = {
                    'left': [
                        {'pos': (0, 1), 'sensor_id': 1, 'name': 'Sensor 1', 'desc': '(Upper-Outer)'},
                        {'pos': (1, 1), 'sensor_id': 2, 'name': 'Sensor 2', 'desc': '(Upper-Inner)'},
                        {'pos': (0, 0), 'sensor_id': 3, 'name': 'Sensor 3', 'desc': '(Lower-Outer)'},
                        {'pos': (1, 0), 'sensor_id': 4, 'name': 'Sensor 4', 'desc': '(Lower-Inner)'},
                    ],
                    'right': [
                        {'pos': (0, 1), 'sensor_id': 1, 'name': 'Sensor 1', 'desc': '(Upper-Inner)'},
                        {'pos': (1, 1), 'sensor_id': 2, 'name': 'Sensor 2', 'desc': '(Upper-Outer)'},
                        {'pos': (0, 0), 'sensor_id': 3, 'name': 'Sensor 3', 'desc': '(Lower-Inner)'},
                        {'pos': (1, 0), 'sensor_id': 4, 'name': 'Sensor 4', 'desc': '(Lower-Outer)'},
                    ]
                }
                
                # Initialize rectangles for visualization
                self.left_rects = []
                self.right_rects = []
                self.left_texts = []
                self.right_texts = []
                self.left_sensor_ids = []
                self.right_sensor_ids = []
                self.left_sensor_labels = []
                self.right_sensor_labels = []
                
                # Create rectangles and text for left finger
                for sensor_info in sensor_layout['left']:
                    x, y = sensor_info['pos']
                    sensor_id = sensor_info['sensor_id']
                    name = sensor_info['name']
                    desc = sensor_info['desc']
                    
                    rect = patches.Rectangle((x, y), 1, 1, linewidth=3, edgecolor='darkblue', 
                                           facecolor='lightblue', alpha=0.8)
                    self.ax1.add_patch(rect)
                    self.left_rects.append(rect)
                    self.left_sensor_ids.append(sensor_id)
                    
                    # Add sensor name label inside the square (upper part)
                    label = self.ax1.text(x + 0.5, y + 0.8, name, 
                                        ha='center', va='center', fontsize=10, 
                                        fontweight='bold', color='darkblue')
                    self.left_sensor_labels.append(label)
                    
                    # Add position description outside the square (below)
                    pos_label = self.ax1.text(x + 0.5, y - 0.15, desc, 
                                            ha='center', va='center', fontsize=8, 
                                            fontweight='normal', color='gray')
                    
                    # Add force value display inside the square (lower part)
                    text = self.ax1.text(x + 0.5, y + 0.3, '0.000 N', ha='center', va='center',
                                       fontsize=11, fontweight='bold', color='black')
                    self.left_texts.append(text)
                
                # Create rectangles and text for right finger  
                for sensor_info in sensor_layout['right']:
                    x, y = sensor_info['pos']
                    sensor_id = sensor_info['sensor_id']
                    name = sensor_info['name']
                    desc = sensor_info['desc']
                    
                    rect = patches.Rectangle((x, y), 1, 1, linewidth=3, edgecolor='darkred',
                                           facecolor='lightcoral', alpha=0.8)
                    self.ax2.add_patch(rect)
                    self.right_rects.append(rect)
                    self.right_sensor_ids.append(sensor_id)
                    
                    # Add sensor name label inside the square (upper part)
                    label = self.ax2.text(x + 0.5, y + 0.8, name, 
                                        ha='center', va='center', fontsize=10, 
                                        fontweight='bold', color='darkred')
                    self.right_sensor_labels.append(label)
                    
                    # Add position description outside the square (below)
                    pos_label = self.ax2.text(x + 0.5, y - 0.15, desc, 
                                            ha='center', va='center', fontsize=8, 
                                            fontweight='normal', color='gray')
                    
                    # Add force value display inside the square (lower part)
                    text = self.ax2.text(x + 0.5, y + 0.3, '0.000 N', ha='center', va='center',
                                       fontsize=11, fontweight='bold', color='black')
                    self.right_texts.append(text)
                
                # Add coordinate axis explanations
                self.ax1.text(1, -0.3, 'Inner <- -> Outer', ha='center', va='center', 
                             fontsize=10, fontweight='bold', color='blue')
                self.ax1.text(-0.3, 1, 'Up\n^\nv\nDown', ha='center', va='center', 
                             fontsize=10, fontweight='bold', color='blue')
                
                self.ax2.text(1, -0.3, 'Inner <- -> Outer', ha='center', va='center', 
                             fontsize=10, fontweight='bold', color='red')
                self.ax2.text(-0.3, 1, 'Up\n^\nv\nDown', ha='center', va='center', 
                             fontsize=10, fontweight='bold', color='red')
                
                # Color map for force visualization
                self.max_force = 25.0  # Maximum force for color scaling
                
                plt.tight_layout()
                
                # Force the window to show and bring to front
                self.fig.show()
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
                
                print(" Sensor visualization window created!")
                print(" Sensor layout description:")
                print("   Left finger: Blue border | Right finger: Red border")
                print("   Upper-Outer/Upper-Inner/Lower-Outer/Lower-Inner show sensor positions on finger")
                print("   Inner: Side facing the object")
                print("   Outer: Side facing outward")
                print(" Tip: If you can't see the window, try using Alt+Tab to switch windows")
                
            except Exception as e:
                print(f" 无法创建图形窗口: {e}")
                print(" 切换到文本输出模式...")
                self.has_display = False
        
        if not self.has_display:
            print("使用文本输出模式显示传感器数据")
            print("=" * 80)
    
    def update_visualization(self, env_id: int = 0):
        """Update the force visualization for a specific environment."""
        try:
            # Get all sensor data
            all_sensor_data = self.sensor_manager.get_all_sensor_data()
            
            # Extract left and right finger force components with proper mapping
            left_force_components = []
            right_force_components = []
            left_magnitudes = []
            right_magnitudes = []
            
            # 按照传感器ID顺序获取力分量
            for sensor_id in self.left_sensor_ids if self.has_display else range(1, 5):
                left_sensor_name = f"panda_leftfinger_sensor_{sensor_id}"
                if left_sensor_name in all_sensor_data:
                    force_vector = all_sensor_data[left_sensor_name][env_id].cpu().numpy().flatten()
                    if len(force_vector) >= 3:
                        fx, fy, fz = float(force_vector[0]), float(force_vector[1]), float(force_vector[2])
                        magnitude = float(torch.norm(all_sensor_data[left_sensor_name][env_id]).cpu().numpy())
                    else:
                        fx, fy, fz = 0.0, 0.0, 0.0
                        magnitude = 0.0
                    left_force_components.append([fx, fy, fz])
                    left_magnitudes.append(magnitude)
                else:
                    left_force_components.append([0.0, 0.0, 0.0])
                    left_magnitudes.append(0.0)
                    
            for sensor_id in self.right_sensor_ids if self.has_display else range(1, 5):
                right_sensor_name = f"panda_rightfinger_sensor_{sensor_id}"
                if right_sensor_name in all_sensor_data:
                    force_vector = all_sensor_data[right_sensor_name][env_id].cpu().numpy().flatten()
                    if len(force_vector) >= 3:
                        fx, fy, fz = float(force_vector[0]), float(force_vector[1]), float(force_vector[2])
                        magnitude = float(torch.norm(all_sensor_data[right_sensor_name][env_id]).cpu().numpy())
                    else:
                        fx, fy, fz = 0.0, 0.0, 0.0
                        magnitude = 0.0
                    right_force_components.append([fx, fy, fz])
                    right_magnitudes.append(magnitude)
                else:
                    right_force_components.append([0.0, 0.0, 0.0])
                    right_magnitudes.append(0.0)
            
            if self.has_display:
                # Update graphical visualization
                for i, (rect, text, force_comp, magnitude, sensor_id) in enumerate(zip(self.left_rects, self.left_texts, left_force_components, left_magnitudes, self.left_sensor_ids)):
                    normalized_force = min(magnitude / self.max_force, 1.0)
                    if magnitude > 0.1:
                        color = plt.cm.Blues(0.4 + normalized_force * 0.6)
                    else:
                        color = 'lightblue'
                    rect.set_facecolor(color)
                    fx, fy, fz = force_comp
                    text.set_text(f'[{fx:.1f},{fy:.1f},{fz:.1f}]')
                    text.set_color('white' if normalized_force > 0.6 else 'darkblue')
                
                for i, (rect, text, force_comp, magnitude, sensor_id) in enumerate(zip(self.right_rects, self.right_texts, right_force_components, right_magnitudes, self.right_sensor_ids)):
                    normalized_force = min(magnitude / self.max_force, 1.0)
                    if magnitude > 0.1:
                        color = plt.cm.Reds(0.4 + normalized_force * 0.6)
                    else:
                        color = 'lightcoral'
                    rect.set_facecolor(color)
                    fx, fy, fz = force_comp
                    text.set_text(f'[{fx:.1f},{fy:.1f},{fz:.1f}]')
                    text.set_color('white' if normalized_force > 0.6 else 'darkred')
                
                max_left = max(left_magnitudes) if left_magnitudes else 0.0
                max_right = max(right_magnitudes) if right_magnitudes else 0.0
                total_force = sum(left_magnitudes) + sum(right_magnitudes)
                
                self.fig.suptitle(f'Gripper Sensor Force Components [fx,fy,fz] | Left Max: {max_left:.2f}N | Right Max: {max_right:.2f}N | Total: {total_force:.2f}N', 
                                 fontsize=14, fontweight='bold')
                
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
            else:
                # Text-based visualization with force components
                max_left = max(left_magnitudes) if left_magnitudes else 0.0
                max_right = max(right_magnitudes) if right_magnitudes else 0.0
                total_force = sum(left_magnitudes) + sum(right_magnitudes)
                
                print("\n" + "="*80)
                print(f"Gripper Sensor Force Components [fx,fy,fz] | Total Force: {total_force:.2f}N")
                print("="*80)
                print("Left Finger Sensors - Force Components [fx,fy,fz]:")
                for i, (force_comp, magnitude) in enumerate(zip(left_force_components, left_magnitudes)):
                    fx, fy, fz = force_comp
                    print(f"  Sensor {i+1}: [{fx:6.2f},{fy:6.2f},{fz:6.2f}] | Mag: {magnitude:6.2f}N")
                print(f"  Max: {max_left:.2f}N")
                print()
                print("Right Finger Sensors - Force Components [fx,fy,fz]:")
                for i, (force_comp, magnitude) in enumerate(zip(right_force_components, right_magnitudes)):
                    fx, fy, fz = force_comp
                    print(f"  Sensor {i+1}: [{fx:6.2f},{fy:6.2f},{fz:6.2f}] | Mag: {magnitude:6.2f}N")
                print(f"  Max: {max_right:.2f}N")
                print()
                print(" Format: [fx,fy,fz] in Newtons | Mag = Force magnitude")
                print("Data type: Step-wise average (non-instantaneous peak)")
                
        except Exception as e:
            print(f" 可视化更新错误: {e}")
    
    def close(self):
        """Close the matplotlib windows."""
        if self.has_display:
            plt.close(self.fig)


def create_custom_lift_env_cfg(usd_path: str, env_cfg: LiftEnvCfg) -> LiftEnvCfg:
    """Create a custom lift environment configuration with the modified USD file."""
    
    # 修改机器人配置使用我们的USD文件
    robot_cfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos={
                "panda_joint1": 0.0,
                "panda_joint2": -0.569,
                "panda_joint3": 0.0,
                "panda_joint4": -2.810,
                "panda_joint5": 0.0,
                "panda_joint6": 3.037,
                "panda_joint7": 0.741,
                "panda_finger_joint1": 0.035,
                "panda_finger_joint2": 0.035,
            },
            joint_vel={".*": 0.0},
        ),
        actuators={
            "panda_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                effort_limit=87.0,
                velocity_limit=2.175,
                stiffness=80.0,
                damping=4.0,
            ),
            "panda_forearm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                effort_limit=12.0,
                velocity_limit=2.61,
                stiffness=80.0,
                damping=4.0,
            ),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint.*"],
                effort_limit=20.0,    # 恢复到20N
                velocity_limit=0.2,
                stiffness=200.0,      # 恢复到200
                damping=20.0,         # 恢复到20
            ),
        },
    )
    
    # 替换原有的机器人配置
    env_cfg.scene.robot = robot_cfg
    
    # 添加end effector frame配置
    marker_cfg = FRAME_MARKER_CFG.copy()
    marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    marker_cfg.prim_path = "/Visuals/FrameTransformer"
    env_cfg.scene.ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
        debug_vis=False,
        visualizer_cfg=marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                name="end_effector",
                offset=OffsetCfg(
                    pos=[0.0, 0.0, 0.1034],
                ),
            ),
        ],
    )
    
    # 添加传感器配置
    sensor_names = [
        "panda_leftfinger_sensor_1", "panda_leftfinger_sensor_2", 
        "panda_leftfinger_sensor_3", "panda_leftfinger_sensor_4",
        "panda_rightfinger_sensor_1", "panda_rightfinger_sensor_2",
        "panda_rightfinger_sensor_3", "panda_rightfinger_sensor_4"
    ]
    
    for sensor_name in sensor_names:
        setattr(env_cfg.scene, f"contact_{sensor_name}", ContactSensorCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Robot/{sensor_name}",
            track_pose=True,
            update_period=0.0,
            debug_vis=True,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        ))
    
    return env_cfg


def main():
    # 获取USD文件路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    usd_path = os.path.join(current_dir, "panda_arm_hand_with_sensors_final.usd")
    
    if not os.path.exists(usd_path):
        print(f"Error: USD file not found at {usd_path}")
        return
    
    print(f"Using USD file: {usd_path}")
    
    # parse configuration (same as reference file)
    env_cfg: LiftEnvCfg = parse_env_cfg(
        "Isaac-Lift-Cube-Franka-IK-Abs-v0",
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    
    # Only modify the robot's USD path, keep other configurations unchanged
    env_cfg.scene.robot.spawn.usd_path = usd_path
    env_cfg.scene.robot.spawn.activate_contact_sensors = True
    
    # Override object mass properties - set cube mass to 1.0 kg for testing
    from isaaclab.sim import MassPropertiesCfg
    env_cfg.scene.object.spawn.mass_props = MassPropertiesCfg(mass=1.0)  # 1 kg instead of original 0.216 kg
    print(f"Object mass overridden: 1.0 kg (original was 0.216 kg)")
    print(f"Expected grip force: ~10N (gravity) + ~5-15N (safety factor) = ~15-25N total")
    
    # Add sensor configurations
    sensor_names = [
        "panda_leftfinger_sensor_1", "panda_leftfinger_sensor_2", 
        "panda_leftfinger_sensor_3", "panda_leftfinger_sensor_4",
        "panda_rightfinger_sensor_1", "panda_rightfinger_sensor_2",
        "panda_rightfinger_sensor_3", "panda_rightfinger_sensor_4"
    ]
    
    for sensor_name in sensor_names:
        setattr(env_cfg.scene, f"contact_{sensor_name}", ContactSensorCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Robot/{sensor_name}",
            track_pose=True,
            update_period=env_cfg.sim.dt * env_cfg.decimation,  # Use control step as mean window
            debug_vis=True,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        ))
    
    # create environment
    env = gym.make("Isaac-Lift-Cube-Franka-IK-Abs-v0", cfg=env_cfg)
    # reset environment at start
    env.reset()
    
    # Print simulation and sensor configuration information
    control_dt = env_cfg.sim.dt * env_cfg.decimation
    print(f"Simulation Configuration:")
    print(f"   Simulation step (sim.dt): {env_cfg.sim.dt:.4f}s")
    print(f"   Control frequency (decimation): {env_cfg.decimation}")
    print(f"   Control period: {control_dt:.4f}s ({1/control_dt:.1f} Hz)")
    print(f"   Sensor mean window: {control_dt:.4f}s")
    print(f"   Sensor type: Step-wise average (non-instantaneous values)")
    
    # Create sensor manager
    sensor_manager = ContactSensorManager(env.unwrapped.num_envs, env.unwrapped.device)
    
    # Add all sensors to manager
    for sensor_name, sensor_obj in env.unwrapped.scene.sensors.items():
        if "contact_panda" in sensor_name and "sensor" in sensor_name:
            clean_name = sensor_name.replace("contact_", "")
            sensor_manager.add_sensor(clean_name, sensor_obj)
    
    print(f"Added {len(sensor_manager.sensors)} sensors")

    # create action buffers (same as reference file)
    actions = torch.zeros(env.unwrapped.action_space.shape, device=env.unwrapped.device)
    actions[:, 3] = 1.0
    # desired object orientation (we only do position control of object)
    desired_orientation = torch.zeros((env.unwrapped.num_envs, 4), device=env.unwrapped.device)
    desired_orientation[:, 1] = 1.0
    # create state machine
    pick_sm = PickAndLiftSm(
        env_cfg.sim.dt * env_cfg.decimation, env.unwrapped.num_envs, env.unwrapped.device, position_threshold=0.01
    )

    # Sensor data print counter
    step_counter = 0
    print_interval = 100  # Print sensor data every 100 steps

    # Create sensor visualizer
    sensor_visualizer = SensorForceVisualizer(sensor_manager)
    visualization_interval = 10  # Update visualization every 10 steps
    
    print("Finding sensor visualization window:")
    print("1. Check if a new matplotlib window appears in the taskbar")
    print("2. Use Alt+Tab to switch windows to find it")
    print("3. The window title should be 'Figure 1' or contain 'Gripper Sensor Force Visualization'")
    print("4. If you can't see it, try unblocking pop-up windows")
    print(" Simulating in 3 seconds...")
    
    import time
    time.sleep(3)  # Give user time to find the window

    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # step environment
            dones = env.step(actions)[-2]

            # observations (same as reference file)
            # -- end-effector frame
            ee_frame_sensor = env.unwrapped.scene["ee_frame"]
            tcp_rest_position = ee_frame_sensor.data.target_pos_w[..., 0, :].clone() - env.unwrapped.scene.env_origins
            tcp_rest_orientation = ee_frame_sensor.data.target_quat_w[..., 0, :].clone()
            # -- object frame
            object_data: RigidObjectData = env.unwrapped.scene["object"].data
            object_position = object_data.root_pos_w - env.unwrapped.scene.env_origins
            # -- target object frame
            desired_position = env.unwrapped.command_manager.get_command("object_pose")[..., :3]

            # advance state machine
            actions = pick_sm.compute(
                torch.cat([tcp_rest_position, tcp_rest_orientation], dim=-1),
                torch.cat([object_position, desired_orientation], dim=-1),
                torch.cat([desired_position, desired_orientation], dim=-1),
            )

            # reset state machine
            if dones.any():
                pick_sm.reset_idx(dones.nonzero(as_tuple=False).squeeze(-1))

            # Print sensor data (periodically)
            step_counter += 1
            if step_counter % print_interval == 0:
                sensor_manager.print_sensor_data(env_id=0)

            # Update sensor visualization
            if step_counter % visualization_interval == 0:
                sensor_visualizer.update_visualization(env_id=0)

    # close the environment
    env.close()
    # Close sensor visualizer
    sensor_visualizer.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close() 
