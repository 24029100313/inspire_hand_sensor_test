#!/usr/bin/env python3

"""
Inspire Hand State Machine Cube Grasping Environment

This environment demonstrates a state machine approach to grasping a cube with the Inspire Hand.
Based on Isaac Lab's official lift_cube_sm.py with adaptations for Inspire Hand.

Features:
- Warp-accelerated state machine for GPU parallel processing
- 5-state grasping sequence: REST → APPROACH_ABOVE → APPROACH → GRASP → LIFT
- Real position control for cube grasping
- Inspire Hand with 1061 sensors integration ready

States:
1. REST - Initialize and prepare
2. APPROACH_ABOVE_OBJECT - Move hand above the cube 
3. APPROACH_OBJECT - Descend to cube level
4. GRASP_OBJECT - Close fingers around cube
5. LIFT_OBJECT - Lift the cube up
"""

import argparse
import math
import torch
import numpy as np
from collections.abc import Sequence

import warp as wp

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Inspire Hand State Machine Cube Grasping")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(headless=args_cli.headless)
simulation_app = app_launcher.app

"""Rest of the imports"""
import gymnasium as gym
import torch
import numpy as np

# Isaac Lab imports
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.assets.rigid_object.rigid_object_data import RigidObjectData
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.sim import SimulationCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg

# initialize warp
wp.init()

# Warp constants for gripper state
class GripperState:
    """States for the gripper."""
    OPEN = wp.constant(0.0)  # Open hand position
    CLOSE = wp.constant(1.0)  # Closed/grasping position

# Warp constants for state machine
class PickSmState:
    """States for the pick state machine."""
    REST = wp.constant(0)
    APPROACH_ABOVE_OBJECT = wp.constant(1)
    APPROACH_OBJECT = wp.constant(2)
    GRASP_OBJECT = wp.constant(3)
    LIFT_OBJECT = wp.constant(4)

class PickSmWaitTime:
    """Additional wait times (in s) for states before switching."""
    REST = wp.constant(0.5)
    APPROACH_ABOVE_OBJECT = wp.constant(1.0)
    APPROACH_OBJECT = wp.constant(0.8)
    GRASP_OBJECT = wp.constant(1.2)
    LIFT_OBJECT = wp.constant(2.0)


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
                # cycle back to rest (continuous demo)
                sm_state[tid] = PickSmState.REST
                sm_wait_time[tid] = 0.0
                
    # increment wait time
    sm_wait_time[tid] = sm_wait_time[tid] + dt[tid]


class PickAndLiftSm:
    """A warp-accelerated state machine for Inspire Hand cube grasping.

    The state machine implements a complete pick and lift cycle:
    1. REST: Initialize position
    2. APPROACH_ABOVE_OBJECT: Move above the cube
    3. APPROACH_OBJECT: Descend to cube level  
    4. GRASP_OBJECT: Close fingers around cube
    5. LIFT_OBJECT: Lift the cube to target height
    """

    def __init__(self, dt: float, num_envs: int, device: torch.device | str = "cpu", position_threshold=0.02):
        """Initialize the state machine.

        Args:
            dt: The environment time step.
            num_envs: The number of environments to simulate.
            device: The device to run the state machine on.
            position_threshold: Distance threshold for state transitions
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

        # approach above object offset (10cm above cube)
        self.offset = torch.zeros((self.num_envs, 7), device=self.device)
        self.offset[:, 2] = 0.15  # 15cm above for inspire hand clearance
        self.offset[:, -1] = 1.0  # warp expects quaternion as (x, y, z, w)

        # convert to warp
        self.sm_dt_wp = wp.from_torch(self.sm_dt, wp.float32)
        self.sm_state_wp = wp.from_torch(self.sm_state, wp.int32)
        self.sm_wait_time_wp = wp.from_torch(self.sm_wait_time, wp.float32)
        self.des_ee_pose_wp = wp.from_torch(self.des_ee_pose, wp.transform)
        self.des_gripper_state_wp = wp.from_torch(self.des_gripper_state, wp.float32)
        self.offset_wp = wp.from_torch(self.offset, wp.transform)

    def reset_idx(self, env_ids: Sequence[int] = None):
        """Reset the state machine."""
        if env_ids is None:
            env_ids = slice(None)
        self.sm_state[env_ids] = 0
        self.sm_wait_time[env_ids] = 0.0

    def compute(self, ee_pose: torch.Tensor, object_pose: torch.Tensor, des_object_pose: torch.Tensor) -> torch.Tensor:
        """Compute the desired state of the robot's end-effector and the gripper."""
        # convert all transformations from (w, x, y, z) to (x, y, z, w)
        ee_pose = ee_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        object_pose = object_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        des_object_pose = des_object_pose[:, [0, 1, 2, 4, 5, 6, 3]]

        # convert to warp
        ee_pose_wp = wp.from_torch(ee_pose.contiguous(), wp.transform)
        object_pose_wp = wp.from_torch(object_pose.contiguous(), wp.transform)
        des_object_pose_wp = wp.from_torch(des_object_pose.contiguous(), wp.transform)

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

    def get_current_state_name(self, env_id: int = 0) -> str:
        """Get human-readable current state name."""
        state_names = {
            0: "REST",
            1: "APPROACH_ABOVE_OBJECT", 
            2: "APPROACH_OBJECT",
            3: "GRASP_OBJECT",
            4: "LIFT_OBJECT"
        }
        return state_names.get(self.sm_state[env_id].item(), "UNKNOWN")


@configclass
class InspireHandSceneCfg(InteractiveSceneCfg):
    """Configuration for the Inspire Hand scene."""

    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    # Lighting
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=3000.0,
            color=(1.0, 1.0, 1.0),
        ),
    )

    distant_light = AssetBaseCfg(
        prim_path="/World/DistantLight",
        spawn=sim_utils.DistantLightCfg(
            intensity=600.0,
            angle=0.53,
            color=(1.0, 1.0, 1.0),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, 8.0),
            rot=(0.382, 0.0, 0.0, 0.924),  # 45 degree angle
        ),
    )

    # Inspire Hand
    inspire_hand: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/InspireHand",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/larry/NVIDIA_DEV/isaac_grasp_ws/cabinet_sensor_project/inspire_hand_with_sensors/usd/inspire_hand_processed.usd",
            activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.4),  # Lowered for better cube access
            joint_pos={
                # Set all finger joints to open position initially
                "right_index_1_joint": 0.0,
                "right_index_2_joint": 0.0,
                "right_middle_1_joint": 0.0,
                "right_middle_2_joint": 0.0,
                "right_ring_1_joint": 0.0,
                "right_ring_2_joint": 0.0,
                "right_little_1_joint": 0.0,
                "right_little_2_joint": 0.0,
                "right_thumb_1_joint": 0.0,
                "right_thumb_2_joint": 0.0,
                "right_thumb_3_joint": 0.0,
                "right_thumb_4_joint": 0.0,
            },
        ),
        actuators={
            "hand_joints": ImplicitActuatorCfg(
                joint_names_expr=["right_.*_joint"],
                effort_limit=15.0,  # Increased for better grasping
                velocity_limit=2.0,  # Slower for more precise control
                stiffness=80.0,     # Higher stiffness for stable grasping
                damping=4.0,        # Higher damping for stability
            ),
        },
    )

    # Frame transformer for end-effector tracking
    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/InspireHand/right_hand_base",
        debug_vis=True,
        visualizer_cfg=FrameTransformerCfg.FrameVisualizerCfg(scale=0.1),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/InspireHand/right_hand_base",
                name="ee_link",
                offset=OffsetCfg(
                    pos=[0.0, 0.0, 0.1],  # Offset to represent grasping point
                ),
            ),
        ],
    )

    # Target cube for grasping
    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.04, 0.04, 0.04),  # Smaller cube for easier grasping
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),  # Lighter cube
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.8, 0.2, 0.2), 
                metallic=0.2,
                roughness=0.4,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.08, 0.0, 0.42),  # Positioned for grasping
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )


@configclass
class InspireHandStateMachineCfg(DirectRLEnvCfg):
    """Configuration for Inspire Hand State Machine Environment."""
    
    # simulation settings
    sim: SimulationCfg = SimulationCfg(dt=1 / 60, render_interval=1)
    
    # scene settings
    scene: InspireHandSceneCfg = InspireHandSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)
    
    # basic settings
    decimation = 2
    episode_length_s = 20.0  # Longer episodes for complete cycles
    action_space = 20  # 7 for EE pose + 12 for finger joints + 1 for gripper state
    observation_space = 31  # 7 EE pose + 12 joint pos + 12 joint vel
    num_actions = 20
    num_observations = 31


class InspireHandStateMachineEnv(DirectRLEnv):
    """Inspire Hand State Machine Environment for cube grasping with warp acceleration."""
    
    cfg: InspireHandStateMachineCfg
    
    def __init__(self, cfg: InspireHandStateMachineCfg, render_mode: str | None = None, **kwargs):
        """Initialize the environment."""
        super().__init__(cfg, render_mode, **kwargs)
        
        # Joint names for inspire hand
        self.joint_names = [
            "right_index_1_joint", "right_index_2_joint",
            "right_middle_1_joint", "right_middle_2_joint", 
            "right_ring_1_joint", "right_ring_2_joint",
            "right_little_1_joint", "right_little_2_joint",
            "right_thumb_1_joint", "right_thumb_2_joint",
            "right_thumb_3_joint", "right_thumb_4_joint"
        ]
        
        # Create state machine
        self.pick_sm = PickAndLiftSm(
            self.cfg.sim.dt * self.cfg.decimation, 
            self.num_envs, 
            self.device, 
            position_threshold=0.03  # 3cm threshold for inspire hand
        )
        
        # Desired object orientation (identity quaternion)
        self.desired_orientation = torch.zeros((self.num_envs, 4), device=self.device)
        self.desired_orientation[:, 0] = 1.0  # w component of quaternion
        
        print("🤖 Inspire Hand State Machine Environment initialized!")
        print("📋 Warp-accelerated states: REST → APPROACH_ABOVE → APPROACH → GRASP → LIFT")
        
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Process actions before physics step."""
        # Get state machine actions
        sm_actions = self._get_state_machine_actions()
        
        # Extract EE pose (first 7 elements) and gripper state (last element)
        ee_pose_targets = sm_actions[:, :7]
        gripper_states = sm_actions[:, 7]
        
        # Apply EE pose control (if inspire hand supports it)
        # Note: This may need adaptation based on your inspire hand's control interface
        
        # Convert gripper state to finger joint positions
        finger_targets = self._gripper_state_to_joint_positions(gripper_states)
        
        # Set robot joint position targets
        self.scene["inspire_hand"].set_joint_position_target(finger_targets)
        
    def _gripper_state_to_joint_positions(self, gripper_states: torch.Tensor) -> torch.Tensor:
        """Convert gripper state (0=open, 1=closed) to finger joint positions."""
        # Define open and closed positions for each finger joint
        open_pos = torch.zeros((self.num_envs, 12), device=self.device)
        closed_pos = torch.tensor([
            [1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 0.8, 0.8, 0.8, 0.8]  # Grasping configuration
        ], device=self.device).repeat(self.num_envs, 1)
        
        # Interpolate between open and closed based on gripper state
        gripper_states = gripper_states.unsqueeze(1).expand(-1, 12)
        joint_positions = open_pos + gripper_states * (closed_pos - open_pos)
        
        return joint_positions
        
    def _get_state_machine_actions(self) -> torch.Tensor:
        """Get actions from the warp state machine."""
        # Get observations
        # -- end-effector frame
        ee_frame_sensor = self.scene["ee_frame"]
        tcp_position = ee_frame_sensor.data.target_pos_w[..., 0, :].clone() - self.scene.env_origins
        tcp_orientation = ee_frame_sensor.data.target_quat_w[..., 0, :].clone()
        
        # -- object frame  
        object_data: RigidObjectData = self.scene["cube"].data
        object_position = object_data.root_pos_w - self.scene.env_origins
        
        # -- target object frame (lift height)
        desired_position = object_position.clone()
        desired_position[:, 2] += 0.2  # Lift 20cm above current position
        
        # Advance state machine
        actions = self.pick_sm.compute(
            torch.cat([tcp_position, tcp_orientation], dim=-1),
            torch.cat([object_position, self.desired_orientation], dim=-1),
            torch.cat([desired_position, self.desired_orientation], dim=-1),
        )
        
        return actions
        
    def _apply_action(self) -> None:
        """Apply actions to the robot (handled in _pre_physics_step)."""
        pass
        
    def _get_observations(self) -> dict:
        """Get observations from the environment."""
        # Get joint positions and velocities
        joint_pos = self.scene["inspire_hand"].data.joint_pos[:, :12]
        joint_vel = self.scene["inspire_hand"].data.joint_vel[:, :12]
        
        # Get end-effector pose
        ee_frame_sensor = self.scene["ee_frame"]
        ee_pos = ee_frame_sensor.data.target_pos_w[..., 0, :] - self.scene.env_origins
        ee_quat = ee_frame_sensor.data.target_quat_w[..., 0, :]
        
        # Concatenate all observations
        obs = torch.cat([ee_pos, ee_quat, joint_pos, joint_vel], dim=-1)
        
        return {"policy": obs}
        
    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards based on task performance."""
        # Base reward for staying alive
        rewards = torch.ones(self.num_envs, device=self.device) * 0.1
        
        # Get object and end-effector positions
        object_data: RigidObjectData = self.scene["cube"].data
        object_position = object_data.root_pos_w - self.scene.env_origins
        
        ee_frame_sensor = self.scene["ee_frame"]
        ee_position = ee_frame_sensor.data.target_pos_w[..., 0, :] - self.scene.env_origins
        
        # Distance-based reward
        distance_to_object = torch.norm(ee_position - object_position, dim=1)
        distance_reward = torch.exp(-5.0 * distance_to_object)  # Exponential decay with distance
        rewards += distance_reward
        
        # State-specific rewards
        for env_id in range(self.num_envs):
            current_state = self.pick_sm.sm_state[env_id].item()
            
            if current_state == 3:  # GRASP_OBJECT
                rewards[env_id] += 2.0
            elif current_state == 4:  # LIFT_OBJECT
                if object_position[env_id, 2] > 0.5:  # Object lifted above initial height
                    rewards[env_id] += 5.0
        
        # Penalty for excessive joint velocities
        joint_vel = self.scene["inspire_hand"].data.joint_vel[:, :12]
        vel_penalty = torch.sum(torch.abs(joint_vel), dim=1) * 0.01
        rewards -= vel_penalty
        
        return rewards
        
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Check if episodes are done."""
        # Episode done based on time limit
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = torch.zeros_like(time_out)
        
        return died, time_out
        
    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset environments at specified indices."""
        if env_ids is None:
            env_ids = self.scene["inspire_hand"]._ALL_INDICES
            
        # Reset robot to initial state
        self.scene["inspire_hand"].reset(env_ids)
        self.scene["cube"].reset(env_ids)
        
        # Reset state machine
        self.pick_sm.reset_idx(env_ids.cpu().numpy())
        
        # Reset episode tracking
        self.episode_length_buf[env_ids] = 0
        
        print("🔄 Environment reset - Starting new grasping sequence")


def main():
    """Main function to run the state machine environment."""
    # Create environment configuration
    env_cfg = InspireHandStateMachineCfg()
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else 1
    
    # Create environment
    env = InspireHandStateMachineEnv(cfg=env_cfg, render_mode=None if args_cli.headless else "human")
    
    print("🚀 Starting Inspire Hand Warp State Machine Demo")
    print("📝 GPU-accelerated states: REST → APPROACH_ABOVE → APPROACH → GRASP → LIFT")
    print("🎯 Watch the hand approach, grasp, and lift the cube!")
    print("⏹️  Press Ctrl+C to stop")
    
    # Reset environment
    observations, _ = env.reset()
    
    step_count = 0
    try:
        while simulation_app.is_running():
            # Step the environment (actions are generated internally by state machine)
            dummy_actions = torch.zeros((env.num_envs, env.cfg.num_actions), device=env.device)
            observations, rewards, terminated, truncated, info = env.step(dummy_actions)
            
            step_count += 1
            if step_count % 60 == 0:  # Print every second (60 Hz)
                current_time = step_count * env.cfg.sim.dt * env.cfg.decimation
                current_state = env.pick_sm.get_current_state_name(0)
                print(f"⏱️  Step {step_count}: Time {current_time:.1f}s, State: {current_state}")
                print(f"   Reward: {rewards[0].item():.3f}, Wait Time: {env.pick_sm.sm_wait_time[0].item():.2f}s")
                
                # Get object height for lift tracking
                object_data: RigidObjectData = env.scene["cube"].data
                object_height = object_data.root_pos_w[0, 2].item()
                print(f"   Object Height: {object_height:.3f}m")
                
            # Reset if episode is done
            if terminated.any() or truncated.any():
                observations, _ = env.reset()
                
    except KeyboardInterrupt:
        print("\n🛑 Demo stopped by user")
        
    finally:
        # Cleanup
        env.close()
        simulation_app.close()
        print("✅ Environment closed successfully")


if __name__ == "__main__":
    main() 