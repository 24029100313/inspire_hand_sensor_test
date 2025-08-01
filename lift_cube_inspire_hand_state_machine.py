#!/usr/bin/env python3

"""
Inspire Hand State Machine Cube Grasping Environment

This environment demonstrates a state machine approach to grasping a cube with the Inspire Hand.
No sensors or MediaPipe required - pure state machine control.

Features:
- State machine for grasping sequence
- No MediaPipe dependency
- Uses inspire_hand_processed.usd (no sensors)
- Automatic cube grasping demonstration

States:
1. APPROACH - Move hand to cube
2. GRASP - Close fingers around cube
3. LIFT - Lift the cube up
4. HOLD - Hold the cube for a moment
5. RELEASE - Open fingers and release
6. RESET - Return to initial position
"""

import argparse
import math
import torch
import numpy as np
from enum import Enum

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
import torch
import numpy as np

# Isaac Lab imports
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.sim import SimulationCfg

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
            pos=(0.0, 0.0, 0.5),
            joint_pos={
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
                effort_limit=10.0,
                velocity_limit=3.14,
                stiffness=40.0,
                damping=2.0,
            ),
        },
    )

    # Test cube for visual reference
    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.0, 0.0), 
                metallic=0.2,
                roughness=0.4,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.1, 0.0, 0.52),  # Positioned for grasping
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )


class GraspState(Enum):
    """Enumeration of grasping states"""
    APPROACH = 1
    GRASP = 2
    LIFT = 3
    HOLD = 4
    RELEASE = 5
    RESET = 6


@configclass
class InspireHandStateMachineCfg(DirectRLEnvCfg):
    """Configuration for Inspire Hand State Machine Environment."""
    
    # simulation settings
    sim: SimulationCfg = SimulationCfg(dt=1 / 60, render_interval=1)
    
    # scene settings
    scene: InspireHandSceneCfg = InspireHandSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)
    
    # basic settings
    decimation = 2
    episode_length_s = 30.0
    action_space = 12  # 12 finger joints
    observation_space = 24  # 12 joint positions + 12 joint velocities
    num_actions = 12
    num_observations = 24


class InspireHandStateMachineEnv(DirectRLEnv):
    """Inspire Hand State Machine Environment for cube grasping."""
    
    cfg: InspireHandStateMachineCfg
    
    def __init__(self, cfg: InspireHandStateMachineCfg, render_mode: str | None = None, **kwargs):
        """Initialize the environment."""
        super().__init__(cfg, render_mode, **kwargs)
        
        # State machine variables
        self.current_state = GraspState.APPROACH
        self.state_timer = 0.0
        self.state_duration = 3.0  # seconds per state
        
        # Joint names for easy access
        self.joint_names = [
            "right_index_1_joint", "right_index_2_joint",
            "right_middle_1_joint", "right_middle_2_joint", 
            "right_ring_1_joint", "right_ring_2_joint",
            "right_little_1_joint", "right_little_2_joint",
            "right_thumb_1_joint", "right_thumb_2_joint",
            "right_thumb_3_joint", "right_thumb_4_joint"
        ]
        
        # Define joint positions for different hand poses
        self.hand_poses = {
            'open': [0.0] * 12,  # All joints at 0 (open hand)
            'grasp': [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.5, 0.5, 0.5, 0.5],  # Grasping pose
            'approach': [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1],  # Pre-grasp pose
        }
        
        print("🤖 Inspire Hand State Machine Environment initialized!")
        print("📋 States: APPROACH → GRASP → LIFT → HOLD → RELEASE → RESET")
        
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Process actions before physics step."""
        # Get state machine actions instead of using provided actions
        state_actions = self._get_state_machine_actions()
        
        # Set robot joint position targets
        self.scene["inspire_hand"].set_joint_position_target(state_actions)
        
    def _get_state_machine_actions(self) -> torch.Tensor:
        """Get actions based on current state machine state."""
        # Update state timer
        dt = self.cfg.sim.dt * self.cfg.decimation
        self.state_timer += dt
        
        # State transitions
        if self.state_timer >= self.state_duration:
            self._transition_to_next_state()
            self.state_timer = 0.0
            
        # Get target pose based on current state
        if self.current_state == GraspState.APPROACH:
            target_pose = self.hand_poses['approach']
            
        elif self.current_state == GraspState.GRASP:
            target_pose = self.hand_poses['grasp']
            
        elif self.current_state == GraspState.LIFT:
            # Keep grasping pose but we could add slight adjustments
            target_pose = self.hand_poses['grasp']
            
        elif self.current_state == GraspState.HOLD:
            target_pose = self.hand_poses['grasp']
            
        elif self.current_state == GraspState.RELEASE:
            target_pose = self.hand_poses['open']
            
        elif self.current_state == GraspState.RESET:
            target_pose = self.hand_poses['open']
            
        else:
            target_pose = self.hand_poses['open']
            
        # Convert to tensor
        actions = torch.tensor(target_pose, dtype=torch.float32, device=self.device)
        actions = actions.unsqueeze(0)  # Add batch dimension
        
        return actions
        
    def _transition_to_next_state(self):
        """Transition to the next state in the sequence."""
        if self.current_state == GraspState.APPROACH:
            self.current_state = GraspState.GRASP
            print("🎯 State: APPROACH → GRASP")
            
        elif self.current_state == GraspState.GRASP:
            self.current_state = GraspState.LIFT
            print("🤏 State: GRASP → LIFT")
            
        elif self.current_state == GraspState.LIFT:
            self.current_state = GraspState.HOLD
            print("⬆️ State: LIFT → HOLD")
            
        elif self.current_state == GraspState.HOLD:
            self.current_state = GraspState.RELEASE
            print("✋ State: HOLD → RELEASE")
            
        elif self.current_state == GraspState.RELEASE:
            self.current_state = GraspState.RESET
            print("📤 State: RELEASE → RESET")
            
        elif self.current_state == GraspState.RESET:
            self.current_state = GraspState.APPROACH
            print("🔄 State: RESET → APPROACH (cycle restart)")
            
    def _apply_action(self) -> None:
        """Apply actions to the robot (handled in _pre_physics_step)."""
        pass
        
    def _get_observations(self) -> dict:
        """Get observations from the environment."""
        # Get joint positions and velocities
        joint_pos = self.scene["inspire_hand"].data.joint_pos[:, :12]  # First 12 joints
        joint_vel = self.scene["inspire_hand"].data.joint_vel[:, :12]  # First 12 joints
        
        # Concatenate observations
        obs = torch.cat([joint_pos, joint_vel], dim=-1)
        
        return {"policy": obs}
        
    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards based on task performance."""
        # Simple reward: stay alive bonus + joint velocity penalty
        rewards = torch.ones(self.num_envs, device=self.device) * 1.0
        
        # Penalty for high joint velocities (encourage smooth motion)
        joint_vel = self.scene["inspire_hand"].data.joint_vel[:, :12]
        vel_penalty = torch.sum(torch.abs(joint_vel), dim=1) * 0.01
        rewards -= vel_penalty
        
        # State-specific rewards
        if self.current_state == GraspState.GRASP:
            # Bonus for maintaining grasp position
            rewards += 0.5
            
        elif self.current_state == GraspState.HOLD:
            # Higher bonus for successful holding
            rewards += 1.0
            
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
        
        # Reset state machine
        self.current_state = GraspState.APPROACH
        self.state_timer = 0.0
        
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
    
    print("🚀 Starting Inspire Hand State Machine Cube Grasping Demo")
    print("📝 Watch the hand go through: APPROACH → GRASP → LIFT → HOLD → RELEASE → RESET")
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
                print(f"⏱️  Step {step_count}: Time {current_time:.1f}s, State: {env.current_state.name}")
                print(f"   Reward: {rewards[0].item():.3f}")
                
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