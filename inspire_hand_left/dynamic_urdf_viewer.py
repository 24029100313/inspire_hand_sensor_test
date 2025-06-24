import os
import time
import numpy as np
import rerun as rr
from typing import Dict, List, Optional
import xml.etree.ElementTree as ET

class URDFViewer:
    def __init__(self, urdf_path: str):
        """
        Initialize the URDF viewer.
        
        Args:
            urdf_path: Path to the URDF file
        """
        self.urdf_path = os.path.abspath(urdf_path)
        self.joint_positions: Dict[str, float] = {}
        self.joint_names: List[str] = []
        
        # Initialize Rerun
        rr.init("urdf_viewer", spawn=True)
        
        # Load the URDF
        self._load_urdf()
    
    def _load_urdf(self):
        """Load the URDF file and extract joint information."""
        try:
            # Parse the URDF file
            tree = ET.parse(self.urdf_path)
            root = tree.getroot()
            
            # Extract all joint names
            self.joint_names = []
            for joint in root.findall('joint'):
                if joint.get('type') in ['revolute', 'prismatic', 'continuous']:
                    joint_name = joint.get('name')
                    self.joint_names.append(joint_name)
                    self.joint_positions[joint_name] = 0.0  # Initialize to zero position
            
            # Log the URDF to Rerun with rotation
            rr.log(
                "robot",
                rr.Transform3D(
                    translation=[0, 0, 0],
                    # Rotate 90 degrees around X-axis
                    rotation=rr.Quaternion(xyzw=[0.7071, 0, 0, 0.7071])  # 90 degrees around X
                )
            )
            
            # Log the URDF file path
            with open(self.urdf_path, 'r') as f:
                urdf_content = f.read()
            rr.log("robot/urdf", rr.TextDocument(urdf_content))
            
            print(f"Loaded URDF with {len(self.joint_names)} joints:")
            for name in self.joint_names:
                print(f"- {name}")
                
                # Initialize joint positions to zero
                self.joint_positions[name] = 0.0
                
        except Exception as e:
            print(f"Error loading URDF: {e}")
            raise
    
    def update_joints(self, joint_positions: Dict[str, float]):
        """
        Update joint positions.
        
        Args:
            joint_positions: Dictionary mapping joint names to their positions
        """
        # Update only the joints that exist in our model
        for name, position in joint_positions.items():
            if name in self.joint_positions:
                self.joint_positions[name] = position
        
        # Log the joint states
        if self.joint_positions:
            rr.log(
                "robot/joint_states",
                rr.JointState(
                    names=list(self.joint_positions.keys()),
                    positions=list(self.joint_positions.values())
                )
            )
    
    def animate_sine_wave(self, duration: float = 10.0, frequency: float = 1.0, amplitude: float = 1.0):
        """
        Animate the robot with a sine wave motion.
        
        Args:
            duration: Total animation duration in seconds
            frequency: Frequency of the sine wave
            amplitude: Amplitude of the motion
        """
        start_time = time.time()
        end_time = start_time + duration
        
        print(f"Starting animation for {duration} seconds...")
        print("Press Ctrl+C to stop early.")
        
        try:
            while time.time() < end_time:
                t = time.time() - start_time
                
                # Update joint positions using sine wave
                new_positions = {}
                for i, joint_name in enumerate(self.joint_names):
                    phase = 2 * np.pi * i / max(1, len(self.joint_names))
                    new_positions[joint_name] = amplitude * np.sin(2 * np.pi * frequency * t + phase)
                
                # Update the visualization
                self.update_joints(new_positions)
                
                # Add a small delay to control the update rate
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            print("\nAnimation stopped by user.")

def main():
    # Get the path to the URDF file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    urdf_path = os.path.join(current_dir, 'urdf_left_with_force_sensor/urdf/urdf_left_with_force_sensor.urdf')
    
    if not os.path.exists(urdf_path):
        print(f"Error: URDF file not found at {urdf_path}")
        print("Please make sure the file exists and the path is correct.")
        return
    
    print(f"Loading URDF from: {urdf_path}")
    
    # Create and run the viewer
    try:
        viewer = URDFViewer(urdf_path)
        
        # Example: Animate the robot with a sine wave
        viewer.animate_sine_wave(duration=60.0, frequency=0.5, amplitude=1.0)
        
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("Visualization complete.")

if __name__ == "__main__":
    main()