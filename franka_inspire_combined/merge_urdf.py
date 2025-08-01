#!/usr/bin/env python3
"""
Script to merge Franka URDF with Inspire Hand URDF
Creates a combined robot with Franka arm + Inspire Hand as end effector
"""

import re

def extract_inspire_hand_content():
    """Extract all links and joints from Inspire Hand URDF (except world)"""
    inspire_urdf_path = "/home/larry/NVIDIA_DEV/isaac_grasp_ws/cabinet_sensor_project/inspire_hand_with_sensors/urdf/inspire_hand_processed.urdf"
    
    with open(inspire_urdf_path, 'r') as f:
        content = f.read()
    
    # Remove everything before and including the first <link name="world"/>
    content = re.sub(r'.*?<link name="world"/>\s*', '', content, flags=re.DOTALL)
    
    # Remove the <joint name="base_joint"> that connects world to base_link
    content = re.sub(r'<joint name="base_joint"[^>]*>.*?</joint>\s*', '', content, flags=re.DOTALL)
    
    # Remove the closing </robot> tag
    content = re.sub(r'</robot>\s*$', '', content, flags=re.DOTALL)
    
    return content.strip()

def create_connection_joint():
    """Create joint to connect Inspire Hand base_link to Franka end effector"""
    return '''
  <!-- Connection joint: Franka end effector to Inspire Hand -->
  <joint name="franka_to_inspire_hand" type="fixed">
    <origin rpy="0 0 0" xyz="0 0 0"/>
    <parent link="panda_end_effector"/>
    <child link="base_link"/>
  </joint>
'''

def merge_urdf_files():
    """Merge Franka and Inspire Hand URDF files"""
    franka_urdf_path = "urdf/franka_inspire_combined.urdf"
    
    # Read Franka URDF
    with open(franka_urdf_path, 'r') as f:
        franka_content = f.read()
    
    # Extract Inspire Hand content
    inspire_content = extract_inspire_hand_content()
    
    # Create connection joint
    connection_joint = create_connection_joint()
    
    # Insert Inspire Hand content before closing </robot> tag
    combined_content = franka_content.replace(
        '</robot>', 
        connection_joint + '\n' + inspire_content + '\n\n</robot>'
    )
    
    # Update robot name
    combined_content = combined_content.replace(
        'name="franka"',
        'name="franka_inspire_combined"'
    )
    
    # Write combined URDF
    output_path = "urdf/franka_inspire_combined.urdf"
    with open(output_path, 'w') as f:
        f.write(combined_content)
    
    print(f"✅ Combined URDF created: {output_path}")
    print("🤖 Robot name: franka_inspire_combined")
    print("🔗 Connection: panda_end_effector → base_link (Inspire Hand)")

if __name__ == "__main__":
    merge_urdf_files() 