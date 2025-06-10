from urdfpy import URDF
import trimesh

# Load URDF
robot = URDF.load('/Users/lr-2002/project/tele_hand/inspire_hand/inspire_hand_left/urdf_left_with_force_sensor/urdf/urdf_left_with_force_sensor.urdf')

# Create a 3D mesh for visualization
robot_mesh = robot.collision_meshes()

# Visualize
scene = trimesh.Scene(robot_mesh)
scene.show()