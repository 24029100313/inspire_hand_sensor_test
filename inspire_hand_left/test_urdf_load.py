import rerun as rr
import numpy as np
import pinocchio as pin
import os
import random
import time
import trimesh
import hppfcl
from typing import Dict, List, Optional, Tuple


class URDFVisualizer:
	def __init__(self, urdf_path: str, mesh_dir: Optional[str] = None):
		"""
		Initialize the URDF visualizer with Rerun.
		
		Args:
			urdf_path: Path to the URDF file
			mesh_dir: Directory containing mesh files (STL). If None, will look in 'meshes' directory
					 relative to the URDF file.
		"""
		# Initialize Rerun
		rr.init("urdf_visualization", spawn=True)
		
		# Load URDF model
		self.urdf_path = os.path.abspath(urdf_path)
		self.model = pin.buildModelFromUrdf(self.urdf_path)
		self.data = self.model.createData()
		
		# Set up mesh directory
		if mesh_dir is None:
			mesh_dir = os.path.join(os.path.dirname(os.path.dirname(self.urdf_path)), "meshes")
		
		# Load mesh files
		self.mesh_files = {}
		if os.path.exists(mesh_dir):
			for file in os.listdir(mesh_dir):
				if file.lower().endswith(".stl"):
					mesh_name = os.path.splitext(file)[0]
					self.mesh_files[mesh_name] = os.path.join(mesh_dir, file)
		
		print(f"Found {len(self.mesh_files)} mesh files")
		
		# Cache for link colors and meshes
		self.link_colors = {}
		self.link_meshes = {}
		
		# Get all body frames
		self.link_ids = {}
		for frame_id, frame in enumerate(self.model.frames):
			if frame.type == pin.FrameType.BODY:
				self.link_ids[frame.name] = frame_id
	
	def _get_mesh_points(self, link_name: str) -> np.ndarray:
		"""Get sampled points from mesh files matching the link name."""
		all_points = []
		link_name_lower = link_name.lower()
		matching_meshes = []
		
		# Try to match mesh files with link name
		
		for mesh_name, mesh_path in self.mesh_files.items():
			mesh_name_lower = mesh_name.lower()
			if (link_name_lower in mesh_name_lower or 
				mesh_name_lower in link_name_lower):
				matching_meshes.append(mesh_path)
		
		# Sample points from matching meshes
		for mesh_path in matching_meshes:
			try:
				if mesh_path not in self.link_meshes:
					self.link_meshes[mesh_path] = trimesh.load(mesh_path)
				
				mesh = self.link_meshes[mesh_path]
				# num_samples = min(500, max(100, 500 // max(1, len(matching_meshes))))
				num_samples = 300
				points, _ = trimesh.sample.sample_surface(mesh, num_samples)
				all_points.extend(points)
			except Exception as e:
				print(f"Error loading mesh {mesh_path}: {e}")
		
		return np.array(all_points) if all_points else None
	
	def _get_link_color(self, link_name: str) -> np.ndarray:
		"""Get a consistent color for the link."""
		if link_name not in self.link_colors:
			self.link_colors[link_name] = [
				random.randint(100, 255),
				random.randint(100, 255),
				random.randint(100, 255)
			]
		return self.link_colors[link_name]
	
	def render(self, joint_angles: np.ndarray) -> Dict[str, np.ndarray]:
		"""
		Render the robot with given joint angles.
		
		Args:
			joint_angles: Array of joint angles (must match model.nq)
			
		Returns:
			Dictionary mapping link names to their point clouds in world frame
		"""
		if len(joint_angles) != self.model.nq:
			raise ValueError(f"Expected {self.model.nq} joint angles, got {len(joint_angles)}")
		
		# Update forward kinematics
		q = np.array(joint_angles, dtype=np.float64)
		pin.forwardKinematics(self.model, self.data, q)
		pin.updateFramePlacements(self.model, self.data)
		
		result = {}
		
		# Visualize each link
		for link_name, frame_id in self.link_ids.items():
			# if 'force_sensor' in link_name:
				# continue
			print(link_name, frame_id)
			# Get link transform
			frame_placement = self.data.oMf[frame_id]
			rotation = frame_placement.rotation
			translation = frame_placement.translation
			
			# Get or generate points for this link
			local_points = self._get_mesh_points(link_name)
			if local_points is None or len(local_points) == 0:
				continue 
				# local_points = np.random.uniform(-0.02, 0.02, (100, 3))
			
			# Transform points to world frame
			global_points = (rotation @ local_points.T).T + translation
			result[link_name] = global_points
			
			# Get link color
			color = self._get_link_color(link_name)
			colors = np.tile(color, (len(global_points), 1)).astype(np.uint8)
			
			# Log to Rerun
			rr.set_time_seconds("sim_time", time.time())
			rr.log(f"robot/{link_name}", 
				  rr.Points3D(positions=global_points, colors=colors))
			
			# Add coordinate frame visualization
			# self._draw_coordinate_frame(link_name, translation, rotation)
		
		return result
	
	def _draw_coordinate_frame(self, link_name: str, position: np.ndarray, rotation: np.ndarray):
		"""Draw a coordinate frame at the given position and orientation."""
		# Define axis points (X, Y, Z in local frame)
		axes_points = np.array([
			[0, 0, 0], [0.03, 0, 0],  # X axis
			[0, 0, 0], [0, 0.03, 0],   # Y axis
			[0, 0, 0], [0, 0, 0.03]     # Z axis
		])
		
		# Transform to world frame
		global_axes = (rotation @ axes_points.T).T + position
		
		# Axis colors (RGB)
		colors = np.array([
			[255, 0, 0], [255, 0, 0],  # Red X
			[0, 255, 0], [0, 255, 0],  # Green Y
			[0, 0, 255], [0, 0, 255]   # Blue Z
		], dtype=np.uint8)
		
		# Log to Rerun
		rr.log(f"robot/{link_name}/axes", 
			  rr.LineStrips3D(
				  strips=global_axes.reshape(3, 2, 3),
				  colors=colors.reshape(3, 2, 3)
			  ))


# Example usage
if __name__ == "__main__":
	import numpy as np
	
	# Initialize visualizer
	current_dir = os.path.dirname(os.path.abspath(__file__))
	urdf_file = os.path.join(current_dir, 'urdf_left_with_force_sensor/urdf/urdf_left_with_force_sensor.urdf')
	visualizer = URDFVisualizer(urdf_file)
	
	# Example: Render with random joint angles
	q = np.ones(visualizer.model.nq) * 0 # All zeros (home position)
	visualizer.render(q)
	
	print(f"URDF model loaded: {urdf_file}")
	print(f"Model has {visualizer.model.nq} degrees of freedom")
	print(f"Visualizing {len(visualizer.link_ids)} links")
