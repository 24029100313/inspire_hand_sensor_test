#!/usr/bin/env python3

import cv2
import time
import signal
import sys
import threading
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pymodbus.client import ModbusTcpClient

# Import our custom modules
from mp_read_hand import HandDetector
from inspire_hand_controller import InspireHandController, HandState
from read_and_vis_data import TouchDataVisualizer, read_all_data
from pysnooper import snoop
class HandTeleoperationSystem:
	"""
	A system for teleoperating the Inspire Hand using hand tracking.
	
	This class integrates hand tracking from MediaPipe with the Inspire Hand controller
	to create a real-time teleoperation system.
	"""
	
	def __init__(self, hand_ip="192.168.11.210", hand_port=6000, target_hand="left", show_touch_data=True, use_simulated_touch=False):
		"""
		Initialize the hand teleoperation system.
		
		Args:
			hand_ip (str): IP address of the Inspire Hand
			hand_port (int): Port of the Inspire Hand
			target_hand (str): Which hand to track ('left', 'right', or 'both')
			show_touch_data (bool): Whether to show touch sensor data visualization
			use_simulated_touch (bool): Whether to use simulated touch data
		"""
		self.hand_ip = hand_ip
		self.hand_port = hand_port
		self.target_hand = target_hand
		self.running = False
		self.last_command_time = 0
		self.command_interval = 0.04  # Send commands every 0.1 seconds
		self.show_touch_data = show_touch_data
		self.use_simulated_touch = use_simulated_touch
		
		# Initialize the hand detector
		print(f"Initializing hand detector for {target_hand} hand...")
		self.detector = HandDetector(target_hand=target_hand)
		
		# Create separate Modbus clients for control and sensing
		self.control_client = None
		self.sensing_client = None
		
		# Initialize the Inspire Hand controller with its own client
		print(f"Connecting to Inspire Hand at {hand_ip}:{hand_port} for control...")
		self.control_client = ModbusTcpClient(host=hand_ip, port=hand_port)
		# Connect the client
		try:
			self.control_client.connect()
			print(f"Control client connected to {hand_ip}:{hand_port}")
		except Exception as e:
			print(f"Error connecting control client: {e}")
			
		self.hand_controller = InspireHandController(ip=hand_ip, port=hand_port, external_client=self.control_client)
		
		# Set default speeds and forces
		self.speeds = [1000, 1000, 1000, 1000, 1000, 1000]
		self.forces = [500, 500, 500, 500, 500, 500]
		
		# Initialize touch data visualizer if needed
		self.touch_visualizer = None
		self.touch_animation = None
		if show_touch_data:
			# Create a separate client for sensing
			print(f"Creating separate connection for touch sensing at {hand_ip}:{hand_port}...")
			if not use_simulated_touch:
				self.sensing_client = ModbusTcpClient(host=hand_ip, port=hand_port)
				# Connect the sensing client
				try:
					self.sensing_client.connect()
					print(f"Sensing client connected to {hand_ip}:{hand_port}")
					print(f"Initializing touch data visualizer with separate connection...")
					self.touch_visualizer = TouchDataVisualizer(use_simulated_data=use_simulated_touch, client=self.sensing_client)
				except Exception as e:
					print(f"Error connecting sensing client: {e}")
					print(f"Falling back to simulated touch data...")
					self.use_simulated_touch = True
					self.touch_visualizer = TouchDataVisualizer(use_simulated_data=True)
			else:
				print(f"Initializing touch data visualizer with simulated data...")
				self.touch_visualizer = TouchDataVisualizer(use_simulated_data=True)
	
	def connect_hand(self):
		"""
		Connect to the Inspire Hand.
		
		Returns:
			bool: Success status
		"""
		return self.hand_controller.connect()
	
	def disconnect_hand(self):
		"""
		Disconnect from the Inspire Hand.
		"""
		self.hand_controller.disconnect()
	
	def process_frame(self, frame):
		"""
		Process a video frame to detect hand landmarks and control the robotic hand.
		
		Args:
			frame: Input frame from camera
			
		Returns:
			frame: Annotated frame with hand landmarks
		"""
		# Detect hand landmarks
		landmarks, annotated_frame = self.detector.detect(frame)
		
		# If hand is detected, control the robotic hand
		if landmarks and len(landmarks) > 0:
			# Get the first detected hand (we only control one hand at a time)
			landmark_set = landmarks[0]
			
			# Convert landmarks to Inspire Hand format (0-1000 values)
			inspire_values = self.detector.convert_fingure_to_inspire(landmark_set)
			
			if inspire_values:
				# Check if it's time to send a new command
				current_time = time.time()
				if current_time - self.last_command_time >= self.command_interval:
					self.last_command_time = current_time
					
					# Extract finger values
					positions = [

						inspire_values['pinky_finger'],
						inspire_values['ring_finger'],
						inspire_values['middle_finger'],
						inspire_values['index_finger'],
						inspire_values['thumb'],
						# -1,  # Fixed thumb position
						  inspire_values['wrist']
					#    -1 
					]
					
					# Send command to the hand
					if self.hand_controller.get_state() == HandState.IDLE:
						self.hand_controller.set_positions(positions)
					
					# Display the values on the frame
					value_text = f"Hand Values: {','.join([str(p) for p in positions])}"
					cv2.putText(annotated_frame, value_text, (10, 30), 
								cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
		
		return annotated_frame
	
	def _setup_touch_visualization(self):
		"""
		Set up the touch data visualization without blocking.
		"""
		if self.touch_visualizer:
			# Create animation
			self.touch_animation = FuncAnimation(
				self.touch_visualizer.fig,
				self.touch_visualizer.update_plot,
				frames=range(1000),  # Limited number of frames
				interval=100,  # Update every 100ms
				blit=False,
				cache_frame_data=False,
			)
			
			# Set up the plot to be non-blocking
			plt.ion()  # Turn on interactive mode
			self.touch_visualizer.fig.show()
			plt.pause(0.1)  # Small pause to allow the window to appear
	@snoop()
	def run(self):
		"""
		Run the hand teleoperation system.
		"""
		# Connect to the Inspire Hand
		if not self.connect_hand():
			print("Failed to connect to the Inspire Hand. Exiting.")
			return
		
		# Set up touch data visualization if enabled (non-blocking)
		if self.show_touch_data and self.touch_visualizer:
			print("Setting up touch data visualization in a separate window...")
			self._setup_touch_visualization()
		
		# Initialize camera
		cap = cv2.VideoCapture(0)
		
		# Check if camera opened successfully
		if not cap.isOpened():
			print("Error: Could not open camera.")
			self.disconnect_hand()
			return
		
		# Set camera resolution
		cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
		cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
		
		self.running = True
		print("Hand teleoperation system is running. Press 'q' to quit.")
		
		try:
			# Main loop
			while self.running and cap.isOpened():
				# Read a frame from the camera
				success, frame = cap.read()
				if not success:
					print("Error: Could not read frame from camera.")
					break
				
				# Process the frame
				annotated_frame = self.process_frame(frame)
				
				# Display the frame
				cv2.imshow('Hand Teleoperation', annotated_frame)
				touch_data = read_all_data(self.sensing_client)
				print('touch data is ', touch_data)

				# Update the touch visualization if it's enabled
				if self.show_touch_data and self.touch_visualizer and plt.fignum_exists(self.touch_visualizer.fig.number):
					# Update the plot without blocking
					plt.pause(0.01)  # Small pause to update the plot
				
				# Break the loop if 'q' is pressed
				if cv2.waitKey(1) & 0xFF == ord('q'):
					break
				
		except KeyboardInterrupt:
			print("Interrupted by user.")
		finally:
			# Clean up
			self.running = False
			cap.release()
			cv2.destroyAllWindows()
			
			# Close touch visualizer if it exists
			if self.touch_visualizer:
				print("Closing touch data visualizer...")
				plt.close('all')
				if hasattr(self.touch_visualizer, 'close'):
					self.touch_visualizer.close()
			
			# Close the sensing client if it exists
			if self.sensing_client:
				# Check if client is connected using the appropriate method
				is_connected = False
				try:
					# Try new API first
					is_connected = self.sensing_client.connected
				except AttributeError:
					# Fall back to old API
					try:
						is_connected = self.sensing_client.is_socket_open()
					except Exception:
						pass
				
				if is_connected:
					print("Closing sensing Modbus client...")
					self.sensing_client.close()
			
			# Disconnect from the hand (will handle the control client)
			self.disconnect_hand()
			print("Hand teleoperation system stopped.")

# Global system for signal handler
system = None

def signal_handler(sig, frame):
	"""
	Handle Ctrl+C to gracefully stop the system.
	"""
	print("\nStopping hand teleoperation system...")
	if system:
		system.running = False

# Main entry point
if __name__ == "__main__":
	# Register signal handler
	signal.signal(signal.SIGINT, signal_handler)
	
	# Parse command line arguments
	import argparse
	
	parser = argparse.ArgumentParser(description="Hand teleoperation system")
	parser.add_argument("--ip", default="192.168.11.210", help="IP address of the Inspire Hand")
	parser.add_argument("--port", type=int, default=6000, help="Port of the Inspire Hand")
	parser.add_argument("--hand", default="left", choices=["left", "right", "both"], help="Which hand to track")
	parser.add_argument("--no-touch", action="store_true", help="Disable touch data visualization")
	parser.add_argument("--simulate-touch", action="store_true", help="Use simulated touch data instead of real sensor data")
	
	args = parser.parse_args()
	
	# Create and run the system
	system = HandTeleoperationSystem(
		hand_ip=args.ip,
		hand_port=args.port,
		target_hand=args.hand,
		show_touch_data=not args.no_touch,
		use_simulated_touch=args.simulate_touch
	)
	system.run()
