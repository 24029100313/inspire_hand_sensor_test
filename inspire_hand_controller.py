import time
import sys 
import signal
import os 
import threading
import enum
from pymodbus.client.sync import ModbusTcpClient
from pymodbus.exceptions import ModbusException

class HandState(enum.Enum):
	"""Enum for the different states of the hand controller."""
	IDLE = 0
	MOVING = 1
	ERROR = 2
	CHECKING = 3

class InspireHandController:
	"""
	Controller for the Inspire robotic hand.
	
	Implements a state machine for controlling the hand via Modbus.
	Features:
	- Automatic error checking every 5 seconds
	- Action space verification
	- Waiting for actions to complete
	"""
	
	# Register address dictionary
	REG_DICT = {
		"ID": 1000,
		"baudrate": 1001,
		"clearErr": 1004,
		"forceClb": 1009,
		"angleSet": 1486,
		"forceSet": 1498,
		"speedSet": 1522,
		"angleAct": 1546,
		"forceAct": 1582,
		"errCode": 1606,
		"statusCode": 1612,
		"temp": 1618,
		"actionSeq": 2320,
		"actionRun": 2322,
	}
	
	# Modbus maximum registers per read
	MAX_REGISTERS_PER_READ = 125
	
	# Action space constraints
	MIN_SPEED = 0
	MAX_SPEED = 1000
	MIN_POSITION = 0
	MAX_POSITION = 1000
	DEFAULT_THUMB_POSITION = 400
	
	# Timing constants
	ERROR_CHECK_INTERVAL = 5.0  # seconds
	ACTION_DELAY = 0.6  # seconds
	ACTION_TIMEOUT = 1.5  # seconds
	
	def __init__(self, ip="192.168.11.210", port=6000):
		"""
		Initialize the hand controller.
		
		Args:
			ip (str): IP address of the Modbus server
			port (int): Port of the Modbus server
		"""
		self.ip = ip
		self.port = port
		self.client = None
		self.state = HandState.IDLE
		self.error_codes = [0, 0, 0, 0, 0, 0]
		self.current_angles = [0, 0, 0, 0, self.DEFAULT_THUMB_POSITION, 1000]
		self.target_angles = [0, 0, 0, 0, self.DEFAULT_THUMB_POSITION, 1000]
		self.speeds = [500, 500, 500, 500, 500, 500]
		self.forces = [500, 500, 500, 500, 500, 500]
		
		# For threading
		self.running = False
		self.error_check_thread = None
		self.action_thread = None
		self.lock = threading.Lock()
	
	def connect(self):
		"""Connect to the Modbus server."""
		try:
			self.client = ModbusTcpClient(self.ip, self.port)
			connection_status = self.client.connect()
			if not connection_status:
				print(f"Failed to connect to Modbus device at {self.ip}:{self.port}")
				return False
			print(f"Successfully connected to Modbus device at {self.ip}:{self.port}")
			
			# Start error checking thread
			self.running = True
			self.error_check_thread = threading.Thread(target=self._error_check_loop)
			self.error_check_thread.daemon = True
			self.error_check_thread.start()
			
			# Set initial speeds and forces
			self.set_speeds(self.speeds)
			self.set_forces(self.forces)
			
			return True
		except Exception as e:
			print(f"Error connecting to Modbus device: {e}")
			return False
	
	def disconnect(self):
		"""Disconnect from the Modbus server."""
		self.running = False
		if self.error_check_thread:
			self.error_check_thread.join(timeout=1.0)
		
		if self.action_thread and self.action_thread.is_alive():
			self.action_thread.join(timeout=1.0)
			
		if self.client and self.client.is_socket_open():
			print("Closing Modbus client connection...")
			self.client.close()
			print("Connection closed.")
	
	def _write_register(self, address, values, reg_name=None):
		"""
		Write values to a register.
		
		Args:
			address (int): Register address
			values (list or int): Values to write
			reg_name (str): Optional register name for context
		
		Returns:
			bool: Success status
		"""
		try:
			with self.lock:
				if isinstance(values, list):
					self.client.write_registers(address, values)
				else:
					self.client.write_register(address, values)
					
				if reg_name and 'angleset' in reg_name.lower():
					time.sleep(self.ACTION_DELAY)
			return True
		except Exception as e:
			print(f"Error writing to register {address}: {e}")
			return False
	
	def _read_register(self, address, count):
		"""
		Read values from a register.
		
		Args:
			address (int): Register address
			count (int): Number of registers to read
		
		Returns:
			list: Register values
		"""
		try:
			with self.lock:
				response = self.client.read_holding_registers(address, count)
			if response.isError():
				print(f"Error reading register {address}: {response}")
				return []
			return response.registers
		except Exception as e:
			print(f"Error reading register {address}: {e}")
			return []
	
	def _write6(self, reg_name, values):
		"""
		Write 6 values to a register.
		
		Args:
			reg_name (str): Register name
			values (list): 6 values to write
		
		Returns:
			bool: Success status
		"""
		if reg_name not in ["angleSet", "forceSet", "speedSet"]:
			print(f"Invalid register name: {reg_name}")
			return False
			
		# Ensure we have 6 values
		if len(values) != 6:
			print(f"Expected 6 values, got {len(values)}")
			return False
			
		# Convert values to 16-bit integers
		val_reg = []
		for i in range(6):
			val_reg.append(values[i] & 0xFFFF)  # Get low 16 bits
			
		return self._write_register(self.REG_DICT[reg_name], val_reg, reg_name)
	
	def _read6(self, reg_name):
		"""
		Read 6 values from a register.
		
		Args:
			reg_name (str): Register name
		
		Returns:
			list: Register values
		"""
		if reg_name in ["angleSet", "forceSet", "speedSet", "angleAct", "forceAct"]:
			values = self._read_register(self.REG_DICT[reg_name], 6)
			if not values or len(values) < 6:
				print(f"Failed to read {reg_name}")
				return []
			return values
			
		elif reg_name in ["errCode", "statusCode", "temp"]:
			values = self._read_register(self.REG_DICT[reg_name], 3)
			if not values or len(values) < 3:
				print(f"Failed to read {reg_name}")
				return []
				
			# Process high and low bytes
			results = []
			for val in values:
				results.append(val & 0xFF)  # Low byte
				results.append((val >> 8) & 0xFF)  # High byte
				
			return results
		else:
			print(f"Invalid register name: {reg_name}")
			return []
	
	def _check_action_space(self, positions=None, speeds=None):
		"""
		Check if the given positions and speeds are within the allowed range.
		
		Args:
			positions (list): List of 6 positions
			speeds (list): List of 6 speeds
		
		Returns:
			bool: True if values are within range, False otherwise
		"""
		if positions:
			for i, pos in enumerate(positions):
				if pos != -1:  # -1 means no change
					if pos < self.MIN_POSITION or pos > self.MAX_POSITION:
						print(f"Position {i} ({pos}) out of range [{self.MIN_POSITION}, {self.MAX_POSITION}]")
						return False
		
		if speeds:
			for i, speed in enumerate(speeds):
				if speed != -1:  # -1 means no change
					if speed < self.MIN_SPEED or speed > self.MAX_SPEED:
						print(f"Speed {i} ({speed}) out of range [{self.MIN_SPEED}, {self.MAX_SPEED}]")
						return False
		
		return True
	
	def _enforce_thumb_position(self, positions):
		"""
		Ensure the thumb position is always at DEFAULT_THUMB_POSITION.
		
		Args:
			positions (list): List of 6 positions
		
		Returns:
			list: Modified positions with thumb at DEFAULT_THUMB_POSITION
		"""
		positions[4] = self.DEFAULT_THUMB_POSITION
		return positions
	
	def _error_check_loop(self):
		"""Background thread for periodic error checking."""
		while self.running:
			if self.state != HandState.ERROR:
				self.state = HandState.CHECKING
				error_codes = self._read6("errCode")
				if error_codes:
					with self.lock:
						self.error_codes = error_codes
					# Check if any error code is non-zero
					if any(code != 0 for code in error_codes):
						print(f"Error detected: {error_codes}")
						self.state = HandState.ERROR
					else:
						self.state = HandState.IDLE
				else:
					print("Failed to read error codes")
					self.state = HandState.IDLE
			
			# Sleep for ERROR_CHECK_INTERVAL seconds
			time.sleep(self.ERROR_CHECK_INTERVAL)
	
	def _wait_for_action_completion(self, timeout=None):
		"""
		Wait for the hand to reach the target position.
		
		Args:
			timeout (float): Maximum time to wait in seconds
			
		Returns:
			bool: True if action completed, False if timed out
		"""
		if timeout is None:
			timeout = self.ACTION_TIMEOUT
			
		start_time = time.time()
		while time.time() - start_time < timeout:
			current_angles = self._read6("angleAct")
			if not current_angles:
				time.sleep(0.1)
				continue
				
			with self.lock:
				self.current_angles = current_angles
				
			# Check if all angles are close to target
			all_reached = True
			# print(current_angles, self.target_angles)
			for i, (current, target) in enumerate(zip(current_angles, self.target_angles)):
				if target == -1:  # Skip if target is -1 (no change)
					continue
				if abs(current - target) > 20:  # Allow small tolerance
					all_reached = False
					break
			
			if all_reached:
				return True
				
			time.sleep(0.1)
			
		return False
	
	def _execute_action(self, positions):
		"""
		Execute the action with the given positions.
		
		Args:
			positions (list): Target positions
			
		Returns:
			bool: Success status
		"""
		self.state = HandState.MOVING
		
		# Read current angles first
		current_angles = self._read6("angleAct")
		if not current_angles or len(current_angles) < 6:
			print("Failed to read current angles")
			current_angles = self.current_angles
		
		# Create a new positions list with -1 for small changes
		filtered_positions = positions.copy()
		for i, pos in enumerate(positions):
			if pos != -1:  # Only check positions that are not already -1
				# If the change is less than 20, don't move this finger
				if abs(pos - current_angles[i]) < 20:
					filtered_positions[i] = -1
					print(f"Skipping finger {i} (change too small: {abs(pos - current_angles[i])})")
				else:
					# Set the target angle for significant changes
					with self.lock:
						self.target_angles[i] = pos
		
		# Check if any fingers need to move
		if all(pos == -1 for pos in filtered_positions):
			print("No significant changes detected, skipping command")
			self.state = HandState.IDLE
			return True
		
		# Write the filtered target angles to the register
		success = self._write6("angleSet", filtered_positions)
		time.sleep(self.ACTION_DELAY)
		
		
		if not success:
			self.state = HandState.IDLE
			return False
		
		# Wait a short time before the action starts
		time.sleep(self.ACTION_DELAY)
		
		# Wait for the action to complete
		action_completed = self._wait_for_action_completion()
		
		self.state = HandState.IDLE
		return action_completed
	
	def set_positions(self, positions):
		"""
		Set the target positions for all fingers.
		
		Args:
			positions (list): List of 6 positions (-1 means no change)
			
		Returns:
			bool: Success status
		"""
		if len(positions) != 6:
			print(f"Expected 6 positions, got {len(positions)}")
			return False
			
		# Check if positions are within range
		if not self._check_action_space(positions=positions):
			return False
			
		# Enforce thumb position
		positions = self._enforce_thumb_position(positions.copy())
		
		# Execute action in background thread
		if self.action_thread and self.action_thread.is_alive():
			print("Another action is already in progress")
			return False
			
		self.action_thread = threading.Thread(target=self._execute_action, args=(positions,))
		self.action_thread.daemon = True
		self.action_thread.start()
		
		return True
	
	def set_speeds(self, speeds):
		"""
		Set the speeds for all fingers.
		
		Args:
			speeds (list): List of 6 speeds (-1 means no change)
			
		Returns:
			bool: Success status
		"""
		if len(speeds) != 6:
			print(f"Expected 6 speeds, got {len(speeds)}")
			return False
			
		# Check if speeds are within range
		if not self._check_action_space(speeds=speeds):
			return False
			
		# Update speeds
		with self.lock:
			for i, speed in enumerate(speeds):
				if speed != -1:  # Skip if speed is -1 (no change)
					self.speeds[i] = speed
		
		# Write speeds to register
		return self._write6("speedSet", speeds)
	
	def set_forces(self, forces):
		"""
		Set the forces for all fingers.
		
		Args:
			forces (list): List of 6 forces (-1 means no change)
			
		Returns:
			bool: Success status
		"""
		if len(forces) != 6:
			print(f"Expected 6 forces, got {len(forces)}")
			return False
			
		# Update forces
		with self.lock:
			for i, force in enumerate(forces):
				if force != -1:  # Skip if force is -1 (no change)
					self.forces[i] = force
		
		# Write forces to register
		return self._write6("forceSet", forces)
	
	def get_current_angles(self):
		"""
		Get the current angles of all fingers.
		
		Returns:
			list: Current angles
		"""
		current_angles = self._read6("angleAct")
		if current_angles:
			with self.lock:
				self.current_angles = current_angles
			return current_angles
		return self.current_angles
	
	def get_error_codes(self):
		"""
		Get the current error codes.
		
		Returns:
			list: Error codes
		"""
		return self.error_codes
	
	def clear_errors(self):
		"""
		Clear all error codes.
		
		Returns:
			bool: Success status
		"""
		return self._write_register(self.REG_DICT["clearErr"], [1])
	
	def get_state(self):
		"""
		Get the current state of the hand controller.
		
		Returns:
			HandState: Current state
		"""
		return self.state
	
	def wait_for_idle(self, timeout=5.0):
		"""
		Wait until the hand controller is idle.
		
		Args:
			timeout (float): Maximum time to wait in seconds
			
		Returns:
			bool: True if idle, False if timed out
		"""
		start_time = time.time()
		while time.time() - start_time < timeout:
			if self.state == HandState.IDLE:
				return True
			time.sleep(0.1)
		return False
	
	def open_close_fingers(self, repeats=2):
		"""
		Open and close each finger individually (fingers 0-3) for the specified number of repeats.
		
		Args:
			repeats (int): Number of times to repeat the open/close sequence for each finger
		
		Returns:
			bool: Success status
		"""
		print("Starting individual finger open/close sequence")
		
		# First, open all fingers using direct method to avoid threading issues
		print("closing all fingers")
		self._execute_action([0, 0, 0, 0, 400, -1])
		print("Opening all fingers")
		self._execute_action([1000, 1000, 1000, 1000, 400, -1])
		
		# Test each finger 0-3 (pinky, ring, middle, index)
		for finger in range(4):
			finger_name = ["pinky", "ring", "middle", "index"][finger]
			print(f"Testing {finger_name} finger (finger {finger})")
			
			for rep in range(repeats):
				# Create a positions list with all fingers open except the current one
				close_positions = [-1, -1, -1, -1, -1, -1]
				close_positions[finger] = 0  # Close current finger
				
				print(f"  Repeat {rep+1}/{repeats}: Closing {finger_name} finger")
				self._execute_action(close_positions)
				
				print('now going to stop ')
				# time.sleep(2)
				# Now open it again
				print(f"  Repeat {rep+1}/{repeats}: Opening {finger_name} finger")
				open_positions = close_positions.copy()
				open_positions[finger] = 1000  # Open current finger
				self._execute_action(open_positions)
		
		# Finally, open all fingers again
		print("Finished finger tests, opening all fingers")
		self._execute_action([1000, 1000, 1000, 1000, 400, -1])
		
		return True


# Global controller for signal handler
controller = None

# Signal handler for graceful shutdown
def signal_handler(sig, frame):
	print('\nCtrl+C pressed. Shutting down gracefully...')
	if controller:
		controller.disconnect()
		print("Disconnected from hand")
	sys.exit(0)

# Example usage
if __name__ == "__main__":
	# Register signal handler
	signal.signal(signal.SIGINT, signal_handler)
	
	controller = InspireHandController()
	try:
		if controller.connect():
			print("Connected to hand")
			
			# Set speeds and forces
			# controller.set_speeds([500, 500, 500, 500, 500, 500])
			controller.set_speeds([1000, 1000, 1000, 1000, 1000, 1000])
			# controller.set_forces([-1, -1, -1, -1, -1, -1])
			
			# Wait a bit for initialization
			time.sleep(1)
			
			# # Move fingers to open position
			# print("Opening hand")
			# controller.set_positions([0, 0, 0, 0, 400, -1])
			# time.sleep(2)
			
			# Test opening and closing individual fingers
			print("Testing individual finger movements")
			print("(Press Ctrl+C to stop and disconnect)")
			controller.open_close_fingers(repeats=2)
			
			# time.sleep(2)
			
			# Read current angles
			angles = controller.get_current_angles()
			print(f"Current angles: {angles}")
			
			# Read error codes
			errors = controller.get_error_codes()
			print(f"Error codes: {errors}")
			
			# Wait for hand to be idle
			controller.wait_for_idle()
			print("Hand is idle")
			
	finally:
		# This will be called if no Ctrl+C was pressed
		if controller:
			controller.disconnect()
			print("Disconnected from hand")
