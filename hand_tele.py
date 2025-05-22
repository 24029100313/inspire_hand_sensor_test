#!/usr/bin/env python3

import cv2
import time
import signal
import sys
import threading
import numpy as np

# Import our custom modules
from mp_read_hand import HandDetector
from inspire_hand_controller import InspireHandController, HandState

class HandTeleoperationSystem:
    """
    A system for teleoperating the Inspire Hand using hand tracking.
    
    This class integrates hand tracking from MediaPipe with the Inspire Hand controller
    to create a real-time teleoperation system.
    """
    
    def __init__(self, hand_ip="192.168.11.210", hand_port=6000, target_hand="left"):
        """
        Initialize the hand teleoperation system.
        
        Args:
            hand_ip (str): IP address of the Inspire Hand
            hand_port (int): Port of the Inspire Hand
            target_hand (str): Which hand to track ('left', 'right', or 'both')
        """
        self.target_hand = target_hand
        self.running = False
        self.last_command_time = 0
        self.command_interval = 0.1  # Send commands every 0.1 seconds
        
        # Initialize the hand detector
        print(f"Initializing hand detector for {target_hand} hand...")
        self.detector = HandDetector(target_hand=target_hand)
        
        # Initialize the Inspire Hand controller
        print(f"Connecting to Inspire Hand at {hand_ip}:{hand_port}...")
        self.hand_controller = InspireHandController(ip=hand_ip, port=hand_port)
        
        # Set default speeds and forces
        self.speeds = [500, 500, 500, 500, 500, 500]
        self.forces = [500, 500, 500, 500, 500, 500]
    
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
                        -1,  # Fixed thumb position
                       -1 
                    ]
                    
                    # Send command to the hand
                    if self.hand_controller.get_state() == HandState.IDLE:
                        self.hand_controller.set_positions(positions)
                    
                    # Display the values on the frame
                    value_text = f"Hand Values: {','.join([str(p) for p in positions])}"
                    cv2.putText(annotated_frame, value_text, (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return annotated_frame
    
    def run(self):
        """
        Run the hand teleoperation system.
        """
        # Connect to the Inspire Hand
        if not self.connect_hand():
            print("Failed to connect to the Inspire Hand. Exiting.")
            return
        
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
    hand_ip = "192.168.11.210"  # Default IP
    hand_port = 6000  # Default port
    target_hand = "left"  # Default hand to track
    
    # Check if IP and port are provided
    if len(sys.argv) > 1:
        hand_ip = sys.argv[1]
    if len(sys.argv) > 2:
        hand_port = int(sys.argv[2])
    if len(sys.argv) > 3 and sys.argv[3] in ['left', 'right', 'both']:
        target_hand = sys.argv[3]
    
    # Create and run the system
    system = HandTeleoperationSystem(hand_ip, hand_port, target_hand)
    system.run()
