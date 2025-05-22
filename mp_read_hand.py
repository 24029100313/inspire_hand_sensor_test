import cv2
import mediapipe as mp
import numpy as np
import time
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

# Create output directory if it doesn't exist
OUTPUT_DIR = 'hand_landmarks'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

class HandDetector:
    """Class for detecting and tracking hand landmarks using MediaPipe."""
    
    def __init__(self, target_hand='both'):
        """
        Initialize the HandDetector.
        
        Args:
            target_hand (str): The hand to detect - 'left', 'right', or 'both'
        """
        self.target_hand = target_hand.lower()
        if self.target_hand not in ['left', 'right', 'both']:
            raise ValueError("target_hand must be 'left', 'right', or 'both'")
            
        self.last_save_time = 0
        self.save_interval = 1.0  # Save every 1 second
        
        # Initialize MediaPipe Hand Landmarker
        self._initialize_landmarker()
        
    def _initialize_landmarker(self):
        """
        Initialize the MediaPipe Hand Landmarker.
        """
        # Initialize MediaPipe Hand Landmarker
        base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,  # Detect up to 2 hands
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            running_mode=vision.RunningMode.VIDEO
        )
        
        # Download the model if not available
        if not os.path.exists('hand_landmarker.task'):
            print("Downloading hand landmarker model...")
            import urllib.request
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            urllib.request.urlretrieve(url, 'hand_landmarker.task')
            print("Model downloaded successfully.")
        
        # Create the hand landmarker
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        
    def detect(self, frame):
        """
        Detect hand landmarks in a frame and return landmarks for the specified hand.
        
        Args:
            frame: Input frame from camera or video
            
        Returns:
            landmarks: Hand landmarks for the specified hand (left, right, or both)
            annotated_frame: Frame with landmarks drawn on it
        """
        # Convert the frame to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create a MediaPipe Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Get the timestamp of the current frame
        timestamp_ms = int(time.time() * 1000)
        
        # Detect hand landmarks
        detection_result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        
        # Filter results based on target hand
        filtered_landmarks = []
        filtered_handedness = []
        
        if detection_result.hand_landmarks and detection_result.handedness:
            for i, handedness in enumerate(detection_result.handedness):
                hand_type = handedness[0].category_name.lower()
                
                # Check if this hand matches our target
                if self.target_hand == 'both' or hand_type == self.target_hand:
                    filtered_landmarks.append(detection_result.hand_landmarks[i])
                    filtered_handedness.append(detection_result.handedness[i])
        
        # Create a filtered detection result
        filtered_result = type('FilteredDetectionResult', (), {})()
        filtered_result.hand_landmarks = filtered_landmarks
        filtered_result.handedness = filtered_handedness
        
        # Draw landmarks on the image
        annotated_frame = self._draw_landmarks_on_image(rgb_frame, filtered_result)
        
        # Convert back to BGR for OpenCV display
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
        
        return filtered_landmarks, annotated_frame
        
    def _draw_landmarks_on_image(self, rgb_image, detection_result):
        """
        Draw hand landmarks on the image.
        
        Args:
            rgb_image: The input RGB image
            detection_result: MediaPipe HandLandmarker detection result
            
        Returns:
            Image with hand landmarks drawn on it
        """
        hand_landmarks_list = detection_result.hand_landmarks
        handedness_list = detection_result.handedness
        annotated_image = np.copy(rgb_image)
        
        # Loop through the detected hands to visualize
        for idx, (hand_landmarks, handedness) in enumerate(zip(hand_landmarks_list, handedness_list)):
            # Draw the hand landmarks
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) 
                for landmark in hand_landmarks
            ])
            
            # Draw hand connections
            mp.solutions.drawing_utils.draw_landmarks(
                annotated_image,
                hand_landmarks_proto,
                mp.solutions.hands.HAND_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                mp.solutions.drawing_styles.get_default_hand_connections_style()
            )
            
            # Get the handedness information (left or right hand)
            handedness_label = handedness[0].category_name
            handedness_score = handedness[0].score
            
            # Calculate text position
            text_x = int(hand_landmarks[0].x * annotated_image.shape[1])
            text_y = int(hand_landmarks[0].y * annotated_image.shape[0] - 10)
            
            # Show handedness on the image
            cv2.putText(
                annotated_image, f"{handedness_label} ({handedness_score:.2f})",
                (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )
        
        return annotated_image
        
    def save_landmarks(self, landmarks, image_width, image_height):
        """
        Save hand landmarks to a file.
        
        Args:
            landmarks: MediaPipe hand landmarks
            image_width: Width of the input image
            image_height: Height of the input image
        """
        # Only save at specified intervals
        current_time = time.time()
        if current_time - self.last_save_time < self.save_interval:
            return
            
        self.last_save_time = current_time
        
        # Create a timestamped filename
        filename = os.path.join(OUTPUT_DIR, f'hand_landmarks_{int(current_time*1000)}.txt')
        
        # Open file for writing
        with open(filename, 'w') as f:
            # For each landmark in the hand
            for idx, landmark in enumerate(landmarks):
                # Convert normalized coordinates to pixel coordinates
                x = landmark.x * image_width
                y = landmark.y * image_height
                z = landmark.z * image_width  # Z uses width as scale
                
                # Write to file: index, x, y, z
                f.write(f'{idx},{x},{y},{z}\n')
        
        print(f"Saved landmarks to {filename}")

def run_hand_detector_demo(target_hand='both'):
    """
    Main function to run the hand detector on camera input.
    
    Args:
        target_hand (str): The hand to detect - 'left', 'right', or 'both'
    """
    # Initialize camera
    cap = cv2.VideoCapture(0)  # Use camera index 0 (default camera)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Create the hand detector
    detector = HandDetector(target_hand=target_hand)
    
    print(f"Hand detector initialized. Tracking {target_hand} hand(s).")
    
    # Continuously capture frames from the camera
    while cap.isOpened():
        # Read a frame from the camera
        success, frame = cap.read()
        if not success:
            print("Error: Could not read frame from camera.")
            break
        
        # Detect hand landmarks
        landmarks, annotated_frame = detector.detect(frame)
        
        # Print detection information
        if landmarks:
            for i, landmark_set in enumerate(landmarks):
                print(f"Hand {i} detected")
                
                # Save landmarks if needed
                detector.save_landmarks(
                    landmark_set,
                    frame.shape[1],
                    frame.shape[0]
                )
        else:
            print("No matching hand detected")
        
        # Display the frame
        cv2.imshow(f'Hand Detector - {target_hand.capitalize()} Hand', annotated_frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys
    
    # Check if a command line argument was provided for target hand
    if len(sys.argv) > 1 and sys.argv[1] in ['left', 'right', 'both']:
        target_hand = sys.argv[1]
    else:
        target_hand = 'left'
        
    run_hand_detector_demo(target_hand)
