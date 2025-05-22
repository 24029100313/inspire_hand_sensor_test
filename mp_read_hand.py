import cv2
import mediapipe as mp
import numpy as np
import time
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

# Global variables
LAST_SAVE_TIME = 0
SAVE_INTERVAL = 1.0  # Save every 1 second
OUTPUT_DIR = 'hand_landmarks'

# Create output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def save_landmarks(landmarks, image_width, image_height, frame_timestamp):
    """
    Save hand landmarks to a file.
    
    Args:
        landmarks: MediaPipe hand landmarks
        image_width: Width of the input image
        image_height: Height of the input image
        frame_timestamp: Timestamp of the current frame
    """
    # Create a timestamped filename
    filename = os.path.join(OUTPUT_DIR, f'hand_landmarks_{int(frame_timestamp*1000)}.txt')
    
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

def draw_landmarks_on_image(rgb_image, detection_result):
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

def print_result(result, output_image, timestamp_ms):
    """
    Print the hand landmarker detection result and save landmarks if needed.
    
    Args:
        result: The detection result from MediaPipe HandLandmarker
        output_image: The output image with annotations
        timestamp_ms: The timestamp in milliseconds
    """
    global LAST_SAVE_TIME
    
    if result.hand_landmarks:
        # Print the hand landmarks for debugging
        for hand_idx, hand_landmarks in enumerate(result.hand_landmarks):
            print(f"Hand {hand_idx} detected at timestamp {timestamp_ms} ms")
            
            # Get handedness (left or right hand)
            if result.handedness and len(result.handedness) > hand_idx:
                handedness = result.handedness[hand_idx][0].category_name
                confidence = result.handedness[hand_idx][0].score
                print(f"  Handedness: {handedness} (confidence: {confidence:.2f})")
            
            # Only save landmarks at specified intervals
            current_time = time.time()
            if current_time - LAST_SAVE_TIME >= SAVE_INTERVAL:
                save_landmarks(
                    hand_landmarks, 
                    output_image.shape[1], 
                    output_image.shape[0], 
                    current_time
                )
                LAST_SAVE_TIME = current_time
    else:
        print(f"No hand detected at timestamp {timestamp_ms} ms")

def run_hand_landmarker():
    """
    Main function to run the hand landmarker on camera input.
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
    
    # Initialize MediaPipe Hand Landmarker
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,  # Detect up to 2 hands
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Download the model if not available
    if not os.path.exists('hand_landmarker.task'):
        print("Downloading hand landmarker model...")
        import urllib.request
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        urllib.request.urlretrieve(url, 'hand_landmarker.task')
        print("Model downloaded successfully.")
    
    # Create the hand landmarker
    with vision.HandLandmarker.create_from_options(options) as landmarker:
        # Continuously capture frames from the camera
        while cap.isOpened():
            # Read a frame from the camera
            success, frame = cap.read()
            if not success:
                print("Error: Could not read frame from camera.")
                break
            
            # Convert the frame to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Create a MediaPipe Image object
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Get the timestamp of the current frame
            timestamp_ms = int(time.time() * 1000)
            
            # Detect hand landmarks
            detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)
            
            # Process the detection result
            annotated_frame = draw_landmarks_on_image(rgb_frame, detection_result)
            
            # Print the result and save landmarks if needed
            print_result(detection_result, annotated_frame, timestamp_ms)
            
            # Convert back to BGR for OpenCV display
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
            
            # Display the frame
            cv2.imshow('Hand Landmarker', annotated_frame)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_hand_landmarker()
