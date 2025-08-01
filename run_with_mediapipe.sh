#!/bin/bash

# Inspire Hand with MediaPipe Integration Demo
# Includes MediaPipe hand gesture control with fallback to demo mode

echo "🚀 Starting Inspire Hand with MediaPipe Demo"
echo "👋 Uses camera for hand gesture control"
echo "🤖 Falls back to demo mode if no hand detected"
echo "⏹️  Press Ctrl+C to stop"

# Run the MediaPipe version
/home/larry/NVIDIA_DEV/isaac_grasp_ws/IsaacLab/isaaclab.sh -p lift_cube_inspire_hand_with_mediapipe.py "$@" 