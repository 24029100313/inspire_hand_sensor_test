#!/bin/bash

# Inspire Hand State Machine Cube Grasping Demo
# Pure state machine implementation - no MediaPipe required

echo "🚀 Starting Inspire Hand State Machine Demo"
echo "📋 States: APPROACH → GRASP → LIFT → HOLD → RELEASE → RESET"
echo "⏹️  Press Ctrl+C to stop"

# Run the state machine version
/home/larry/NVIDIA_DEV/isaac_grasp_ws/IsaacLab/isaaclab.sh -p lift_cube_inspire_hand_state_machine.py "$@" 