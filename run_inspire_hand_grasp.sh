#!/bin/bash

echo "🤖 Inspire Hand 抓取任务 - MediaPipe控制 + 1061传感器实时监控"
echo "📁 项目目录: $(pwd)"

# Isaac Lab路径
ISAACLAB_PATH="/home/larry/NVIDIA_DEV/isaac_grasp_ws/IsaacLab"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🎯 Isaac Lab: $ISAACLAB_PATH"
echo "📦 抓取任务：Inspire Hand抓取立方体"
echo "🔍 传感器监控：1061个传感器pad实时检测"
echo "🎮 控制方式：MediaPipe手势识别控制"
echo ""

# 检查Isaac Lab路径
if [ ! -f "$ISAACLAB_PATH/isaaclab.sh" ]; then
    echo "❌ 错误: Isaac Lab路径未找到: $ISAACLAB_PATH"
    exit 1
fi

# 检查USD文件
USD_FILE="/home/larry/NVIDIA_DEV/isaac_grasp_ws/cabinet_sensor_project/inspire_hand_with_sensors/usd/inspire_hand_processed_with_pads.usd"
if [ ! -f "$USD_FILE" ]; then
    echo "❌ 错误：找不到USD文件"
    exit 1
fi

echo "✅ 找到Isaac Lab: $ISAACLAB_PATH"
echo "✅ 找到USD文件: $USD_FILE"
echo ""

# 运行程序
echo "🚀 启动Inspire Hand + MediaPipe抓取任务..."
"$ISAACLAB_PATH/isaaclab.sh" -p "$SCRIPT_DIR/lift_cube_inspire_hand.py" --num_envs 1 --device cuda "$@"
