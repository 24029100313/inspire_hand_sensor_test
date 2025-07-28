# Inspire Hand 控制系统

这是一个用于控制Inspire灵巧手的综合软件包，支持多种控制方式和功能。该项目集成了手部姿态检测、远程操作、触觉传感器数据可视化以及ROS仿真等功能。

## 项目概述

Inspire Hand是一个多指灵巧手，具有6个自由度，可以通过多种方式进行控制：
- Modbus TCP网络通信
- 串口通信  
- 手势识别遥操作
- ROS仿真控制

## 主要功能模块

### 1. 基础控制模块

#### `inspire_hand_controller.py`
- 完整的Inspire手部控制器类
- 基于状态机的控制逻辑
- 支持自动错误检测和状态监控
- 提供标准化的API接口

**主要功能：**
- 手指角度控制
- 力控制
- 速度控制
- 状态监控和错误处理

#### `demo_modbus.py`
- Modbus TCP通信示例脚本
- 演示基本的手部控制操作
- 包含连接、设置参数、读取信息等基础功能

**使用方法：**
```bash
pip3 install pymodbus==2.5.3
python3 demo_modbus.py
```

#### `library.py`
- 串口通信库
- 底层数据处理函数
- 支持串口方式的手部控制

### 2. 手势识别遥操作模块

#### `hand_tele.py`
- 手势识别遥操作系统主程序
- 集成MediaPipe手部检测
- 实时手势到机械手映射
- 支持触觉反馈可视化

**功能特点：**
- 实时手部姿态检测
- 手势到机械手动作的映射
- 触觉传感器数据可视化
- 多线程处理确保实时性

#### `mp_read_hand.py`
- MediaPipe手部检测模块
- 支持左手、右手或双手检测
- 提供手部关键点数据
- 保存手部姿态数据功能

**主要类和方法：**
- `HandDetector`: 手部检测器类
- 支持实时检测和数据保存
- 可配置检测目标（左手/右手/双手）

### 3. 传感器数据可视化模块

#### `read_and_vis_data.py`
- 触觉传感器数据读取和可视化
- 实时数据绘图
- 支持多个手指的传感器数据

**功能：**
- 实时读取触觉传感器数据
- 数据可视化绘图
- 传感器状态监控

#### `touch_data.py`
- 触觉数据处理模块
- 传感器数据解析和处理

### 4. ROS仿真模块

#### `inspire_hand_left/` 目录
ROS功能包，包含完整的左手仿真环境：

**文件结构：**
- `urdf/`: 手部的URDF模型文件
- `launch/`: ROS启动文件
- `meshes/`: 3D模型网格文件
- `config/`: 配置文件

**使用方法：**

1. **RVIZ可视化：**
```bash
roslaunch urdf_left display.launch
```

2. **Gazebo仿真：**
```bash
roslaunch urdf_left gazebo.launch
```

3. **程序控制：**
```bash
python3 left_hand_control.py
```

**控制方式：**
- GUI界面控制
- ROS Topic命令控制
- 脚本自动化控制

## 硬件连接和配置

### 网络连接方式（推荐）
1. 使用网线连接Inspire手与主机
2. 配置网络：
   - IP地址：192.168.11.222
   - 子网掩码：255.255.255.0
3. 测试连接：`ping 192.168.11.210`

### 串口连接方式
1. 查看串口设备：通常为`/dev/ttyUSB0`或`/dev/ttyUSB1`
2. 给USB端口添加执行权限
3. 修改代码中的端口配置

## 安装依赖

### Python依赖
```bash
pip3 install pymodbus==2.5.3
pip3 install opencv-python
pip3 install mediapipe
pip3 install matplotlib
pip3 install numpy
pip3 install pysnooper
```

### ROS依赖
```bash
sudo apt-get install ros-noetic-joint-state-publisher-gui
sudo apt-get install ros-noetic-robot-state-publisher
sudo apt-get install ros-noetic-gazebo-*
```

## 控制接口说明

### 基础控制命令
- `setpos(pos1,pos2,pos3,pos4,pos5,pos6)`: 设置驱动器位置 (范围: -1~2000)
- `setangle(angle1,angle2,angle3,angle4,angle5,angle6)`: 设置手指角度 (范围: -1~1000)
- `setpower(power1,power2,power3,power4,power5,power6)`: 设置力控阈值 (范围: 0~1000)
- `setspeed(speed1,speed2,speed3,speed4,speed5,speed6)`: 设置速度 (范围: 0~1000)

### 状态读取命令
- `get_actpos()`: 读取实际位置
- `get_actangle()`: 读取实际角度
- `get_actforce()`: 读取实际受力
- `get_error()`: 读取故障信息
- `get_status()`: 读取状态信息
- `get_temp()`: 读取温度信息
- `get_current()`: 读取电流信息

### 系统管理命令
- `set_clear_error()`: 清除错误
- `set_save_flash()`: 保存参数到Flash
- `set_force_clb()`: 校准力传感器

## 使用示例

### 1. 基础控制示例
```python
from inspire_hand_controller import InspireHandController

# 创建控制器实例
hand = InspireHandController(ip="192.168.11.210", port=6000)

# 连接手部
hand.connect()

# 设置手指角度
hand.set_angle([500, 500, 500, 500, 500, 500])

# 读取状态
status = hand.get_status()
print(f"Hand status: {status}")
```

### 2. 手势遥操作示例
```python
from hand_tele import HandTeleoperationSystem

# 创建遥操作系统
tele_system = HandTeleoperationSystem(
    hand_ip="192.168.11.210",
    target_hand="left",
    show_touch_data=True
)

# 开始遥操作
tele_system.start()
```

### 3. ROS控制示例
```bash
# 启动仿真环境
roslaunch urdf_left gazebo.launch

# 发送控制命令
rostopic pub -r 10 /gripper_controller/command trajectory_msgs/JointTrajectory "
header:
  seq: 0
  stamp: {secs: 0, nsecs: 0}
  frame_id: ''
joint_names: ['left_index_1_joint', 'left_little_1_joint', 'left_middle_1_joint', 'left_ring_1_joint', 'left_thumb_1_joint', 'left_thumb_swing_joint']
points:
  - positions: [1.2, 0.5, 0.5, 0.5, 0.5, 1.08]
    velocities: []
    accelerations: []
    effort: []
    time_from_start: {secs: 1, nsecs: 0}"
```

## 文件说明

| 文件名 | 功能说明 |
|--------|----------|
| `inspire_hand_controller.py` | 主控制器类，提供完整的手部控制API |
| `hand_tele.py` | 手势识别遥操作主程序 |
| `mp_read_hand.py` | MediaPipe手部检测模块 |
| `read_and_vis_data.py` | 触觉传感器数据可视化 |
| `demo_modbus.py` | Modbus通信示例 |
| `library.py` | 串口通信库 |
| `touch_data.py` | 触觉数据处理 |
| `inspire_hand.py` | 基础手部控制接口 |
| `async_read_and_write.py` | 异步读写操作 |
| `hand_landmarker.task` | MediaPipe手部检测模型文件 |

## 开发者信息

- **项目类型**: 机器人控制系统
- **编程语言**: Python, ROS
- **硬件平台**: Inspire灵巧手
- **通信协议**: Modbus TCP, 串口通信
- **可视化**: OpenCV, Matplotlib, RViz, Gazebo

## 注意事项

1. **版本兼容性**: 确保使用指定版本的依赖包，特别是`pymodbus==2.5.3`
2. **网络配置**: 网络连接方式需要正确配置IP地址
3. **权限设置**: 串口连接需要适当的设备权限
4. **实时性要求**: 遥操作模式对系统性能有一定要求
5. **安全操作**: 操作机械手时注意安全，避免过大的力控设置

## 故障排除

1. **连接问题**: 检查网络配置或串口权限
2. **依赖问题**: 确认所有Python包和ROS包已正确安装
3. **权限问题**: 为串口设备添加用户权限
4. **性能问题**: 确保系统资源充足，特别是实时操作模式

## 扩展开发

该项目提供了良好的模块化结构，可以轻松扩展以下功能：
- 更复杂的手势识别算法
- 多手协同控制
- 力反馈集成
- 机器学习驱动的控制策略
- 与其他机器人系统的集成

## 原始项目

本项目基于 [Lr-2002/inspire_hand](https://github.com/Lr-2002/inspire_hand) 进行了扩展和文档化。
