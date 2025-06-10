import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    # 定义包名称
    package_name = 'urdf_left_with_force_sensor'
    
    # 启动 glove_hand_control 节点
    glove_hand_control = Node(
        package='urdf_left_with_force_sensor',
        executable='glove_hand_control.py',
        name='glove_hand_control',
        output='screen',
    )
    
    # 启动 hand_modbus_control_node 节点
    hand_modbus_control_node = Node(
        package='inspire_hand_modbus_ros2',  
        executable='hand_modbus_control_node',  
        name='hand_modbus_control',
        output='screen',
    )

    left_controller_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory(package_name), 'launch', 'left_controller.launch.py'
        )])
    )

    return LaunchDescription([
        glove_hand_control,
        hand_modbus_control_node,  
        left_controller_launch,
    ])



