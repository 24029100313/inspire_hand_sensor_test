import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    package_name = 'urdf_left_with_force_sensor'

    # 机器人描述启动文件
    left = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory(package_name), 'launch', 'left.launch.py'
        )),
        launch_arguments={'use_sim_time': 'true'}.items()
    )

    # gazebo 启动文件
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('gazebo_ros'), 'launch', 'gazebo.launch.py'
        )),
    )

    # 生成 spawn 实体节点
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-topic', 'robot_description', '-entity', 'left'],
        output='screen'
    )

    # 关节状态和轨迹控制器 spawner 列表
    controller_names = [
        "joint_state",
        "joint_trajectory_controller",
        # 六维力传感器控制器
        "palm_force_sensor_controller",
        "thumb_force_sensor_1_controller",
        "thumb_force_sensor_2_controller",
        "thumb_force_sensor_3_controller",
        "thumb_force_sensor_4_controller",
        "index_force_sensor_1_controller",
        "index_force_sensor_2_controller",
        "index_force_sensor_3_controller",
        "middle_force_sensor_1_controller",
        "middle_force_sensor_2_controller",
        "middle_force_sensor_3_controller",
        "ring_force_sensor_1_controller",
        "ring_force_sensor_2_controller",
        "ring_force_sensor_3_controller",
        "little_force_sensor_1_controller",
        "little_force_sensor_2_controller",
        "little_force_sensor_3_controller",
    ]

    # 列表推导式生成所有 spawner Node
    spawner_nodes = [
        Node(
            package='controller_manager',
            executable='spawner',
            arguments=[name],
            output='screen'
        )
        for name in controller_names
    ]

    return LaunchDescription([
        left,
        gazebo,
        spawn_entity,
        *spawner_nodes  
    ])

