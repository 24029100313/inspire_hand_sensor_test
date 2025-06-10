import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
import xacro

def generate_launch_description():
    # 定义包名称
    package_name = 'urdf_left_with_force_sensor'

    # 获取 xacro 文件和控制器配置文件路径
    xacro_file = os.path.join(get_package_share_directory(package_name), 'urdf', 'urdf_left_with_force_sensor_1.xacro')
    controller_config = os.path.join(get_package_share_directory(package_name), 'config', 'left_controller_rviz.yaml')
    
    if not os.path.isfile(controller_config):
        raise FileNotFoundError(f"Controller config file not found: {controller_config}")
    
    # 将 xacro 文件转换为 URDF
    robot_description_content = xacro.process_file(xacro_file).toxml()

    # 定义参数
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')

    # 包含自定义机器人描述的启动文件
    left = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(get_package_share_directory(package_name), 'launch', 'rviz.launch.py')),
        launch_arguments={'use_sim_time': use_sim_time}.items()
    )

    # 启动 ros2_control_node，加载机器人描述和控制器配置
    controller_manager = Node(
        package='controller_manager',
        executable='ros2_control_node',
        name='controller_manager',
        output='screen',
        parameters=[
            {'robot_description': robot_description_content},
            controller_config  
        ]
    )
    # 启动关节状态广播器及各类传感器控制器
    spawners = [
        "joint_state_broadcaster",
        "joint_trajectory_controller",
    ]
    # 列表推导式创建 Node对象
    spawner_nodes = [
        Node(
            package='controller_manager',
            executable='spawner',
            arguments=[spawner],
            output='screen'
        ) for spawner in spawners
    ]

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false', description='Use simulation (Gazebo) clock if true'),
        left,
        controller_manager,
    ] + spawner_nodes) 
