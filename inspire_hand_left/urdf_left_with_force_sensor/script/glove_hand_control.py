#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState       #仿真
from service_interfaces.srv import Setangle  # 硬件驱动 导入设置角度服务

class GloveHandController(Node):
    def __init__(self):
        super().__init__('set_angle_client')
        
        # 初始化各关节角度
        self.current_angles = [1000] * 6  # 左手6个关节数据

        # 左手对应的关节名称和索引
        self.joint_index_map = {
            'left_thumb_1_joint': 0,
            'left_thumb_2_joint': 1,
            'left_index_1_joint': 2,
            'left_middle_1_joint': 3,
            'left_ring_1_joint': 4,
            'left_little_1_joint': 5,
        }

        # 声明每个关节的最大弧度值
        self.declare_parameter('inspire_hand_1/max_radians/left_thumb_1_joint', 1.1641)
        self.declare_parameter('inspire_hand_1/max_radians/left_thumb_2_joint', 0.5864)
        self.declare_parameter('inspire_hand_1/max_radians/left_index_1_joint', 1.4381)
        self.declare_parameter('inspire_hand_1/max_radians/left_middle_1_joint', 1.4381)
        self.declare_parameter('inspire_hand_1/max_radians/left_ring_1_joint', 1.4381)
        self.declare_parameter('inspire_hand_1/max_radians/left_little_1_joint', 1.4381)

        self.set_angle_service_client = self.create_client(Setangle, '/Setangle')
        while not self.set_angle_service_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().info('Service not available, waiting...')

        # 订阅 /joint_states 话题
        self.joint_states_subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_states_callback,
            10
        )

        self.timer = self.create_timer(0.02, self.call_set_angle_services)  # 50Hz
        self.last_joint_states = None  

    def joint_states_callback(self, msg):
        # 打印接收到的关节状态
        self.get_logger().info(f"Received joint states: {msg.position}")
        self.last_joint_states = msg  # 保存最新的关节状态

        # 提取关节位置并进行线性转换
        for i in range(len(msg.name)):
            joint_name = msg.name[i]
            if joint_name in self.joint_index_map:
                index = self.joint_index_map[joint_name]
                # 获取特定关节的最大弧度值
                max_radian = self.get_parameter(f'inspire_hand_1/max_radians/{joint_name}').get_parameter_value().double_value
                
                # 将弧度转换为0-1000
                self.current_angles[index] = 1000 - max(0, min(1000, int((msg.position[i] / max_radian) * 1000)))

                self.get_logger().info(f"Joint '{joint_name}' position: {msg.position[i]} -> Converted angle: {self.current_angles[index]}")

    def call_set_angle_services(self):
        if self.last_joint_states is None:
            return  # 未接收到关节状态，不调用服务

        request = Setangle.Request()
        request.angle0 = self.current_angles[self.joint_index_map['left_little_1_joint']]
        request.angle1 = self.current_angles[self.joint_index_map['left_ring_1_joint']]
        request.angle2 = self.current_angles[self.joint_index_map['left_middle_1_joint']]
        request.angle3 = self.current_angles[self.joint_index_map['left_index_1_joint']]
        request.angle4 = self.current_angles[self.joint_index_map['left_thumb_2_joint']]
        request.angle5 = self.current_angles[self.joint_index_map['left_thumb_1_joint']]
        request.id = 1
        request.status = 'set_angle'

        # 调用服务
        future = self.set_angle_service_client.call_async(request)
        future.add_done_callback(self.handle_service_response)

    def handle_service_response(self, future):
        try:
            response = future.result()
            if response is not None:
                self.get_logger().info(f"Service call successful: {response.angle_accepted}")
            else:
                self.get_logger().error("Service call failed: Response is None.")
        except Exception as e:
            self.get_logger().error(f"Service call failed: {e}")

def main(args=None):
    rclpy.init(args=args)
    glove_hand_controller = GloveHandController()
    
    try:
        rclpy.spin(glove_hand_controller)
    except KeyboardInterrupt:
        pass
    finally:
        glove_hand_controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

