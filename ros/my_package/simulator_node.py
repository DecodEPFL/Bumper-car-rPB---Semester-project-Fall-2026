from .parameters import CarParams as carParams
from .simulator import MLPModel
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import numpy as np
import time
from .gruPolicy import PolicyGRU

class SimulatorNode(Node):
    def __init__(self):
        super().__init__('simulator_node')

        self.state_pub = self.create_publisher(Float64MultiArray, 'state', 10)
        self.goal_pub = self.create_publisher(Float64MultiArray, 'goal_state', 10)
        self.control_sub = self.create_subscription(
            Float64MultiArray,
            'control_input',
            self.control_callback,
            10
        )

        self.timer = self.create_timer(0.1, self.timer_callback)
        
        self.last_input_time = time.time()
        self.timeout = 1.0
        
        self.init_state = np.array([2.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.goal_state = np.array([5.0, 7.0, -3*np.pi/6, 0, 0.0, 0.0, 0.0])
        self.u = np.array([0,0])
        
        self.simulator = MLPModel(self.init_state, carParams)

    
    def control_callback(self, msg):
        self.last_input_time = time.time()
        self.u = np.array(msg.data)
        self.get_logger().info(f'Received control input: {self.u}')


    def timer_callback(self):
        if (time.time() - self.last_input_time) > self.timeout:
            self.u = np.array([0.0, 0.0])
            self.get_logger().info(f'Not recieved control input in 1 sec !!')
        
        self.simulator.update(self.u)
        
        state_msg = Float64MultiArray()
        state_msg.data = self.simulator.car_state.tolist()
        self.state_pub.publish(state_msg)
        
        goal_msg = Float64MultiArray()
        goal_msg.data = self.goal_state.tolist()
        self.goal_pub.publish(goal_msg)

        self.get_logger().info(
            f'Published state={self.simulator.car_state}, goal={self.goal_state}'
        )
        
def main(args=None):
    rclpy.init(args=args)
    node = SimulatorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()