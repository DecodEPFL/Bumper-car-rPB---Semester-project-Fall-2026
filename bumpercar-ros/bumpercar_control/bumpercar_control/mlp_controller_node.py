import numpy as np
import rclpy
from rclpy.node import Node


import time
import os
from ament_index_python.packages import get_package_share_directory
from .mlp_controller import NNController
from std_msgs.msg import Float64MultiArray


class ControllerNode(Node):
    def __init__(self):
        super().__init__("mlp_controller")

        self.state_sub = self.create_subscription(
            Float64MultiArray,
            'state',
            self.state_callback,
            10
        )
        self.goal_sub = self.create_subscription(
            Float64MultiArray,
            'goal_state',
            self.goal_callback,
            10
        )
        self.control_pub = self.create_publisher(Float64MultiArray, 'control_input', 10)
        
        # Parameters
        
        ### CHOOSE MODEL ###
        useMlp = True
        useGRU = False

        if sum([useMlp, useGRU]) != 1:
            raise ValueError("Exactly one of usePidController, useMlp, or useGRU must be True")

        if useGRU:
            model_type = "GRU"
            model_path = "PLACEHOLDER_DOES_NOTHING"
            
        elif useMlp:
            model_type = "MLP"
            default_model_path = os.path.join(
                get_package_share_directory("bumpercar_control"),
                "onnx_models",
                "policy_weights_1000cars_preTrain.onnx",
            )
            self.declare_parameter("model_path", default_model_path)
            model_path = self.get_parameter("model_path").value

        ## state of car
        self.state = None
        self.goal_state = None

        self.controller = NNController(model_path=model_path, model_type=model_type)
                
        self.control_period = 1.0 / 25.0
        self.control_timer = self.create_timer(
            self.control_period,
            self.control_loop
        )

        self.get_logger().info("Controller node started")
        
        # self.get_logger().info(f"Loaded ONNX model from {model_path}")


    
    
    def benchmark_inference(self):
        times = []

        car_state = np.array([1.0, 2.0, 0.3, 0.5, -0.2, 0.1, 0.0], dtype=np.float32)
        car_state_final = np.array([5.0, 4, 0.7, 0.2, 0.0, 0.0, 0.0], dtype=np.float32)
        for _ in range(10):
            u = self.controller.compute_control(car_state, car_state_final, self)

        # measurement
        for _ in range(1000):
            start = time.perf_counter()
            u = self.controller.compute_control(car_state, car_state_final, self)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms

        times = np.array(times)

        print(f"avg: {times.mean()} ms")
        print(f"min: {times.min()} ms")
        print(f"max: {times.max()} ms")
        print(f"std: {times.std()} ms")
        
    def compute_and_publish(self, car_state, car_state_final):
        u = self.controller.compute_control(car_state, car_state_final, self)
        msg = Float64MultiArray()
        msg.data = u.tolist()
        self.control_pub.publish(msg)

        self.get_logger().info(
            f'State={self.state}, Goal={self.goal_state}, Control={u}'
        )
    
    def control_loop(self):
        if self.state is None or self.goal_state is None:
            return
        self.compute_and_publish(self.state, self.goal_state)    
        
    def state_callback(self, msg):
        self.state = np.array(msg.data, dtype=np.float32)
    

    def goal_callback(self, msg):
        self.goal_state = np.array(msg.data, dtype=np.float32)

    
    


def main(args=None):
    rclpy.init(args=args)
    node = ControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

    # rclpy.init(args=args)

    # node = MyNode()

    # # Run benchmark
    # node.benchmark_inference()

    # node.destroy_node()
    # rclpy.shutdown()

if __name__ == '__main__':
    main()