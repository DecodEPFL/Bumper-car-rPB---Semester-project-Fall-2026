#!/usr/bin/env python3

import os

import torch
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from .bumpercar_sys import BumpercarSystem
from .parameters import CarParams as car_params
from .rPB_controller import PerfBoostController
from bump_msgs.msg import ControlInput, EKFStateArray

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class PBControllerNode(Node):
    def __init__(self):
        super().__init__("rpb_controller_node")

        self.n_agents = 2
        self.nx = 7 * self.n_agents
        self.nu = 2 * self.n_agents

        control_package_share_dir = get_package_share_directory("bumpercar_control")
        sim_package_share_dir = get_package_share_directory("bumpercar_sim")
        dynamics_model_path = os.path.join(
            sim_package_share_dir,
            "mlp_simulation_models",
            "model_kinematic_mlp.pth",
        )

        self.system = BumpercarSystem(
            params=car_params,
            x_init=None,
            u_init=None,
            n_agents=self.n_agents,
            model_path=dynamics_model_path,
            dt=0.04,
        ).to(device)
        self.get_logger().info(f"Loaded rPB dynamics model: {dynamics_model_path}")

        pRB_model_path = os.path.join(
            control_package_share_dir,
            "rPB_control_models",
            "rPB_best.pt",
        )
        self.declare_parameter("checkpoint_path", pRB_model_path)
        checkpoint_path = str(self.get_parameter("checkpoint_path").value)

        self.controller = PerfBoostController(
            noiseless_forward=self.system.noiseless_forward,
            input_init=self.system.x_init,
            output_init=self.system.u_init,
            dim_internal=args.get("dim_internal", 8),
            dim_nl=args.get("dim_nl", 8),
            initialization_std=args.get("cont_init_std", 0.1),
            output_amplification=1,
            ren_internal_state_init=None,
        ).to(device)
        self.controller.load_state_dict(checkpoint_path)
        
        self.system.positionPID.reset(
            batch_size=1,
            device=device,
            dtype=torch.float32,
        )

        self.xbar = torch.tensor(
            [[2.5, 3.0, -2.5, 3.0]],
            dtype=torch.float32,
            device=device,
        )

        self.v = torch.zeros((1, self.nu), dtype=torch.float32, device=device)
        self.logged_first_control = False

        self.state_sub = self.create_subscription(
            EKFStateArray,
            "/cars_states",
            self.cars_state_callback,
            10,
        )

        self.car0_pub = self.create_publisher(
            ControlInput,
            "/car0_safe_input",
            10,
        )

        self.car1_pub = self.create_publisher(
            ControlInput,
            "/car1_safe_input",
            10,
        )

    def state_to_vec(self, state_msg):
        return [
            state_msg.x,
            state_msg.y,
            state_msg.theta,
            state_msg.v_f,
            state_msg.beta_f,
            state_msg.beta_r,
            state_msg.delta,
        ]

    def cars_state_callback(self, msg):
        x_list = []
        for i in range(self.n_agents):
            x_list.extend(self.state_to_vec(msg.states[i]))

        x = torch.tensor(
            x_list,
            dtype=torch.float32,
            device=device,
        ).view(1, self.nx)

        x_controller = x.view(1, 1, self.nx)
        v_controller = self.v.view(1, 1, self.nu)
        xbar_controller = self.xbar.view(1, 1, self.nu)

        with torch.no_grad():
            dxref = self.controller(
                x_controller,
                v_controller,
                xbar_controller,
            ).view(1, self.nu)

            safe_input, self.v = self.system._physical_controller(
                x=x,
                v=self.v,
                xbar=self.xbar,
                dxref=dxref,
            )

        if not self.logged_first_control:
            self.get_logger().info(
                f"First rPB dxref={dxref.squeeze(0).detach().cpu().tolist()}, "
                f"safe_input={safe_input.squeeze(0).detach().cpu().tolist()}"
            )
            self.logged_first_control = True

        safe_input = safe_input.squeeze(0).detach().cpu()

        car0_msg = ControlInput()
        car0_msg.header = msg.header
        car0_msg.throttle = float(safe_input[0])
        car0_msg.steering = float(safe_input[1])

        car1_msg = ControlInput()
        car1_msg.header = msg.header
        car1_msg.throttle = float(safe_input[2])
        car1_msg.steering = float(safe_input[3])

        self.car0_pub.publish(car0_msg)
        self.car1_pub.publish(car1_msg)


def main(args=None):
    rclpy.init(args=args)
    node = PBControllerNode()

    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
