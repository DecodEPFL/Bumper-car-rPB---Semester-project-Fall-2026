#!/usr/bin/env python3
import threading
import time

import numpy as np
import torch
import rclpy
from rclpy.node import Node

from bump_msgs.msg import ControlInput, EKFState, EKFStateArray

from .arena_experiment_params import getCarInitParams
from .parameters import CarParams
from .simulator import MLPModel


OBSTACLE_RADIUS = 0.625


class ArenaSimulatorNode(Node):
    def __init__(self):
        super().__init__("arena_simulator")

        self.declare_parameter("rate_hz", 25.0)
        self.declare_parameter("input_timeout", 0.25)
        self.declare_parameter("draw_arena", True)

        self.x0, self.x_final, self.obstacle_centers = self._load_arena_experiment()

        self.rate_hz = float(self.get_parameter("rate_hz").value)
        self.dt = 1.0 / self.rate_hz
        self.input_timeout = float(self.get_parameter("input_timeout").value)

        car0_initial_state = self.x0[0]
        car1_initial_state = self.x0[1]

        self.cars = [
            MLPModel(car0_initial_state, CarParams),
            MLPModel(car1_initial_state, CarParams),
        ]
        self.inputs = [np.zeros(2, dtype=np.float32), np.zeros(2, dtype=np.float32)]
        self.last_input_time = [0.0, 0.0]
        self.draw_arena = bool(self.get_parameter("draw_arena").value)
        self.state_lock = threading.Lock()
        self.history = [[tuple(state[:2])] for state in self.x0]
        self.colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

        self.state_pub = self.create_publisher(EKFStateArray, "/cars_states", 10)
        self.car0_sub = self.create_subscription(
            ControlInput,
            "/car0_safe_input",
            lambda msg: self.input_callback(msg, 0),
            10,
        )
        self.car1_sub = self.create_subscription(
            ControlInput,
            "/car1_safe_input",
            lambda msg: self.input_callback(msg, 1),
            10,
        )

        if self.draw_arena:
            try:
                import matplotlib.pyplot as plt
                from matplotlib.patches import Circle
            except ImportError as exc:
                raise RuntimeError(
                    "draw_arena is enabled, but matplotlib is not installed in this Python environment"
                ) from exc

            self.plt = plt
            self.circle_patch = Circle
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(7, 7))
            self.fig.canvas.manager.set_window_title("Arena Simulator")
            self.plot_timer = self.create_timer(self.dt, self.plot_callback)

        self.timer = self.create_timer(self.dt, self.timer_callback)
        self.get_logger().info(
            f"Arena simulator started at {self.rate_hz:.1f} Hz, publishing /cars_states"
        )


    def _load_arena_experiment(self):
        x0, x_final, obstacle_centers, _, _, _ = getCarInitParams(torch.device("cpu"))
        x0 = x0.detach().cpu().numpy().astype(np.float32).reshape(-1, 7)
        x_final = x_final.detach().cpu().numpy().astype(np.float32).reshape(-1, 7)
        obstacle_centers = [
            tuple(center.detach().cpu().numpy().reshape(-1)[:2].astype(float))
            for center in obstacle_centers
        ]
        return x0, x_final, obstacle_centers

    def _state_parameter(self, name):
        value = list(self.get_parameter(name).value)
        if len(value) != 7:
            raise ValueError(f"{name} must contain 7 values: x, y, theta, v_f, beta_f, beta_r, delta")
        return np.array(value, dtype=np.float32)

    def input_callback(self, msg, car_index):
        self.inputs[car_index] = np.array([msg.throttle, msg.steering], dtype=np.float32)
        self.last_input_time[car_index] = time.monotonic()

    def timer_callback(self):
        now = time.monotonic()
        for car_index, car in enumerate(self.cars):
            car_input = self.inputs[car_index]
            if now - self.last_input_time[car_index] > self.input_timeout:
                car_input = np.zeros(2, dtype=np.float32)
            car.update(car_input, dt=self.dt)

        with self.state_lock:
            for car_index, car in enumerate(self.cars):
                self.history[car_index].append(tuple(car.car_state[:2]))

        self.publish_states()

    def publish_states(self):
        msg = EKFStateArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "arena"
        msg.states = [self.to_state_msg(car.car_state, msg.header) for car in self.cars]
        self.state_pub.publish(msg)

    def to_state_msg(self, state, header):
        state_msg = EKFState()
        state_msg.header = header
        state_msg.x = float(state[0])
        state_msg.y = float(state[1])
        state_msg.theta = float(state[2])
        state_msg.v_f = float(state[3])
        state_msg.beta_f = float(state[4])
        state_msg.beta_r = float(state[5])
        state_msg.delta = float(state[6])
        return state_msg


    def draw_car(self, x, y, theta, color):
        length = 0.55
        width = 0.32
        dx = np.array([length / 2, length / 2, -length / 2, -length / 2])
        dy = np.array([width / 2, -width / 2, -width / 2, width / 2])
        c = np.cos(theta)
        s = np.sin(theta)
        px = x + c * dx - s * dy
        py = y + s * dx + c * dy
        self.ax.fill(px, py, color=color, alpha=0.75, edgecolor="black", linewidth=1.0)

    def draw_obstacles(self):
        for center in self.obstacle_centers:
            self.ax.add_patch(
                self.circle_patch(
                    xy=center,
                    radius=OBSTACLE_RADIUS,
                    facecolor="0.35",
                    edgecolor="black",
                    alpha=0.25,
                    linewidth=1.0,
                    zorder=0,
                )
            )

    def plot_callback(self):
        with self.state_lock:
            states = np.array([car.car_state for car in self.cars], dtype=np.float32)
            histories = [path.copy() for path in self.history]

        self.ax.clear()
        self.ax.set_title("Arena Simulator")
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.set_xlim(-3.5, 3.5)
        self.ax.set_ylim(-4.0, 6.0)
        self.ax.grid(True, alpha=0.3)
        self.draw_obstacles()

        for car_index, state in enumerate(states):
            color = self.colors[car_index % len(self.colors)]
            start = self.x0[car_index]
            goal = self.x_final[car_index]
            history = histories[car_index]

            self.ax.plot(start[0], start[1], marker="o", markersize=8, color=color, fillstyle="none")
            self.ax.plot(goal[0], goal[1], marker="*", markersize=12, color=color)
            self.ax.plot([start[0], goal[0]], [start[1], goal[1]], linestyle="--", color=color, alpha=0.25)

            if history:
                path = np.array(history, dtype=np.float32)
                self.ax.plot(path[:, 0], path[:, 1], color=color, linewidth=2)

            self.draw_car(float(state[0]), float(state[1]), float(state[2]), color)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


def main(args=None):
    rclpy.init(args=args)
    node = ArenaSimulatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        if hasattr(node, "plt"):
            node.plt.close("all")


if __name__ == "__main__":
    main()
