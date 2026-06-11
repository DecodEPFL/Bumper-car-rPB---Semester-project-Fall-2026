#!/usr/bin/env python3
import threading

import matplotlib.pyplot as plt
import numpy as np
import rclpy
from matplotlib.patches import Circle
from rclpy.node import Node

from bump_msgs.msg import EKFStateArray


OBSTACLE_RADIUS = 0.625


class VisualizationNode(Node):
    def __init__(self):
        super().__init__('visualization_node')

        self.declare_parameter('n_agents', 2)
        self.declare_parameter('x_min', -3.5)
        self.declare_parameter('x_max', 3.5)
        self.declare_parameter('y_min', -4.0)
        self.declare_parameter('y_max', 6.0)
        self.declare_parameter('obstacle_centers', [])

        self.n_agents = int(self.get_parameter('n_agents').value)
        self.x_min = float(self.get_parameter('x_min').value)
        self.x_max = float(self.get_parameter('x_max').value)
        self.y_min = float(self.get_parameter('y_min').value)
        self.y_max = float(self.get_parameter('y_max').value)
        self.obstacle_centers = self._parse_obstacle_centers()

        self.state_lock = threading.Lock()
        self.latest_states = None
        self.history = [[] for _ in range(self.n_agents)]
        self.colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
        self.car_patches = [None for _ in range(self.n_agents)]

        self.state_sub = self.create_subscription(
            EKFStateArray,
            '/cars_states',
            self.state_callback,
            10,
        )

        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(7, 7))
        self.fig.canvas.manager.set_window_title('Bumpercar Arena')
        self.timer = self.create_timer(1.0 / 25.0, self.plot_callback)

        self.get_logger().info('Visualization node started, subscribed to /cars_states')

    def _parse_obstacle_centers(self):
        values = list(self.get_parameter('obstacle_centers').value)
        if len(values) % 2 != 0:
            self.get_logger().warn('obstacle_centers must contain x/y pairs; ignoring last value')
            values = values[:-1]
        return [(float(values[i]), float(values[i + 1])) for i in range(0, len(values), 2)]

    def state_callback(self, msg):
        if len(msg.states) < self.n_agents:
            self.get_logger().warn('Received fewer car states than expected.')
            return

        states = np.array([
            [state.x, state.y, state.theta, state.v_f, state.beta_f, state.beta_r, state.delta]
            for state in msg.states[:self.n_agents]
        ], dtype=np.float32)

        with self.state_lock:
            self.latest_states = states
            for car_index, state in enumerate(states):
                self.history[car_index].append((float(state[0]), float(state[1])))

    def draw_car(self, x, y, theta, color):
        length = 0.55
        width = 0.32
        dx = np.array([length / 2, length / 2, -length / 2, -length / 2])
        dy = np.array([width / 2, -width / 2, -width / 2, width / 2])
        c = np.cos(theta)
        s = np.sin(theta)
        px = x + c * dx - s * dy
        py = y + s * dx + c * dy
        return self.ax.fill(px, py, color=color, alpha=0.75, edgecolor='black', linewidth=1.0)[0]

    def draw_obstacles(self):
        for center in self.obstacle_centers:
            circle = Circle(
                xy=center,
                radius=OBSTACLE_RADIUS,
                facecolor='0.35',
                edgecolor='black',
                alpha=0.25,
                linewidth=1.0,
                zorder=0,
            )
            self.ax.add_patch(circle)

    def plot_callback(self):
        with self.state_lock:
            if self.latest_states is None:
                return
            states = self.latest_states.copy()
            histories = [path.copy() for path in self.history]

        self.ax.clear()
        self.ax.set_title('Bumpercar Arena')
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_xlim(self.x_min, self.x_max)
        self.ax.set_ylim(self.y_min, self.y_max)
        self.ax.grid(True, alpha=0.3)
        self.draw_obstacles()

        for car_index, state in enumerate(states):
            color = self.colors[car_index % len(self.colors)]
            history = histories[car_index]
            if history:
                path = np.array(history, dtype=np.float32)
                self.ax.plot(path[:, 0], path[:, 1], color=color, linewidth=2)
                self.ax.plot(path[0, 0], path[0, 1], marker='o', markersize=7, color=color, fillstyle='none')

            self.draw_car(float(state[0]), float(state[1]), float(state[2]), color)

        if states.shape[0] >= 2:
            distance = np.linalg.norm(states[0, 0:2] - states[1, 0:2])
            self.ax.text(0.02, 0.97, f'distance {distance:.2f} m', transform=self.ax.transAxes, va='top')

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


def main(args=None):
    rclpy.init(args=args)
    node = VisualizationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        plt.close('all')


if __name__ == '__main__':
    main()
