#!/usr/bin/env python3
import threading

import numpy as np
import matplotlib.pyplot as plt

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


class VisualizationNode(Node):
    def __init__(self):
        super().__init__('visualization_node')

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

        self.state_lock = threading.Lock()

        self.latest_state = None
        self.latest_goal_state = None

        self.history_x = []
        self.history_y = []

        self.arrow_length = 0.5

        # Fixed-size map window
        self.xmin = 0.0
        self.xmax = 10.0
        self.ymin = 0.0
        self.ymax = 10.0
        self.window_width = self.xmax - self.xmin
        self.window_height = self.ymax - self.ymin
        self.edge_margin = 1.0

        self.get_logger().info('Visualization node started, subscribed to /state and /goal_state')

        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.fig.canvas.manager.set_window_title('Vehicle State Visualization')

        self.timer = self.create_timer(0.1, self.plot_callback)

    def state_callback(self, msg):
        state = np.array(msg.data, dtype=np.float32)

        if state.shape[0] < 3:
            self.get_logger().warn(f'Received state with too few elements: shape={state.shape}')
            return

        with self.state_lock:
            self.latest_state = state.copy()
            self.history_x.append(float(state[0]))
            self.history_y.append(float(state[1]))

    def goal_callback(self, msg):
        goal_state = np.array(msg.data, dtype=np.float32)

        if goal_state.shape[0] < 3:
            self.get_logger().warn(f'Received goal state with too few elements: shape={goal_state.shape}')
            return

        with self.state_lock:
            self.latest_goal_state = goal_state.copy()

            # Clear previous trajectory on new goal
            self.history_x = []
            self.history_y = []

            if self.latest_state is not None:
                self.history_x.append(float(self.latest_state[0]))
                self.history_y.append(float(self.latest_state[1]))

        self.get_logger().info(
            f'Received new goal state: x={goal_state[0]:.3f}, y={goal_state[1]:.3f}, theta={goal_state[2]:.3f}'
        )

    def maybe_shift_window(self, points_x, points_y):
        if not points_x or not points_y:
            return

        px_min = min(points_x)
        px_max = max(points_x)
        py_min = min(points_y)
        py_max = max(points_y)

        shift_x = 0.0
        shift_y = 0.0

        if px_min < self.xmin + self.edge_margin:
            shift_x = px_min - (self.xmin + self.edge_margin)
        elif px_max > self.xmax - self.edge_margin:
            shift_x = px_max - (self.xmax - self.edge_margin)

        if py_min < self.ymin + self.edge_margin:
            shift_y = py_min - (self.ymin + self.edge_margin)
        elif py_max > self.ymax - self.edge_margin:
            shift_y = py_max - (self.ymax - self.edge_margin)

        self.xmin += shift_x
        self.xmax += shift_x
        self.ymin += shift_y
        self.ymax += shift_y

    def plot_callback(self):
        with self.state_lock:
            if self.latest_state is None:
                return

            x = float(self.latest_state[0])
            y = float(self.latest_state[1])
            theta = float(self.latest_state[2])

            hx = self.history_x.copy()
            hy = self.history_y.copy()

            goal_state = None if self.latest_goal_state is None else self.latest_goal_state.copy()

        dx = self.arrow_length * np.cos(theta)
        dy = self.arrow_length * np.sin(theta)

        all_x = list(hx) + [x]
        all_y = list(hy) + [y]

        if goal_state is not None:
            gx = float(goal_state[0])
            gy = float(goal_state[1])
            all_x.append(gx)
            all_y.append(gy)

        self.maybe_shift_window(all_x, all_y)

        self.ax.clear()

        # Draw trajectory line
        if len(hx) > 1:
            self.ax.plot(hx, hy, label='trajectory')

        # Draw current position
        self.ax.plot(x, y, marker='o', label='current state')

        # Draw current heading
        self.ax.arrow(
            x, y, dx, dy,
            head_width=0.15,
            head_length=0.2,
            length_includes_head=True
        )

        # Draw goal state
        if goal_state is not None:
            gx = float(goal_state[0])
            gy = float(goal_state[1])
            gtheta = float(goal_state[2])

            gdx = self.arrow_length * np.cos(gtheta)
            gdy = self.arrow_length * np.sin(gtheta)

            self.ax.plot(gx, gy, marker='x', markersize=10, label='goal state')
            self.ax.arrow(
                gx, gy, gdx, gdy,
                head_width=0.15,
                head_length=0.2,
                length_includes_head=True
            )

        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_title('Vehicle Position and Orientation')
        self.ax.set_xlim(self.xmin, self.xmax)
        self.ax.set_ylim(self.ymin, self.ymax)
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.grid(True)
        self.ax.legend()

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