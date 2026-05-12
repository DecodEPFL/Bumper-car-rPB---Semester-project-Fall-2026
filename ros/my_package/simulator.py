#!/usr/bin/env python3
import numpy as np
import onnxruntime as ort
from .utils import normalize_angle


class MLPModel:
    def __init__(self,
                 initial_state,
                 params,
                 model_path="/home/ubuntubirger/ros2_ws/src/my_package/mlp_sim_onnx_models/model_kinematic_mlp.onnx",
                 input_dim=2,
                 state_dim=4,
                 output_dim=3,
                 x_scaling=[1.9, 0.6, 0.1, 2.012]):

        # Internal variables
        self.prev_car_state = np.array(initial_state, dtype=np.float32)
        self.car_state = np.array(initial_state, dtype=np.float32)
        self.par = params

        # Scaling
        self.x_scaling = np.array(x_scaling, dtype=np.float32)

        # Load ONNX model
        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # Optional sanity check
        input_shape = self.session.get_inputs()[0].shape
        output_shape = self.session.get_outputs()[0].shape
        print(f"Loaded ONNX simulator model: {model_path}")
        print(f"Input name: {self.input_name}, shape: {input_shape}")
        print(f"Output name: {self.output_name}, shape: {output_shape}")

    def pose_dynamics(self, car_state):
        theta = car_state[2]
        vf = car_state[3]
        beta_f = car_state[4]
        beta_r = car_state[5]

        omega = vf * np.sin(beta_f - beta_r) / ((self.par.lf + self.par.lr) * np.cos(beta_r))
        vx_b = vf * np.cos(beta_f)
        vy_b = vf * np.sin(beta_f) - self.par.lf * omega

        pose_dot = np.zeros(7, dtype=np.float32)
        pose_dot[0] = vx_b * np.cos(theta) - vy_b * np.sin(theta)
        pose_dot[1] = vx_b * np.sin(theta) + vy_b * np.cos(theta)
        pose_dot[2] = omega

        return pose_dot

    def pose_forward(self, delta_t=0.1):
        k1 = self.pose_dynamics(self.car_state)
        k2 = self.pose_dynamics(self.car_state + 0.5 * delta_t * k1)
        k3 = self.pose_dynamics(self.car_state + 0.5 * delta_t * k2)
        k4 = self.pose_dynamics(self.car_state + delta_t * k3)
        next_pose = self.car_state + (delta_t / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        next_pose[2] = normalize_angle(next_pose[2])

        return next_pose[0:3]

    def build_mlp_input(self, car_input):
        vf = self.car_state[3]
        beta_f = self.car_state[4]
        beta_r = self.car_state[5]
        delta = self.car_state[6]

        # Convert to MLP coordinates
        alpha_f = beta_f - delta
        alpha_r = beta_r

        # inverted input: [steering, acceleration]
        kinematic_input = np.array([car_input[1], car_input[0]], dtype=np.float32)
        kinematic_state = np.array([vf, alpha_f, alpha_r, delta], dtype=np.float32)

        mlp_input = np.concatenate([
            kinematic_state / self.x_scaling,
            kinematic_input
        ], axis=0).astype(np.float32)

        if mlp_input.shape != (6,):
            raise ValueError(f"Expected MLP input shape (6,), got {mlp_input.shape}")

        return mlp_input.reshape(1, 6)

    def velocity_forward(self, car_input, delta_t=0.1):
        delta = self.car_state[6]

        mlp_input = self.build_mlp_input(car_input)

        output = self.session.run(
            [self.output_name],
            {self.input_name: mlp_input}
        )[0]

        next_kinematic_state = output.squeeze(0)  # shape (3,)

        vf_plus = max(0.0, next_kinematic_state[0]) if next_kinematic_state[0] >= 0.03 else 0.0
        alpha_f_plus = next_kinematic_state[1]
        alpha_r_plus = next_kinematic_state[2]

        # Predict delta_k+1
        delta_ref = car_input[1] * (self.par.steering_range / 2)
        delta_dot = np.clip(
            (delta_ref - delta) / self.par.tl_steering,
            -self.par.steering_speed,
            self.par.steering_speed
        )
        delta_plus = delta + delta_dot * delta_t

        # Convert back to kinematic coordinates
        beta_f_plus = alpha_f_plus + delta_plus
        beta_r_plus = alpha_r_plus

        next_velocities = np.zeros(4, dtype=np.float32)
        next_velocities[0] = vf_plus
        next_velocities[1] = beta_f_plus
        next_velocities[2] = beta_r_plus
        next_velocities[3] = delta_plus

        return next_velocities

    def update(self, car_input):
        self.prev_car_state = self.car_state.copy()

        next_pose = self.pose_forward(delta_t=0.1)
        next_velocities = self.velocity_forward(car_input, delta_t=0.1)

        self.car_state[0:3] = next_pose
        self.car_state[3:] = next_velocities