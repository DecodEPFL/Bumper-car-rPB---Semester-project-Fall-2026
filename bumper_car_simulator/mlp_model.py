#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
from utils import normalize_angle

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_sizes=(128, 64)):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], output_dim)
        )

    def forward(self, x):
        return self.model(x)

class MLPModel:
    def __init__(self,
                 initial_state,
                 params,
                 model_path = "best_models/model_kinematic_mlp.pth",
                 input_dim = 2,
                 state_dim = 4,
                 output_dim = 3,
                 hidden_sizes = [256, 128],
                 x_scaling = [1.9, 0.6, 0.1, 2.012]):
        
        # Internal variables
        # self.prev_car_state = torch.tensor(initial_state, dtype=torch.float32)
        # car_state = torch.tensor(initial_state, dtype=torch.float32)
        self.par = params
        
        # Scaling
        self.x_scaling = torch.tensor(x_scaling, dtype=torch.float32)

        # Load weights
        self.model = MLP(input_dim + state_dim, output_dim, hidden_sizes)
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()

        # Save weights
        self.weights = []
        self.biases = []
        for layer in self.model.model:
            if isinstance(layer, nn.Linear):
                self.weights.append(layer.weight.detach())
                self.biases.append(layer.bias.detach())

    def pose_dynamics(self, car_state):
        # Unpack state
        theta = car_state[:, 2]
        vf = car_state[:, 3]
        beta_f = car_state[:, 4]
        beta_r = car_state[:, 5]

        # Convert kinematic state to body frame
        omega = vf*torch.sin(beta_f - beta_r) / ((self.par.lf + self.par.lr)*torch.cos(beta_r))
        vx_b = vf * torch.cos(beta_f)
        vy_b = vf * torch.sin(beta_f) - self.par.lf * omega

        # Compute derivatives
        pose_dot = torch.zeros(car_state.shape[0], 7, dtype=car_state.dtype, device=car_state.device)
        pose_dot[:, 0] = vx_b * torch.cos(theta) - vy_b * torch.sin(theta)
        pose_dot[:, 1] = vx_b * torch.sin(theta) + vy_b * torch.cos(theta)
        pose_dot[:, 2] = omega

        return pose_dot

    def pose_forward(self, car_state, delta_t=0.1):
        # RK4 integration for pose
        k1 = self.pose_dynamics(car_state)
        k2 = self.pose_dynamics(car_state + 0.5 * delta_t * k1)
        k3 = self.pose_dynamics(car_state + 0.5 * delta_t * k2)
        k4 = self.pose_dynamics(car_state + delta_t * k3)
        next_pose = car_state + (delta_t / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        angle_normalized = normalize_angle(next_pose[:, 2])
        next_pose = torch.cat([
            next_pose[:, :2],
            angle_normalized.unsqueeze(1)
        ], dim=-1)
        
        return next_pose

    def velocity_forward(self, car_state, car_input, delta_t=0.1):
    # Unpack state
        vf      = car_state[:, 3]
        beta_f  = car_state[:, 4]
        beta_r  = car_state[:, 5]
        delta   = car_state[:, 6]

        # Convert to MLP coordinates
        alpha_f = beta_f - delta
        alpha_r = beta_r

        kinematic_input = torch.stack([car_input[:, 1], car_input[:, 0]], dim=-1)  # [B, 2]
        
        # FIX 1: was torch.tensor([vf, ...]) which detaches graph
        kinematic_state = torch.stack([vf, alpha_f, alpha_r, delta], dim=-1)  # [B, 4]

        # Predict [vf, alpha_f, alpha_r]_k+1
        h0 = torch.cat([kinematic_state / self.x_scaling, kinematic_input], dim=-1)  # [B, 6]
        h1 = torch.relu(h0 @ self.weights[0].T + self.biases[0])
        h2 = torch.relu(h1 @ self.weights[1].T + self.biases[1])
        next_kinematic_state = h2 @ self.weights[2].T + self.biases[2]  # [B, 3]

        vf_plus     = torch.where(next_kinematic_state[:, 0] >= 0.03,
                                torch.clamp(next_kinematic_state[:, 0], min=0.0),
                                torch.zeros_like(next_kinematic_state[:, 0]))

        alpha_f_plus = next_kinematic_state[:, 1]
        alpha_r_plus = next_kinematic_state[:, 2]

        # Predict delta_k+1
        delta_ref = car_input[:, 1] * (self.par.steering_range / 2)
        delta_dot = torch.clamp((delta_ref - delta) / self.par.tl_steering,
                                -self.par.steering_speed, self.par.steering_speed)  # FIX 3: was torch.clip
        delta_plus = delta + delta_dot * delta_t

        # Convert back to kinematic coordinates
        beta_f_plus = alpha_f_plus + delta_plus
        beta_r_plus = alpha_r_plus

        next_velocities = torch.stack([vf_plus, beta_f_plus, beta_r_plus, delta_plus], dim=-1)  # [B, 4]
        return next_velocities

    def update(self, x_state, car_input, dt=0.1):
        # Update the state
        next_velocities = self.velocity_forward(car_input=car_input, car_state=x_state, delta_t=dt)
        next_pose = self.pose_forward(car_state=x_state, delta_t=dt)
        car_state_new = torch.cat([next_pose, next_velocities], dim=-1)
        return car_state_new