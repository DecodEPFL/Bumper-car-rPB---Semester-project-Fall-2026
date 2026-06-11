#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
from plants.bumpercar.utils import normalize_angle
from .parameters import CarParams as params

import torch
import torch.nn as nn


class PositionPidController(torch.nn.Module):
    def __init__(
        self,
        kp_p,
        ki_p,
        kd_p,
        kp_theta,
        ki_theta,
        kd_theta,
        params,
        n_agents,
        dt,
    ):
        super().__init__()

        self.kp_p = kp_p
        self.ki_p = ki_p
        self.kd_p = kd_p
        self.kp_theta = kp_theta
        self.ki_theta = ki_theta
        self.kd_theta = kd_theta
        self.par = params
        self.n_agents = n_agents
        self.dt = dt

        self.errorSum_p = None
        self.prevError_p = None
        self.errorSum_theta = None
        self.prevError_theta = None

    def reset(self, batch_size, device, dtype):
        self.errorSum_p = torch.zeros(
            batch_size, self.n_agents, device=device, dtype=dtype
        )
        self.prevError_p = torch.zeros(
            batch_size, self.n_agents, device=device, dtype=dtype
        )
        self.errorSum_theta = torch.zeros(
            batch_size, self.n_agents, device=device, dtype=dtype
        )
        self.prevError_theta = torch.zeros(
            batch_size, self.n_agents, device=device, dtype=dtype
        )

    def calculate_control_input(self, p_final, car_state):
        """
        p_final:   [B, n_agents, 2]
        car_state: [B, n_agents, 7]

        returns:
            car_input [B, 2*n_agents]
            layout: [drive_1, steer_1, drive_2, steer_2, ...]
        """
        batch_size = car_state.shape[0]
        if (self.errorSum_p is None
            or self.errorSum_p.shape[0] != batch_size
            or self.errorSum_p.device != car_state.device
            or self.errorSum_p.dtype != car_state.dtype):
            
            self.reset(
                batch_size=batch_size,
                device=car_state.device,
                dtype=car_state.dtype,
            )

        p = car_state[:, :, 0:2]
        theta = car_state[:, :, 2]
        vf = car_state[:, :, 3]

        error_vec = p_final - p
        error_pos = torch.linalg.norm(error_vec, dim=-1)

        self.errorSum_p = self.errorSum_p + error_pos * self.dt
        d_error_p = (error_pos - self.prevError_p) / self.dt
        self.prevError_p = error_pos

        v_ref = (
            self.kp_p * error_pos
            + self.ki_p * self.errorSum_p
            + self.kd_p * d_error_p
        )

        theta_desired = torch.atan2(error_vec[:, :, 1], error_vec[:, :, 0])
        error_theta = normalize_angle(theta_desired - theta)

        self.errorSum_theta = self.errorSum_theta + error_theta * self.dt
        d_error_theta = normalize_angle(error_theta - self.prevError_theta) / self.dt
        self.prevError_theta = error_theta

        delta = (
            self.kp_theta * error_theta
            + self.ki_theta * self.errorSum_theta
            + self.kd_theta * d_error_theta
        )

        dir_to_goal = error_vec / (error_pos.unsqueeze(-1) + 1e-6)
        vel_vector = torch.stack(
            [
                vf * torch.cos(theta),
                vf * torch.sin(theta),
            ],
            dim=-1,
        )
        v_along_goal = torch.sum(vel_vector * dir_to_goal, dim=-1)

        v_ref = torch.where(
            v_along_goal < 0,
            torch.full_like(v_ref, 0.1),
            v_ref,
        )

        drive_accel = (v_ref / self.par.max_speed) * 0.62 + 0.05
        drive_accel = torch.clamp(drive_accel, 0.0, 1.0)

        drive_brake = -torch.clamp(
            -(v_ref / (vf + 1e-5) - 1.0) * 1.04 * 0.6 + 0.2,
            0.0,
            1.0,
        )

        drive = torch.where(v_ref < vf, drive_brake, drive_accel)
        steer = delta / (self.par.steering_range / 2)

        stop_radius = 0.1
        inside_stop_radius = error_pos < stop_radius

        drive = torch.where(
            inside_stop_radius,
            torch.zeros_like(drive),
            drive,
        )
        steer = torch.where(
            inside_stop_radius,
            torch.zeros_like(steer),
            steer,
        )

        drive = torch.clamp(drive, -1.0, 1.0)
        steer = torch.clamp(steer, -1.0, 1.0)

        car_input = torch.stack([drive, steer], dim=-1)
        return car_input.reshape(car_state.shape[0], 2 * self.n_agents)

    
    
class BumpercarSystem(nn.Module):
    def __init__(
        self,
        params,
        x_init=None,
        u_init=None,
        n_agents=2,
        dt=0.04,
        model_path="plants/bumpercar/model_kinematic_mlp.pth",
    ):
        super().__init__()

        self.positionPID = PositionPidController(
            kp_p=0.5, kd_p=0.0, ki_p=0.0,
            kp_theta=1.0, kd_theta=0.0, ki_theta=0.0,
            params=params, n_agents=n_agents, dt=dt)
        
        self.par = params
        self.n_agents = n_agents
        self.agent_state_dim = 7
        self.agent_in_dim = 2

        self.state_dim = self.agent_state_dim * self.n_agents
        self.in_dim = self.agent_in_dim * self.n_agents
        self.dt = dt

        x_init = torch.zeros((1, self.state_dim)) if x_init is None else x_init.reshape(1, -1)
        u_init = torch.zeros((1, self.in_dim)) if u_init is None else u_init.reshape(1, -1)

        self.register_buffer("x_init", x_init.float())
        self.register_buffer("u_init", u_init.float())

        assert self.x_init.shape[1] == self.state_dim
        assert self.u_init.shape[1] == self.in_dim

        self.dynamics = MLPDynamicsModel(
            initial_state=None,
            params=params,
            model_path=model_path,
        )

    def pos_indices(self):
        idx = []
        for i in range(self.n_agents):
            base = self.agent_state_dim * i
            idx += [base + 0, base + 1]
        return idx

    def _split_agent_state(self, x, agent_idx):
        base = self.agent_state_dim * agent_idx
        return x[:, base:base + self.agent_state_dim]

    def _split_agent_input(self, u, agent_idx):
        base = self.agent_in_dim * agent_idx
        return u[:, base:base + self.agent_in_dim]

    def _physical_controller(self, x, v, xbar, dxref):
        # x must be [B, state_dim], not [B, T, state_dim]
        if x.dim() == 3:
            x = x.squeeze(1)
        if xbar.dim() == 3:
            xbar = xbar.squeeze(1)
        if dxref.dim() == 3:
            dxref = dxref.squeeze(1)

        pos = x[:, self.pos_indices()]
        ref = xbar + dxref
        e = ref - pos
        v = v + e

        car_state = x.reshape(x.shape[0], self.n_agents, self.agent_state_dim)
        p_final = ref.reshape(ref.shape[0], self.n_agents, self.agent_in_dim)

        car_input = self.positionPID.calculate_control_input(
            p_final=p_final,
            car_state=car_state,
        )

        return car_input, v



    def noiseless_forward(self, t, x, v, u, xbar):
        """
        x:    [B, 1, 7*n_agents]
        v:    [B, 1, 2*n_agents]
        u:    [B, 1, 2*n_agents], controller output / dxref
        xbar: [B, 1, 2*n_agents], refrence positions
        """

        batch_size = x.shape[0]

        x_flat = x.view(batch_size, self.state_dim)
        v_flat = v.view(batch_size, self.in_dim)
        dxref = u.view(batch_size, self.in_dim)
        xbar_flat = xbar.view(batch_size, self.in_dim)

        car_input, v_next = self._physical_controller(
            x=x_flat,
            v=v_flat,
            xbar=xbar_flat,
            dxref=dxref,
        )

        next_states = []

        for i in range(self.n_agents):
            x_i = self._split_agent_state(x_flat, i)
            u_i = self._split_agent_input(car_input, i)

            x_next_i = self.dynamics.update(
                x_state=x_i,
                car_input=u_i,
                dt=self.dt,
            )

            next_states.append(x_next_i)

        x_next = torch.cat(next_states, dim=-1)

        return x_next.view(batch_size, 1, self.state_dim), v_next.view(batch_size, 1, self.in_dim)

    def forward(self, t, x, v, u, w, xbar):
        f, v = self.noiseless_forward(t=t, x=x, v=v, u=u, xbar=xbar)
        f = f + w.view(-1, 1, self.state_dim)
        return f, v

    def rollout(self, controller, data, train=False):
        controller.reset()
        
        batch_size = data.shape[0]
        self.positionPID.reset(
            batch_size=batch_size,
            device=data.device,
            dtype=data.dtype,    
            )

        nx = self.state_dim
        x = data[:, 0:1, :nx].detach().clone()
        u = self.u_init.detach().clone().repeat(batch_size, 1, 1)
        v = torch.zeros_like(u)

        w = torch.zeros_like(data[:, :, :nx])


        # dataset stores x-y references:
        ref = data[:, :, nx:nx + nx]
        xbar = ref[:, :, self.pos_indices()]

        for t in range(data.shape[1]):
            x, v = self.forward(
                t=t,
                x=x,
                v=v,
                u=u,
                w=w[:, t:t + 1, :],
                xbar=xbar[:, t:t + 1, :],
            )

            u = controller(x, v, xbar[:, t:t + 1, :])

            e = ref[:, t:t + 1, :] - x

            if t == 0:
                x_log, u_log, v_log, e_log = x, u, v, e
            else:
                x_log = torch.cat((x_log, x), dim=1)
                u_log = torch.cat((u_log, u), dim=1)
                v_log = torch.cat((v_log, v), dim=1)
                e_log = torch.cat((e_log, e), dim=1)

        controller.reset()

        if not train:
            x_log = x_log.detach()
            u_log = u_log.detach()

        self.v_log = v_log.detach()
        return x_log, e_log, u_log



class MLPDynamicsModel(nn.Module):
    def __init__(self,
                 initial_state,
                 params,
                 model_path = "plants/bumpercar/model_kinematic_mlp.pth",
                 input_dim = 2,
                 state_dim = 4,
                 output_dim = 3,
                 hidden_sizes = [256, 128],
                 x_scaling = [1.9, 0.6, 0.1, 2.012]):
        
        super().__init__()

        # Internal variables
        # self.prev_car_state = torch.tensor(initial_state, dtype=torch.float32)
        # car_state = torch.tensor(initial_state, dtype=torch.float32)
        self.par = params
        
        # Scaling
        self.register_buffer("x_scaling", torch.tensor(x_scaling, dtype=torch.float32))

        # Load weights
        self.model = MLP(input_dim + state_dim, output_dim, hidden_sizes)
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

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
        next_kinematic_state = self.model(h0)  # [B, 3]

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
    
    
