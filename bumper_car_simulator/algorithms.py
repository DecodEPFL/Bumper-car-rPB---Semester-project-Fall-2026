#!/usr/bin/env python3
import numpy as np
import casadi as ca
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from mlp_model import MLPModel
from utils import beta_to_delta, delta_to_beta

def inspect_constraints(sol, lbg, ubg, g_labels):
    # sol['g'] contains the evaluated constraint values at the solution
    g_values = sol['g'].full().flatten()
    
    df = pd.DataFrame({
        'Comment': g_labels,
        'Lower Bound (lbg)': lbg,
        'Current g(x)': g_values,
        'Upper Bound (ubg)': ubg
    })
    
    # Add a column to easily spot violations
    df['Status'] = 'Feasible'
    df.loc[df['Current g(x)'] < df['Lower Bound (lbg)'] - 1e-6, 'Status'] = 'VIOLATED (Low)'
    df.loc[df['Current g(x)'] > df['Upper Bound (ubg)'] + 1e-6, 'Status'] = 'VIOLATED (High)'
    
    return df

class ControllerCBF:
    def __init__(self, safety_radius, barrier_gain, barrier_type, par):
        self.safety_radius = safety_radius
        self.wall_safety_radius = 1.0
        self.barrier_gain = barrier_gain
        self.barrier_type = barrier_type
        self.par = par

    def input_conversion(self, car_state_i, user_input_i):
        vf_i = car_state_i[3]
        if user_input_i[0] < 0.0:
            u_throttle = -np.clip((-user_input_i[0] * 100.0 - 20.0) / (80.0 - 20.0), 0.0, 1.0)
        else:
            u_throttle = np.clip((user_input_i[0] * 100.0 - 5.0) / (67.0 - 5.0), 0.0, 1.0)
        
        if u_throttle < 0.0:
            vf_ref = (1.0 + u_throttle / 1.04) * vf_i
        elif u_throttle / (1/self.par.max_speed) > vf_i:
            vf_ref = u_throttle / (1/self.par.max_speed)
        else:
            vf_ref = vf_i
        delta_ref = user_input_i[1] * self.par.steering_range / 2

        return vf_ref, delta_ref

    def car_dynamics(self, car_state_i):
        theta_i, vf_i, beta_f_i, beta_r_i, delta_i = car_state_i[2:]

        beta_b_cog_i = np.arctan2(self.par.lr*np.tan(beta_f_i) + self.par.lf*np.tan(beta_r_i), self.par.lf + self.par.lr)
        beta_g_cog_i = theta_i + beta_b_cog_i
        v_g_cog_i = vf_i * np.cos(beta_f_i) / np.cos(beta_b_cog_i)
        vx_cog_i = v_g_cog_i * np.cos(beta_g_cog_i)
        vy_cog_i = v_g_cog_i * np.sin(beta_g_cog_i)
        
        S_beta_f_i = np.sqrt(np.cos(beta_f_i)**2 + (self.par.lr/(self.par.lf+self.par.lr)*np.sin(beta_f_i))**2)
        v_g_cog_dot_1_i = ((self.par.lr/(self.par.lf+self.par.lr))**2 - 1.0) * np.sin(beta_f_i) * np.cos(beta_f_i) * vf_i / (S_beta_f_i * self.par.tl_steering)
        v_g_cog_dot_2_i = -0.32*vf_i + 0.379*pow(vf_i, 2) -0.155*pow(vf_i, 3) - 4.75*pow(beta_f_i - delta_i, 2)
        beta_b_cog_dot_1_i = (self.par.lf + self.par.lr) * self.par.lr / (np.cos(beta_f_i)**2 * ((self.par.lf+self.par.lr)**2 + (self.par.lr*np.tan(beta_f_i) + self.par.lf*np.tan(beta_r_i))**2) * self.par.tl_steering)

        f_i = np.array([vx_cog_i, 
                        vy_cog_i, 
                        v_g_cog_i/self.par.lr*np.sin(beta_b_cog_i), 
                        -v_g_cog_dot_1_i*delta_i + S_beta_f_i*(v_g_cog_dot_2_i - 0.53*vf_i), 
                        -beta_b_cog_dot_1_i*delta_i]).reshape(5, 1)
        
        g_i = np.array([[0.0, 0.0], 
                        [0.0, 0.0], 
                        [0.0, 0.0], 
                        [0.53*S_beta_f_i, v_g_cog_dot_1_i], 
                        [0, beta_b_cog_dot_1_i]]).reshape(5, 2)
        
        return f_i, g_i

    def gradient_C3BF(self, car_state_i, car_state_j):
        # Read state
        x_i, y_i, theta_i, vf_i, beta_f_i, beta_r_i = car_state_i[:6]
        x_j, y_j, theta_j, vf_j, beta_f_j, beta_r_j = car_state_j[:6]

        # Car i
        beta_b_cog_i = np.arctan2(self.par.lr*np.tan(beta_f_i) + self.par.lf*np.tan(beta_r_i), self.par.lf + self.par.lr)
        beta_g_cog_i = theta_i + beta_b_cog_i
        v_g_cog_i = vf_i * np.cos(beta_f_i) / np.cos(beta_b_cog_i)
        vx_cog_i = v_g_cog_i * np.cos(beta_g_cog_i)
        vy_cog_i = v_g_cog_i * np.sin(beta_g_cog_i)

        # Car j
        beta_b_cog_j = np.arctan2(self.par.lr*np.tan(beta_f_j) + self.par.lf*np.tan(beta_r_j), self.par.lf + self.par.lr)
        beta_g_cog_j = theta_j + beta_b_cog_j
        v_g_cog_j = vf_j * np.cos(beta_f_j) / np.cos(beta_b_cog_j)
        vx_cog_j = v_g_cog_j * np.cos(beta_g_cog_j)
        vy_cog_j = v_g_cog_j * np.sin(beta_g_cog_j)

        # C3BF i --> j
        p_rel_i_j = np.array([x_j - x_i, y_j - y_i])
        p_rel_norm = np.sqrt(np.dot(p_rel_i_j, p_rel_i_j))
        
        v_rel_i_j = np.array([vx_cog_j - vx_cog_i, vy_cog_j - vy_cog_i])
        v_rel_norm = np.sqrt(np.dot(v_rel_i_j, v_rel_i_j))
        
        if p_rel_norm > 2.0 + self.safety_radius:
            h = 0
            grad_h_i_j = np.zeros((5, 1))
            grad_h_j_i = np.zeros((5, 1))

            return h, grad_h_i_j, grad_h_j_i
        
        # Compute barrier function gradient
        if p_rel_norm < self.safety_radius:
            # print(f"Warning: Inside safety radius! p_rel = {p_rel_norm}")
            cos_phi = -np.sqrt(pow(2 * self.safety_radius - p_rel_norm, 2) - pow(self.safety_radius, 2)) / (2 * self.safety_radius - p_rel_norm + 1e-5)
            tan_phi_sq = pow(self.safety_radius, 2) / (pow(2 * self.safety_radius - p_rel_norm, 2) - pow(self.safety_radius, 2) + 1e-5)
            correction_term = -(p_rel_norm + 1e-5) / (2 * self.safety_radius - p_rel_norm)
        else:
            cos_phi = np.sqrt(abs(pow(p_rel_norm, 2) - pow(self.safety_radius, 2))) / (p_rel_norm + 1e-5)
            tan_phi_sq = pow(self.safety_radius, 2) / (pow(p_rel_norm, 2) - pow(self.safety_radius, 2) + 1e-5)
            correction_term = 1.0
        
        h = np.dot(p_rel_i_j, v_rel_i_j) + v_rel_norm * p_rel_norm * cos_phi
        
        grad_h_1_i_j = np.array([-v_rel_i_j[0],
                            -v_rel_i_j[1],
                            vy_cog_i*p_rel_i_j[0] - vx_cog_i*p_rel_i_j[1],
                            -np.cos(beta_g_cog_i)*p_rel_i_j[0] - np.sin(beta_g_cog_i)*p_rel_i_j[1],
                            vy_cog_i*p_rel_i_j[0] - vx_cog_i*p_rel_i_j[1],
                            ])

        grad_h_2_i_j = np.array([-p_rel_i_j[0],
                            -p_rel_i_j[1],
                            0.0,
                            0.0,
                            0.0
                            ]) * v_rel_norm / (p_rel_norm + 1e-5) * cos_phi

        grad_h_3_i_j = np.array([-p_rel_i_j[0],
                            -p_rel_i_j[1],
                            0.0,
                            0.0,
                            0.0
                            ]) * v_rel_norm / (p_rel_norm + 1e-5) * cos_phi * tan_phi_sq * correction_term

        grad_h_4_i_j = np.array([0.0,
                            0.0,
                            np.dot(v_rel_i_j, np.array([vy_cog_i, -vx_cog_i])),
                            -np.dot(v_rel_i_j, np.array([np.cos(beta_g_cog_i), np.sin(beta_g_cog_i)])),
                            np.dot(v_rel_i_j, np.array([vy_cog_i, -vx_cog_i]))
                            ]) * p_rel_norm / (v_rel_norm + 1e-5) * cos_phi

        grad_h_i_j = grad_h_1_i_j + grad_h_2_i_j + grad_h_3_i_j + grad_h_4_i_j

        # C3BF j --> i
        p_rel_j_i = np.array([x_i - x_j, y_i - y_j])
        v_rel_j_i = np.array([vx_cog_i - vx_cog_j, vy_cog_i - vy_cog_j])
        
        grad_h_1_j_i = np.array([-v_rel_j_i[0],
                                 -v_rel_j_i[1],
                                 vy_cog_j*p_rel_j_i[0] - vx_cog_j*p_rel_j_i[1],
                                 -np.cos(beta_g_cog_j)*p_rel_j_i[0] - np.sin(beta_g_cog_j)*p_rel_j_i[1],
                                 vy_cog_j*p_rel_j_i[0] - vx_cog_j*p_rel_j_i[1],
                                 ])

        grad_h_2_j_i = np.array([-p_rel_j_i[0],
                                 -p_rel_j_i[1],
                                 0.0,
                                 0.0,
                                 0.0
                                 ]) * v_rel_norm / (p_rel_norm + 1e-5) * cos_phi

        grad_h_3_j_i = np.array([-p_rel_j_i[0],
                                 -p_rel_j_i[1],
                                 0.0,
                                 0.0,
                                 0.0
                                 ]) * v_rel_norm / (p_rel_norm + 1e-5) * cos_phi * tan_phi_sq * correction_term

        grad_h_4_j_i = np.array([0.0,
                                 0.0,
                                 np.dot(v_rel_j_i, np.array([vy_cog_j, -vx_cog_j])),
                                 -np.dot(v_rel_j_i, np.array([np.cos(beta_g_cog_j), np.sin(beta_g_cog_j)])),
                                 np.dot(v_rel_j_i, np.array([vy_cog_j, -vx_cog_j]))
                                 ]) * p_rel_norm / (v_rel_norm + 1e-5) * cos_phi

        grad_h_j_i = grad_h_1_j_i + grad_h_2_j_i + grad_h_3_j_i + grad_h_4_j_i

        return h, grad_h_i_j, grad_h_j_i

    def gradient_RadiusCBF(self, car_state_i, car_state_j):
        # Read state
        x_i, y_i, theta_i, vf_i, beta_f_i, beta_r_i = car_state_i[:6]
        x_j, y_j, theta_j, vf_j, beta_f_j, beta_r_j = car_state_j[:6]

        # Car i
        beta_b_cog_i = np.arctan2(self.par.lr*np.tan(beta_f_i) + self.par.lf*np.tan(beta_r_i), self.par.lf + self.par.lr)
        beta_g_cog_i = theta_i + beta_b_cog_i
        v_g_cog_i = vf_i * np.cos(beta_f_i) / np.cos(beta_b_cog_i)
        vx_cog_i = v_g_cog_i * np.cos(beta_g_cog_i)
        vy_cog_i = v_g_cog_i * np.sin(beta_g_cog_i)

        # Car j
        beta_b_cog_j = np.arctan2(self.par.lr*np.tan(beta_f_j) + self.par.lf*np.tan(beta_r_j), self.par.lf + self.par.lr)
        beta_g_cog_j = theta_j + beta_b_cog_j
        v_g_cog_j = vf_j * np.cos(beta_f_j) / np.cos(beta_b_cog_j)
        vx_cog_j = v_g_cog_j * np.cos(beta_g_cog_j)
        vy_cog_j = v_g_cog_j * np.sin(beta_g_cog_j)

        # Radius barrier
        p_rel_i_j = np.array([x_j - x_i, y_j - y_i])
        p_rel_j_i = np.array([x_i - x_j, y_i - y_j])
        p_rel_norm = np.sqrt(np.dot(p_rel_i_j, p_rel_i_j))

        h = p_rel_norm - self.safety_radius

        h_dot = 1/p_rel_norm * (-np.dot(p_rel_i_j, [vx_cog_i, vy_cog_i]) - np.dot(p_rel_j_i, [vx_cog_j, vy_cog_j]))

        grad_h_i_j = np.array([vx_cog_i - vx_cog_j,
                               vy_cog_i - vy_cog_j,
                               np.dot(p_rel_i_j, np.array([vy_cog_i, -vx_cog_i])),
                               -np.dot(p_rel_i_j, np.array([np.cos(beta_g_cog_i), np.sin(beta_g_cog_i)])),
                               np.dot(p_rel_i_j, np.array([vy_cog_i, -vx_cog_i]))
                               ]) / p_rel_norm
        
        grad_h_j_i = np.array([- vx_cog_i + vx_cog_j,
                               - vy_cog_i + vy_cog_j,
                               np.dot(p_rel_i_j, np.array([-vy_cog_j, vx_cog_j])),
                               np.dot(p_rel_i_j, np.array([np.cos(beta_g_cog_j), np.sin(beta_g_cog_j)])),
                               np.dot(p_rel_i_j, np.array([-vy_cog_j, vx_cog_j]))
                               ]) / p_rel_norm

        return h, h_dot, grad_h_i_j, grad_h_j_i

    def gradient_walls(self, car_state_i, normal_vector, wall_position):
        # Read state
        x_i, y_i, theta_i, vf_i, beta_f_i, beta_r_i = car_state_i[:6]
        beta_b_cog_i = np.arctan2(self.par.lr*np.tan(beta_f_i) + self.par.lf*np.tan(beta_r_i), self.par.lf + self.par.lr)
        beta_g_cog_i = theta_i + beta_b_cog_i

        h_wall = np.dot(normal_vector, [x_i, y_i]) - (wall_position + self.wall_safety_radius)

        h_dot_wall = vf_i * (normal_vector[0]*np.cos(beta_g_cog_i) + normal_vector[1]*np.sin(beta_g_cog_i))

        grad_h_wall = np.array([0.0,
                                0.0,
                                vf_i * (-normal_vector[0]*np.sin(beta_g_cog_i) + normal_vector[1]*np.cos(beta_g_cog_i)),
                                normal_vector[0]*np.cos(beta_g_cog_i) + normal_vector[1]*np.sin(beta_g_cog_i),
                                vf_i * (-normal_vector[0]*np.sin(beta_g_cog_i) + normal_vector[1]*np.cos(beta_g_cog_i))
                                ])

        return h_wall, h_dot_wall, grad_h_wall
    
    def compute_safe_input(self, cars_state, user_inputs):
        raise NotImplementedError("You need to implement this method!")

class PositionPidController():
    def __init__(self, kp_p, ki_p, kd_p, kp_theta, ki_theta, kd_theta, params):
        self.kp_p = kp_p
        self.ki_p = ki_p
        self.kd_p = kd_p
        self.kp_theta = kp_theta
        self.kd_theta = kd_theta
        self.ki_theta = ki_theta
        self.errorSum_p = 0.0
        self.prevError_p = 0.0
        self.errorSum_theta = 0.0
        self.prevError_theta = 0.0
        self.input_old = np.zeros(2)
        self.par = params

    def calculateControlInput(self, p_final, car_state, dt):
        p = car_state[0:2]
        theta = car_state[2]
        vf = car_state[3]
        p_final = p_final[0:2]

        error_vec = p_final - p
        error_pos = np.linalg.norm(error_vec)
        if error_pos < 0.1:
            print("True")
            v_ref = 0.0
            delta = 0.0
            return np.array([v_ref, delta])

        self.errorSum_p += error_pos * dt
        d_error_p = (error_pos - self.prevError_p) / dt
        self.prevError_p = error_pos
        v_ref = (
            self.kp_p * error_pos
            + self.ki_p * self.errorSum_p
            + self.kd_p * d_error_p
        )

        theta_desired = np.arctan2(error_vec[1], error_vec[0])
        error_theta = theta_desired - theta
        error_theta = (error_theta + np.pi) % (2*np.pi) - np.pi
        self.errorSum_theta += error_theta * dt
        d_error_theta = (error_theta - self.prevError_theta) / dt
        self.prevError_theta = error_theta
        delta = (
            self.kp_theta * error_theta
            + self.ki_theta * self.errorSum_theta
            + self.kd_theta * d_error_theta
        )
        
        dir_to_goal = error_vec / np.linalg.norm(error_vec)
        vel_vector = np.array([car_state[3]*np.cos(theta), car_state[3]*np.sin(theta)])
        v_along_goal = np.dot(vel_vector, dir_to_goal)
        if v_along_goal < 0:
            v_ref = 0.1 
        
        if v_ref < vf:
            self.input_old[0] = -np.clip(-(v_ref / (vf + 1e-5) - 1.0) * 1.04 * 0.6 + 0.2, 0.0, 1.0)
        else:
            self.input_old[0] = np.clip((v_ref/self.par.max_speed) * 0.62 + 0.05, 0.0, 1.0)   
        
        self.input_old[1] = delta / (self.par.steering_range / 2)
        
        return self.input_old


class PolicyGRU(nn.Module):
    def __init__(self, params, hidden=(128, 64), num_layers=1, dropout=0.05):
        super().__init__()

        x_scale = torch.tensor([3., 3.,np.pi, 1., 1., 1., 1.])
        self.register_buffer("x_scale", x_scale.detach().clone().float())  # expected shape: (7,)
        self.register_buffer(
            "u_max",
            torch.tensor(
                [params.max_speed, params.steering_range],
                dtype=torch.float32,
            ),
        ) 

        gru_hidden = hidden[0]
        head_hidden = hidden[1]

        self.gru = nn.GRU(
            input_size=8,              
            hidden_size=gru_hidden,
            num_layers=num_layers,
            batch_first=True,        
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.Linear(gru_hidden, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, 2)
        )

        nn.init.uniform_(self.head[-1].weight, -1.0, 1.0)
        nn.init.uniform_(self.head[-1].bias, -1.0, 1.0)

        self.params = params

    def forward(self, x, h0=None, return_hidden=True):
        unbatched = False

        if x.dim() == 2:
            x = x.unsqueeze(0)   
            unbatched = True

        elif x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0) 
            unbatched = True

        if x.size(-1) != 8:
            raise ValueError(f"Expected input feature size 8, got {x.size(-1)}")
        
        x = x.clone()
        x[..., :7] = x[..., :7] / self.x_scale.clamp_min(1e-6)

        y, h_n = self.gru(x, h0)
        y_last = y[:, -1, :]             

        u = self.head(y_last)            
        u = torch.tanh(u) * self.u_max   

        if unbatched:
            u = u.squeeze(0)

        if return_hidden:
            return u, h_n
        return u

class PolicyMLP(nn.Module):
    def __init__(self, x_scale, params, hidden=(128, 64)):
        super().__init__()
        self.register_buffer("x_scale", x_scale.detach().clone().float())  # (7,)
        self.register_buffer("u_max",   torch.tensor([params.max_speed, params.steering_range],   dtype=torch.float32))  # (2,)
        self.net = nn.Sequential(
            nn.Linear(8, hidden[0]),
            nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]),
            nn.ReLU(),
            nn.Linear(hidden[1], 2),
        )
        nn.init.uniform_(self.net[-1].weight, -1.0, 1.0)
        nn.init.uniform_(self.net[-1].bias, -1.0, 1.0)
        self.params = params

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        x_n = x / self.x_scale
        u   = self.net(x_n)

        v_ref = torch.tanh(u[..., [0]])
        delta_ref = torch.tanh(u[..., [1]])
        
        return torch.cat([v_ref, delta_ref], dim=-1)



class NN_Controller():
    def __init__(self, type, params): #Type: 'GRU', 'MLP'
        self.throttle_old = 0
        self.delta_old = 0
        self.X_SCALE = torch.tensor([3., 3.,np.pi, 1., 1., 1., 1., 1.])
        self.policy_type = type
        
        if self.policy_type == "GRU": 
            self.policy = PolicyGRU(params=params)
        elif self.policy_type == "MLP": 
            self.policy = PolicyMLP(self.X_SCALE, params=params)
        else:
            assert("[NN_controller] Not valid Neural network policy input")

    def rollout_policy(self, policy, x0_batch, x_final_batch, model, T_STEPS=80):
        x_dim = 7
        u_dim = 2
        batch_size = x0_batch.shape[0]
        X = torch.zeros(batch_size, T_STEPS, x_dim, device=x0_batch.device)
        U = torch.zeros(batch_size, T_STEPS, u_dim, device=x0_batch.device)
        x_state = x0_batch.clone().detach().requires_grad_(True)


        h = None
        for k in range(T_STEPS):
            if x_state.dim() == 1:
                x_state = x_state.unsqueeze(0)  # [1, 7]
                    
            error = x_final_batch - x_state
            
            # Convert to body frame
            dx = error[:, 0]
            dy = error[:, 1]
            theta = x_state[:, 2]
            dx_local =  torch.cos(theta) * dx + torch.sin(theta) * dy
            dy_local = -torch.sin(theta) * dx + torch.cos(theta) * dy
            
            sin_dtheta = torch.sin(x_final_batch[:, 2] - x_state[:, 2])
            cos_dtheta = torch.cos(x_final_batch[:, 2] - x_state[:, 2])

            error_body = torch.cat([
                dx_local.unsqueeze(1),
                dy_local.unsqueeze(1),
                sin_dtheta.unsqueeze(1),
                cos_dtheta.unsqueeze(1),
                error[:, 3:]
            ], dim=-1)

            policy_input = torch.cat([error_body], dim=-1)
            if self.policy_type == "MLP":
                u = policy(policy_input)
            elif self.policy_type == "GRU":
                policy_input = policy_input.unsqueeze(1)   # (B, 1, 8)
                u, h = policy(policy_input, h0=h, return_hidden=True)

            x_state = model.update(x_state, u)

            normalized_angle = torch.atan2(torch.sin(x_state[:, 2]), torch.cos(x_state[:, 2]))
            x_state = torch.cat([
                x_state[:, :2],
                normalized_angle.unsqueeze(1),
                x_state[:, 3:]
            ], dim=-1)

            X[:, k, :] = x_state
            U[:, k, :] = u
        return X, U
    
    def cost(self, X, U, final_point):        
        Q = torch.as_tensor(np.diag([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 7.0]), dtype=torch.float32) * 0.3
        Q_final = torch.as_tensor(np.diag([50.0, 50.0, 20.0, 10.0, 0.0, 0.0, 0.0]), dtype=torch.float32) * 5
        R = torch.as_tensor(np.diag([0.1, 0.1]), dtype=torch.float32) * 0.5

        # State error
        X_err = final_point.unsqueeze(1) - X

        # Global position error
        dx = X_err[:, :, 0]
        dy = X_err[:, :, 1]

        # Heading of current state
        theta = X[:, :, 2]
        c = torch.cos(theta)
        s = torch.sin(theta)

        # Rotate position error into body frame
        dx_local = c * dx + s * dy
        dy_local = -s * dx + c * dy

        # Wrapped heading error
        dtheta = torch.atan2(torch.sin(X_err[:, :, 2]), torch.cos(X_err[:, :, 2]))

        # Rebuild error in local coordinates
        X_err = torch.stack([
            dx_local,
            dy_local,
            dtheta,
            X_err[:, :, 3],
            X_err[:, :, 4],
            X_err[:, :, 5],
            X_err[:, :, 6],
        ], dim=-1)

        # Final-state error over last 5 steps
        X_err_final = final_point.unsqueeze(1) - X[:, -5:, :]

        dx_f = X_err_final[:, :, 0]
        dy_f = X_err_final[:, :, 1]
        theta_f = X[:, -5:, 2]
        c_f = torch.cos(theta_f)
        s_f = torch.sin(theta_f)

        dx_local_f = c_f * dx_f + s_f * dy_f
        dy_local_f = -s_f * dx_f + c_f * dy_f
        dtheta_f = torch.atan2(torch.sin(X_err_final[:, :, 2]), torch.cos(X_err_final[:, :, 2]))

        X_err_final = torch.stack([
            dx_local_f,
            dy_local_f,
            dtheta_f,
            X_err_final[:, :, 3],
            X_err_final[:, :, 4],
            X_err_final[:, :, 5],
            X_err_final[:, :, 6],
        ], dim=-1)

        U = U[:, :-1, :].float()

        cost_x = torch.einsum("bti,ij,btj->bt", X_err, Q, X_err)
        cost_u = torch.einsum("bti,ij,btj->bt", U, R, U)
        cost_x_final = torch.einsum("bti,ij,btj->bt", X_err_final, Q_final, X_err_final)

        L = cost_x.sum(dim=1) + cost_u.sum(dim=1) + cost_x_final.sum(dim=1)
        return L

    def train_mlp(self, init_points, final_points, car_state_init, params, lr=1e-4, epochs=2000, device="cpu", 
                  experiment_name = ""):
        torch.manual_seed(42)
        self.policy.to(device)
        self.policy.train()
        
        opt = torch.optim.Adam(self.policy.parameters(), lr=lr)                               # weight_decay: add an L2 regularization term to the loss for each parameter
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=20, min_lr=1e-6)    # reduce lr if no improvement for 50 epochs
        
        init_points = init_points.to(device)
        final_points = final_points.to(device)
        N = len(init_points)
        batch_size = 1 if N == 1 else 32
        
        # Training and validation
        val_ratio = 0.2
        n_val = max(1, int(val_ratio * N)) if N > 1 else 0
        perm = torch.randperm(N, device=device)
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]

        init_train = init_points[train_idx]
        final_train = final_points[train_idx]
        init_val = init_points[val_idx]
        final_val = final_points[val_idx]
        
        train_losses = []
        val_losses = []

        #initialize model
        model = MLPModel(car_state_init, params=params)
        
        for ep in range(epochs):
            self.policy.train()
            perm_train = torch.randperm(len(init_train), device=device)
            init_train = init_train[perm_train]
            final_train = final_train[perm_train]
            
            batch_losses = []
            for i in range(0, len(init_train), batch_size):
                x0_batch = init_train[i : i + batch_size]
                x_final_batch =  final_train[i : i + batch_size]
                X, U = self.rollout_policy(policy=self.policy, x_final_batch=x_final_batch, x0_batch=x0_batch, model=model)
                loss = self.cost(X, U, x_final_batch).mean()

                opt.zero_grad()
                torch.autograd.set_detect_anomaly(False)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1)
                opt.step()

                batch_losses.append(loss.item())
            train_loss = float(np.mean(batch_losses))
            train_losses.append(train_loss)

            # Validation
            self.policy.eval()
            with torch.no_grad():
                X_val, U_val = self.rollout_policy(
                    policy=self.policy,
                    x_final_batch=final_val,
                    x0_batch=init_val,
                    model=model,
                )
                val_loss = self.cost(X_val, U_val, final_val).mean().item()

            val_losses.append(val_loss)
            scheduler.step(val_loss)
            
            print(f"ep {ep}: train = {train_loss:.6f}, val = {val_loss:.6f}")

            if ep % 50 == 0 and ep != 0:
                torch.save(self.policy.state_dict(), f"NN_controllers/{self.policy_type}/policy_weights_{experiment_name}.pth")
                np.save(f"experiment_data/{self.policy_type}/val_means_{experiment_name}.npy", np.array(val_losses))
            
            if ep % 50 == 0:
                fig, ax = plt.subplots()
                ax.plot(train_losses, label="train")
                ax.plot(val_losses, label="val")
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Loss")
                ax.set_title(f"Training / Validation Loss (ep {ep})")
                ax.legend()
                fig.savefig(f"figs/{self.policy_type}/train_val_loss_{experiment_name}.png", dpi=100)
                plt.close(fig)
            
        # Final validation print
        self.policy.eval()
        with torch.no_grad():
            X_val, U_val = self.rollout_policy(
                policy=self.policy,
                x_final_batch=final_val,
                x0_batch=init_val,
                model=model,
            )
            final_val_loss = self.cost(X_val, U_val, final_val).mean().item()

        print(f"Final validation loss: {final_val_loss:.6f}")

        if self.policy_type == "GRU":
            torch.save(self.policy.state_dict(), f"gru_models/policy_weights_{experiment_name}.pth")
        elif self.policy_type == "MLP":
            torch.save(self.policy.state_dict(), f"mlp_control_models/policy_weights_{experiment_name}.pth")        
        
        np.save(f"experiment_data/{self.policy_type}/train_means_{experiment_name}.npy", np.array(train_losses))
        np.save(f"experiment_data/{self.policy_type}/val_means_{experiment_name}.npy", np.array(val_losses))
    
    
    def load_weigths(self, filename):
        self.policy.load_state_dict(torch.load(filename), strict=False)
        return
    
       

class DistributedCBF(ControllerCBF):
    def __init__(self, safety_radius, barrier_gain, barrier_type, car_number, car_index, par):
        # Initialize internal variables
        super().__init__(safety_radius, barrier_gain, barrier_type, par)
        self.car_number = car_number
        self.car_index = car_index
        self.input_old = np.zeros(2)
       
    def compute_safe_input(self, cars_state, user_inputs, car_state_final):
        # Read state
        vf_i = cars_state[self.car_index, 3]

        # Compute desired input
        vf_ref, delta_ref = self.input_conversion(cars_state[self.car_index], user_inputs[self.car_index])
        
        # Car i dynamics
        f_i, g_i = self.car_dynamics(cars_state[self.car_index])
        
        # Define optimization variables
        u_i = ca.SX.sym('u', 2)
        s_i = ca.SX.sym('s', 8)

        # Constraints variables
        constraint_list = []
        lb_constraints = []
        ub_constraints = []

        # Cost function
        Q = np.diag([10.0, 1.0])
        S = np.diag([1000.0]*8)
        u_desired = [vf_ref, delta_ref]
        cost = ca.mtimes([(u_i - u_desired).T, Q, (u_i - u_desired)]) + ca.mtimes([s_i.T, S, s_i])

        # Cars constraints
        for j in range(self.car_number):
            if j == self.car_index:
                continue
            else:
                # Car j dynamics
                f_j, g_j = self.car_dynamics(cars_state[j])
                vf_ref_j, delta_ref_j = self.input_conversion(cars_state[j], user_inputs[j])

                # Compute the gradient
                if self.barrier_type == "C3BF":
                    h_i_j, grad_h_i_j, grad_h_j_i = self.gradient_C3BF(cars_state[self.car_index], cars_state[j])
                    h_i_j_dot = 0.0
                elif self.barrier_type == "Radius":
                    h_i_j, h_i_j_dot, grad_h_i_j, grad_h_j_i = self.gradient_RadiusCBF(cars_state[self.car_index], cars_state[j])
                else:
                    raise ValueError("Unknown CBF type. Choose 'C3BF' or 'Radius'")
            
                # Build the constraint
                lf_h_i_j = np.dot(grad_h_i_j.T, f_i)
                lf_h_j_i = np.dot(grad_h_j_i.T, f_j)
                lg_h_i_j = np.dot(grad_h_i_j.T, g_i)
                lg_h_j_i = np.dot(grad_h_j_i.T, g_j)
                h_i_j_gain = self.barrier_gain

                constraint_list.append(ca.dot(lg_h_i_j.T, u_i))
                # lb_constraints.append(-h_i_j_gain*(1.0*h_i_j + 1.5*h_i_j_dot) - lf_h_i_j - (lf_h_j_i + np.dot(lg_h_j_i, [vf_ref_j, delta_ref_j])))
                lb_constraints.append(-h_i_j_gain*(1.0*h_i_j + 1.5*h_i_j_dot) - lf_h_i_j - (lf_h_j_i + np.dot(lg_h_j_i, [0.6, 0.1])))
                ub_constraints.append(ca.inf)

        # Walls constraints
        normal_vectors = np.array([[1, 0], [1, 1]/np.sqrt(2), [0, 1], [-1, 1]/np.sqrt(2), [-1, 0], [-1, -1]/np.sqrt(2), [0, -1], [1, -1]/np.sqrt(2)])
        walls_position = np.array([0, np.sqrt(2), 0, -8/np.sqrt(2), -10, -18/np.sqrt(2), -10, -8/np.sqrt(2)])

        for k in range(len(normal_vectors)):
            h_wall, h_dot_wall, grad_h_wall = self.gradient_walls(cars_state[self.car_index], normal_vectors[k], walls_position[k])
            lf_h_wall = np.dot(grad_h_wall, f_i)
            lg_h_wall = np.dot(grad_h_wall, g_i)
            h_wall_gain = self.barrier_gain

            # Build constraint
            constraint_list.append(ca.dot(lg_h_wall.T, u_i))# + s_i[k])
            lb_constraints.append(-h_wall_gain*(h_wall + 2.0*h_dot_wall) - lf_h_wall)
            ub_constraints.append(ca.inf)

            # Slack constraints
            constraint_list.append(s_i[k])
            lb_constraints.append(0.0)
            ub_constraints.append(ca.inf)

        # Input constraints
        constraint_list.append(u_i)
        lb_constraints.extend([(1.0 - 1.0/1.04)*vf_i, -self.par.steering_range/2])
        ub_constraints.extend([self.par.max_speed, self.par.steering_range/2])

        # Setup NLP
        nlp = {
            'f': cost,
            'x': ca.vertcat(u_i, s_i),
            'g': ca.vertcat(*constraint_list),
        }
        
        opts = {"ipopt.print_level": 0, "print_time": 0, "verbose": False}
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
        
        # Solve
        sol = solver(lbg=ca.vertcat(*lb_constraints), ubg=ca.vertcat(*ub_constraints))
        solution = sol['x']
        
        if not solver.stats()['success']:
            self.input_old[0] = -1.0
            self.input_old[1] = 0.0
            print(f"Optimization failed for Car {self.car_index}!")
        else:
            vf_safe = float(solution[0])
            delta_safe = float(solution[1])
            
            if vf_safe < vf_i:
                self.input_old[0] = -np.clip(-(vf_safe / (vf_i + 1e-5) - 1.0) * 1.04 * 0.6 + 0.2, 0.0, 1.0)
            else:
                self.input_old[0] = np.clip((vf_safe/self.par.max_speed) * 0.62 + 0.05, 0.0, 1.0)
            
            self.input_old[1] = delta_safe / (self.par.steering_range / 2)
        
        return self.input_old

class CentralizedCBF(ControllerCBF):
    def __init__(self, safety_radius, barrier_gain, barrier_type, car_number, coupled, par):
        super().__init__(safety_radius, barrier_gain, barrier_type, par)
        self.car_number = car_number
        self.coupled = coupled
        self.input_old = np.zeros([car_number, 2])

        self.last_solution = None
        self.last_lam_x0 = None
        self.last_lam_g0 = None

    def compute_safe_input(self, cars_state, user_inputs):
        """
        Compute safe inputs for all cars using centralized QP with C3BF constraints
        
        Args:
            cars_state: (car_number, 7) array with car states
            user_inputs: (car_number, 2) array with desired inputs
            
        Returns:
            throttle_safe: (car_number,) array of safe throttle commands
            delta_safe: (car_number,) array of safe steering commands
        """
        
        # Define optimization variables: u = [u1, u2, ...] where ui = [vf_ref, delta_ref]
        u = ca.SX.sym('u', 2 * self.car_number)
        s = ca.SX.sym('s', 8 * self.car_number)
        
        constraint_list = []
        lb_constraints = []
        ub_constraints = []
        g_labels = []
        
        # Cost function
        Q = ca.diag([10.0, 1.0])
        S = np.diag([1000.0]*8)
        cost = 0.0
        
        # Process each car
        for i in range(self.car_number):
            # Compute desired input
            vf_ref, delta_ref = self.input_conversion(cars_state[i], user_inputs[i])
            
            # Optimization variables for car i
            u_i = u[2*i:2*(i+1)]
            s_i = s[8*i:8*(i+1)]
            u_desired = ca.DM([vf_ref, delta_ref])

            # Car i dynamics
            f_i, g_i = self.car_dynamics(cars_state[i])

            start = i + 1 if self.coupled else 0

            # Cars constraints
            for j in range(start, self.car_number):
                # If decoupled, skip own constraint
                if j == i:
                    continue

                # Car j dynamics
                f_j, g_j = self.car_dynamics(cars_state[j])#  if self.coupled else (np.zeros((5,1)), np.zeros((5,2)))

                # Optimization variable
                u_j = u[2*j:2*j+2]

                # Compute the gradient
                if self.barrier_type == "C3BF":
                    h_i_j, grad_h_i_j, grad_h_j_i = self.gradient_C3BF(cars_state[i], cars_state[j])
                    h_i_j_dot = 0.0
                elif self.barrier_type == "Radius":
                    h_i_j, h_i_j_dot, grad_h_i_j, grad_h_j_i = self.gradient_RadiusCBF(cars_state[i], cars_state[j])
                else:
                    raise ValueError("Unknown CBF type. Choose 'C3BF' or 'Radius'")
                
                # Build the constraint
                lf_h_i_j = np.dot(grad_h_i_j.T, f_i)
                lf_h_j_i = np.dot(grad_h_j_i.T, f_j)
                lg_h_i_j = np.dot(grad_h_i_j.T, g_i)
                lg_h_j_i = np.dot(grad_h_j_i.T, g_j) if self.coupled else np.array([0.0, 0.0])
                h_i_j_gain = self.barrier_gain

                resp_coefficient = 1.0# if self.coupled else 0.5
                constraint_list.append(ca.dot(lg_h_i_j.T, u_i) + ca.dot(lg_h_j_i.T, u_j))
                lb_constraints.append(resp_coefficient*(-h_i_j_gain*(h_i_j + 1.7*h_i_j_dot) - (lf_h_i_j + lf_h_j_i)))
                ub_constraints.append(ca.inf)
                g_labels.append(f"CBF ({i},{j})")

            # Walls constraints
            normal_vectors = np.array([[1, 0], [1, 1]/np.sqrt(2), [0, 1], [-1, 1]/np.sqrt(2), [-1, 0], [-1, -1]/np.sqrt(2), [0, -1], [1, -1]/np.sqrt(2)])
            walls_position = np.array([0, np.sqrt(2), 0, -8/np.sqrt(2), -10, -18/np.sqrt(2), -10, -8/np.sqrt(2)])

            for k in range(len(normal_vectors)):
                h_wall, h_dot_wall, grad_h_wall = self.gradient_walls(cars_state[i], normal_vectors[k], walls_position[k])
                lf_h_wall = np.dot(grad_h_wall, f_i)
                lg_h_wall = np.dot(grad_h_wall, g_i)
                h_wall_gain = self.barrier_gain

                # Build constraint
                constraint_list.append(ca.dot(lg_h_wall.T, u_i) + s_i[k])
                lb_constraints.append(-h_wall_gain*(1.0*h_wall + 2.0*h_dot_wall) - lf_h_wall)
                ub_constraints.append(ca.inf)
                g_labels.append(f"CBF wall [{normal_vectors[k, 0]:.3f},{normal_vectors[k, 1]:.3f}]")

                # Slack constraints
                constraint_list.append(s_i[k])
                lb_constraints.append(0.0)
                ub_constraints.append(ca.inf)
                g_labels.append(f"Slack Wall {k}")

            # Input constraints
            constraint_list.append(u_i)
            lb_constraints.extend([0.0, -self.par.steering_range/2])
            ub_constraints.extend([self.par.max_speed, self.par.steering_range/2])
            g_labels.extend(["Velocity constraint", "Steering constraint"])

            # Add tracking cost
            cost += ca.mtimes([(u_i - u_desired).T, Q, (u_i - u_desired)])
            cost += ca.mtimes([s_i.T, S, s_i])

        # Setup NLP
        nlp = {
            'f': cost,
            'x': ca.vertcat(u, s),
            'g': ca.vertcat(*constraint_list),
        }
        
        opts = {"ipopt.print_level": 0, "print_time": 0, "verbose": False}
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
        
        # First guess
        x0_guess = self.last_solution if self.last_solution is not None else np.zeros((2+8)*self.car_number)
        lam_x_guess = self.last_lam_x0 if self.last_lam_x0 is not None else np.zeros((2+8)*self.car_number)
        lam_g_guess = self.last_lam_g0 if self.last_lam_g0 is not None else np.zeros(len(lb_constraints))

        # Solve
        args = {
            "lbg": ca.vertcat(*lb_constraints),
            "ubg": ca.vertcat(*ub_constraints),
            "x0": x0_guess,
            "lam_x0": lam_x_guess,
            "lam_g0": lam_g_guess
        }
        sol = solver(**args)
        solution = sol['x']
        
        if not solver.stats()['success']:
            self.input_old[:, 0] = -np.ones(self.car_number)
            self.input_old[:, 1] = -np.zeros(self.car_number)
            print("Centralized optimization failed")

            print(f"lb length: {len(lb_constraints)}")
            print(f"ub length: {len(ub_constraints)}")
            print(f"labels length: {len(g_labels)}")

            results_df = inspect_constraints(sol, lb_constraints, ub_constraints, g_labels)
            print(cars_state)
            print(results_df.to_string()) # Shows the whole table
        else:
            # Save solution
            self.last_solution = sol['x']
            self.last_lam_x0 = sol['lam_x']
            self.last_lam_g0 = sol['lam_g']

            # Input conversion
            for i in range(self.car_number):
                vf_safe = float(solution[2*i])
                delta_safe = float(solution[2*i + 1])
                
                if vf_safe < cars_state[i, 3]:
                    self.input_old[i, 0] = -np.clip(-(vf_safe / (cars_state[i, 3] + 1e-5) - 1.0) * 1.04 * 0.6 + 0.2, 0.0, 1.0)
                else:
                    self.input_old[i, 0] = np.clip((vf_safe/self.par.max_speed) * 0.62 + 0.05, 0.0, 1.0)

                self.input_old[i, 1] = delta_safe / (self.par.steering_range / 2)
        
        return self.input_old

class NMPC:
    def __init__(self, method, safety_radius, barrier_gain, horizon, sampling_time, nx, nu, car_number, car_index, par):
        self.method = method
        self.safety_radius = safety_radius
        self.barrier_gain = barrier_gain
        self.horizon = horizon
        self.sampling_time = sampling_time
        self.throttle_old = 0.0
        self.delta_old = 0.0
        self.nx = nx
        self.nu = nu
        self.car_number = car_number
        self.car_index = car_index
        self.par = par

        # Timing
        self.time_counter = 0.0
        self.max_time = 0.0

        # Define symbols
        x = ca.SX.sym('x')
        y = ca.SX.sym('y')
        theta = ca.SX.sym('theta')
        vf = ca.SX.sym('vf')
        states = ca.vertcat(x, y, theta, vf)

        vf_ref = ca.SX.sym('vf_ref')
        beta_ref = ca.SX.sym('beta_ref')
        inputs = ca.vertcat(vf_ref, beta_ref)

        # System dynamics
        delta = ca.atan2((self.par.lf+self.par.lr)/self.par.lr * ca.sin(beta_ref) , ca.cos(beta_ref))
        S_delta = ca.sqrt(ca.power(ca.cos(delta), 2) + ca.power(self.par.lr/(self.par.lf+self.par.lr)*ca.sin(delta), 2))
        vc_dot_1 = -0.32*vf + 0.379*ca.power(vf, 2) -0.155*ca.power(vf, 3) + 0.53*(vf_ref - vf)

        dyn = ca.vertcat(
            S_delta*vf * ca.cos(theta + beta_ref),
            S_delta*vf * ca.sin(theta + beta_ref),
            S_delta*vf / self.par.lr * ca.sin(beta_ref),
            vc_dot_1
        )

        simp_dyn = ca.vertcat(
            vf * ca.cos(theta),
            vf * ca.sin(theta),
            0.0,
            0.0
        )

        f = ca.Function('f', [states, inputs], [dyn])
        f_simp = ca.Function('f_simp', [states], [simp_dyn])

        # MPC optimization variables
        U = ca.SX.sym('U', self.nu, self.horizon)
        X = ca.SX.sym('X', self.car_number*self.nx, self.horizon + 1)
        S = ca.SX.sym('S', 5, self.horizon)  # Slack variables for barrier function radius + box constraints
        P = ca.SX.sym('P', self.car_number*self.nx + self.nu)  # All bumper cars states + desired_input

        # Objective
        Q = np.diag([10.0, 1.0])
        R = np.array([50.0, 100.0, 100.0, 100.0, 100.0])
        desired_input = P[self.car_number*self.nx: self.car_number*self.nx + self.nu]
        obj = 0

        # Constraints initialization
        g = []
        self.lbg = []
        self.ubg = []

        # Build dynamics and objective
        for k in range(self.horizon):
            # Optimization variables at time k
            current_state = X[:self.nx, k]
            next_state = X[:self.nx, k + 1]
            current_obs_state = X[self.nx:, k]
            next_obs_state = X[self.nx:, k + 1]
            current_input = U[:, k]
            current_slack = S[:, k]

            # Input constraints
            max_beta = ca.atan2((self.par.lf+self.par.lr)/self.par.lr * ca.sin(self.par.steering_range/2) , ca.cos(self.par.steering_range/2))
            self.lbg.extend([-self.par.max_speed, -max_beta])
            self.ubg.extend([self.par.max_speed, max_beta])
            g.append(current_input)

            # Dynamics constraint
            next_state_pred = current_state + self.sampling_time * f(current_state, current_input)
            next_obs_state_pred = current_obs_state + self.sampling_time * f_simp(current_obs_state)
            next_X_pred = ca.vertcat(next_state_pred, next_obs_state_pred)
            g.append(next_X_pred - ca.vertcat(next_state, next_obs_state))
            self.lbg.extend([0.0]*self.car_number*self.nx)
            self.ubg.extend([0.0]*self.car_number*self.nx)

            # Collision avoidance constraint
            p_rel = ca.vertcat(current_obs_state[0] - current_state[0], current_obs_state[1] - current_state[1])
            p_rel_norm = ca.sqrt(ca.dot(p_rel, p_rel))
            g.append(p_rel_norm - self.safety_radius + current_slack[0])
            self.lbg.append(0.0)
            self.ubg.append(ca.inf)

            # Slack variable constraint
            g.append(current_slack)
            self.lbg.extend([0.0]*5)
            self.ubg.extend([ca.inf]*5)

            # Box constraints
            # Xmin
            g.append(next_state[0] + current_slack[1])
            self.lbg.append(0.0)
            self.ubg.append(ca.inf)

            # Xmax
            g.append(next_state[0] - current_slack[2])
            self.lbg.append(-ca.inf)
            self.ubg.append(10.0)

            # Ymin
            g.append(next_state[1] + current_slack[3])
            self.lbg.append(0.0)
            self.ubg.append(ca.inf)

            # Ymax
            g.append(next_state[1] - current_slack[4])
            self.lbg.append(-ca.inf)
            self.ubg.append(10.0)

            # Cost
            if k == 0:
                obj += ca.mtimes([(current_input - desired_input).T, Q, (current_input - desired_input)])

            for s in range(5):
                obj += R[s] * (10*current_slack[s] + current_slack[s]**2)

        # Initial condition constraint
        g.append(X[:, 0] - P[:self.car_number*self.nx])
        self.lbg.extend([0.0]*self.car_number*self.nx)
        self.ubg.extend([0.0]*self.car_number*self.nx)

        # Optimization variables
        OPT_variables = ca.vertcat(ca.reshape(U, -1, 1), ca.reshape(X, -1, 1), ca.reshape(S, -1, 1))
        nlp_prob = {
            'f': obj,
            'x': OPT_variables,
            'g': ca.vertcat(*g),
            'p': P
        }

        opts = {'ipopt.print_level': 0, 'print_time': 0, 'record_time': True}
        self.solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

    def compute_safe_input(self, cars_state, user_input, model):
        # Read car state
        if model == "Lagrangian":
            vx_b = cars_state[self.car_index, 3]
            vy_b = cars_state[self.car_index, 4]
            omega = cars_state[self.car_index, 5]
            delta = cars_state[self.car_index, 6]

            # Compute velocities
            beta_f = np.atan2(vy_b + self.par.lf*omega + np.sin(delta)*self.par.vel_threshold, vx_b + np.cos(delta)*self.par.vel_threshold)
            vf = vx_b / np.cos(beta_f)
        elif model == "Simplified Lagrangian":
            vf = cars_state[self.car_index, 3]

        # Input conversion
        if user_input[0] < 0.0:
            u_throttle = -np.clip((-user_input[0] * 100.0 - 20.0) / (80.0 - 20.0), 0.0, 1.0)
        else:
            u_throttle = np.clip((user_input[0] * 100.0 - 5.0) / (67.0 - 5.0), 0.0, 1.0)
        if u_throttle < 0.0:
            v_ref = (1.0 + u_throttle / 1.04) * vf  # brake
        elif u_throttle / (1/self.par.max_speed) > vf:
            v_ref = u_throttle / (1/self.par.max_speed)  # throttle
        else:
            v_ref = vf  # rolling

        beta_ref = delta_to_beta(user_input[1]*self.par.steering_range/2, self.par.lf, self.par.lr)

        # Parameters
        indices = [0, 1, 2, 3] # x, y, theta, vf
        partial_cars_state = []
        for i in range(cars_state.shape[0]):
            partial_cars_state.append(cars_state[i, indices])
        partial_cars_state = np.array(partial_cars_state)
        p = np.vstack([partial_cars_state.reshape(-1,1), v_ref, beta_ref])

        # Initial guess
        u0 = np.tile([v_ref, beta_ref], (1, self.horizon))
        x0 = [partial_cars_state]
        for j in range(1, self.horizon+1, 1):
            x0.append(x0[j-1])                  # To be updated with a model prediction
        x0 = np.array(x0)
        s0 = np.zeros((5, self.horizon))

        x_init = np.vstack([u0.reshape(-1,1), x0.reshape(-1,1), s0.reshape(-1,1)])

        # Solve the optimization problem
        sol = self.solver(x0=x_init, p=p, lbg=self.lbg, ubg=self.ubg)
        solution = sol['x'][0:self.nu*self.horizon]
        u_opt_throttle = solution[::2]
        u_opt_beta = solution[1::2]
        u_opt = np.vstack([u_opt_throttle, u_opt_beta]).reshape(self.nu, self.horizon)

        # Timing
        stats = self.solver.stats()
        # print(f"Solve time: {stats['t_wall_total']}")
        self.time_counter += stats['t_wall_total']

        if stats['t_wall_total'] > self.max_time:
            self.max_time = stats['t_wall_total']
        
        if not stats['success']:
            self.throttle_old = -1.0
            self.delta_old = 0.0
            print(f"Optimization failed: Safe input [{self.throttle_old}, {self.delta_old}]")
        else:
            # Compute safe throttle
            v_ref_safe = u_opt[0, 0]
            # if v_ref_safe < vf:
            #     self.throttle_old = -np.clip(-(v_ref_safe / (vf + 1e-5) - 1.0) * 1.04 * 0.6 + 0.19, 0.0, 1.0)
            # else:
            #     self.throttle_old = np.clip((1/self.par.max_speed)*v_ref_safe * 0.62 + 0.049, 0.0, 1.0)
            if v_ref_safe < vf:
                self.throttle_old = -np.clip(-(v_ref_safe / (vf + 1e-5) - 1.0) * 1.04 * 0.6 + 0.2, 0.0, 1.0)
            else:
                self.throttle_old = np.clip((v_ref_safe/self.par.max_speed) * 0.62 + 0.05, 0.0, 1.0)

            # Compute safe delta
            delta_ref_safe = beta_to_delta(u_opt[1, 0], self.par.lf, self.par.lr)
            self.delta_old = delta_ref_safe/(self.par.steering_range/2)

        return self.throttle_old, self.delta_old