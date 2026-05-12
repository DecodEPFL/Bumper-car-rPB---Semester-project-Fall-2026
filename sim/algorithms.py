#!/usr/bin/env python3
import numpy as np
import casadi as ca
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from mlp_model import MLPModel
from utils import beta_to_delta, delta_to_beta


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
                torch.save(self.policy.state_dict(), f"sim/NN_controllers/{self.policy_type}/policy_weights_{experiment_name}.pth")
                np.save(f"sim/experiment_data/{self.policy_type}/val_means_{experiment_name}.npy", np.array(val_losses))
            
            if ep % 50 == 0:
                fig, ax = plt.subplots()
                ax.plot(train_losses, label="train")
                ax.plot(val_losses, label="val")
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Loss")
                ax.set_title(f"Training / Validation Loss (ep {ep})")
                ax.legend()
                fig.savefig(f"sim/figs/{self.policy_type}/train_val_loss_{experiment_name}.png", dpi=100)
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
            torch.save(self.policy.state_dict(), f"sim/NN_controllers/GRU/policy_weights_{experiment_name}.pth")
        elif self.policy_type == "MLP":
            torch.save(self.policy.state_dict(), f"sim/NN_controllers/MLP/policy_weights_{experiment_name}.pth")        
        
        np.save(f"sim/experiment_data/{self.policy_type}/train_means_{experiment_name}.npy", np.array(train_losses))
        np.save(f"sim/experiment_data/{self.policy_type}/val_means_{experiment_name}.npy", np.array(val_losses))
    
    
    def load_weigths(self, filename):
        self.policy.load_state_dict(torch.load(filename), strict=False)
        return