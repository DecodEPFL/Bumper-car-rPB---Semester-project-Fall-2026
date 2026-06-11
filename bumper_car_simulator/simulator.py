#!/usr/bin/env python3
import numpy as np
import torch
np.set_printoptions(precision=2, suppress=True)
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
import random

from algorithms import DistributedCBF, CentralizedCBF, NMPC, PositionPidController, NN_Controller, PolicyMLP
from lagrangian_model import LagrangianModel
from mlp_model import MLPModel
from parameters import params
from utils import normalize_angle
from trainNN import gaussianCarDistribution
matplotlib.use('macosx')
from evaluateNN import create_grid_states

# --------------------------------------------
# ------------ 1. Configuration --------------
# --------------------------------------------
# Simulation variables
car_number = 10
simulation_horizon = 100
model = "MLP"  # "Lagrangian" or "MLP"

# Controller variables
barrier_type = "Radius" # "Radius" or "C3BF"
centralized = False
coupled = True
random_input = False
target_center = False

# Plot variables
plot_fig = True
full_trajectory = False

animate = True

# Car initialization: state = (x, y, theta, vf, beta_f, beta_r, delta)
car0_state_init = np.array([5.0, 5.0, 0, 3.0, 0.1, 0.3, 0.5])
car_state_final = np.array([5.0, 5.0, 0, 0, 0.0, 0.0, 0.0])
car0_input_init = np.array([0.0, 0.0])
input_size = car0_input_init.shape[0]
state_size = car0_state_init.shape[0]
cars_state_init = np.array([car0_state_init])
cars_state_final = np.array([car_state_final])


# Birger: Initialize models
std_init = 10
std_final = 0 

INIT_TYPE = "GAUSSIAN" ## "GAUSSIAN" or "GRID"
if INIT_TYPE == "GAUSSIAN": 
    cars_state_init, cars_state_final = gaussianCarDistribution(car0_state_init, car_state_final, std_init, std_final, car_number)
    cars_state = cars_state_init

elif INIT_TYPE == "GRID":
    cars_state_init, cars_state_final = create_grid_states(car_state_final)
    cars_state = cars_state_init
    car_number = cars_state_init.shape[0]

else:
    assert("Not valid initialization")

# Input generation
cars_input_init = np.array([car0_input_init])
cars_input = np.zeros([car_number, input_size, simulation_horizon])
cars_input[:, :, 0] = cars_input_init[:car_number]
safe_input = cars_input[:, :, 0]



models = []
if model == "Lagrangian":
    print("Using Lagrangian model")
    for i in range(car_number):
        models.append(LagrangianModel(cars_state_init[i], params))
elif model == "MLP":
    print("Using MLP model")
    for i in range(car_number):
        models.append(MLPModel(cars_state_init[i], params))
else:
    raise ValueError("Unknown model type. Choose 'Lagrangian' or 'MLP'.")


# Plot variables
state_traj = np.zeros([car_number, state_size, simulation_horizon])
input_traj = np.zeros([car_number, input_size, simulation_horizon])
state_traj[..., 0] = np.array(cars_state_init[:car_number])
input_traj[..., 0] = np.array(cars_input_init[:car_number])


# Controller parameters
safety_radius = 1.7
barrier_gain = 1.0
horizon_NMPC = 20
time_step = 0.1
internal_state_dim = 4
internal_input_dim = 2

# Distributed ontrollers
NMPCs = []
C3BFs = []
for j in range(car_number):
    # NMPCs.append(NMPC("C3BF", safety_radius, barrier_gain, horizon_NMPC, time_step, internal_state_dim, internal_input_dim, car_number, j, params))
    C3BFs.append(DistributedCBF(safety_radius, barrier_gain, barrier_type, car_number, j, params))

# Centralized controller
CentralizedController = CentralizedCBF(safety_radius, barrier_gain, barrier_type, car_number, coupled, params)


# Birger: Init controllers
kp = 1
PidController = PositionPidController(
    kp_p=1,     kd_p=0,     ki_p=0,  # P,I,D, distance
    kp_theta=1*kp, kd_theta=0.2*kp, ki_theta=0.0,
    params=params, # P,I,D, theta
)

nn_controller = NN_Controller(type="GRU", params=params)
nn_controller.load_weigths("NN_controllers/GRU/policy_weights_GRU_50_cars.pth")
nn_controller.policy.eval()
gru_hidden = [None for _ in range(len(cars_state))]

# mlp_controller.load_weigths("mlp_control_models/policy_weights_1000cars_preTrain.pth")
# mlp_controller.load_weigths("mlp_control_models/policy_weights_2000cars_preTrain.pth")
# mlp_controller.load_weigths("mlp_control_models/policy_weights_2000cars.pth")
# mlp_controller.load_weigths("mlp_control_models/policy_weights_200cars.pth")
# mlp_controller.load_weigths("mlp_control_models/policy_weight_BEST_final_body_with_orientation.pth")

usePidController = False
useMlp = False
useGRU = True

if sum([usePidController, useMlp, useGRU]) != 1:
    raise ValueError("Exactly one of usePidController, useMlp, or useGRU must be True")

# --------------------------------------------
# ------------ 2. Simulation -----------------
# --------------------------------------------
for t in range(simulation_horizon):
    print(f"Simulation time {t*0.1:.2f}")

    # Store data
    for i in range(car_number):
        # if cars_state[i, 3] < 0.01: # If the car is almost stopped, point it towards the center to encourage movement
        #     if target_center:
        #         angle_to_center = np.arctan2(5.0 - models[si].car_state[1], 5.0 - models[i].car_state[0])
        #         alpha = normalize_angle(angle_to_center - models[i].car_state[2])
        #         cars_input[i, 1, t] = np.clip(alpha/(params.steering_range/2), -1.0, 1.0)

        if (cars_state[i][0] < 1.0) or (cars_state[i][0] > 9.0) or (cars_state[i][1] < 1.0) or (cars_state[i][1] > 9.0):
            dx = max(cars_state[i][0] - 9, 0) - min(cars_state[i][0] - 1, 0)
            dy = max(cars_state[i][1] - 9, 0) - min(cars_state[i][1] - 1, 0)
            print(f"Car {i} out of bounds! Violation: {np.sqrt(dx**2 + dy**2)}")

    # Compute safe input
        if usePidController:
            safe_input[0, :] = PidController.calculateControlInput(
                p_final=car_state_final,
                car_state=cars_state[i],
                dt=time_step,)
            
        elif useMlp or useGRU:
            cars_state_t = torch.tensor(cars_state[i], dtype=torch.float32)
            cars_state_final_t = torch.tensor(car_state_final, dtype=torch.float32)
            error = cars_state_final_t - cars_state_t 

            dx_i = error[0]
            dy_i = error[1]
            theta = torch.tensor(cars_state[i, 2], dtype=torch.float32)

            # Rotate into body frame
            dx_local = torch.cos(theta) * dx_i + torch.sin(theta) * dy_i
            dy_local = -torch.sin(theta) * dx_i + torch.cos(theta) * dy_i
            # Angle error wrapped
            sin_dtheta = torch.sin(error[2])
            cos_dtheta = torch.cos(error[2])

            error_body = torch.cat([
                dx_local.unsqueeze(0).unsqueeze(0),
                dy_local.unsqueeze(0).unsqueeze(0),
                sin_dtheta.unsqueeze(0).unsqueeze(0),
                cos_dtheta.unsqueeze(0).unsqueeze(0),
                error[3:].unsqueeze(0)   
            ], dim=1) 
            
            
            if useMlp:
                policy_input = error_body
                u = nn_controller.policy(policy_input)
                safe_input[i, :] = u[0].detach().cpu().numpy()
            elif useGRU:
                with torch.no_grad():
                    policy_input = error_body.unsqueeze(1)  # (1, 1, 8)

                    u, h = nn_controller.policy(
                        policy_input,
                        h0=gru_hidden[i],
                        return_hidden=True,
                    )

                    gru_hidden[i] = h.detach()
                    safe_input[i, :] = u[0].detach().cpu().numpy()

        elif centralized:     
            safe_input = CentralizedController.compute_safe_input(cars_state, cars_input[:, :, t])
        else:
            for i in range(car_number):
                safe_input[i, :] = C3BFs[i].compute_safe_input(cars_state, cars_input[:, :, t], car_state_final)

    # Update state
    for i in range(car_number):
        if model == "MLP":
            if t == 0:
                c_state = torch.tensor(cars_state_init[i], dtype=torch.float32).unsqueeze(0)
            else:
                c_state = torch.tensor(cars_state[i], dtype=torch.float32).unsqueeze(0)

            c_input = torch.tensor(safe_input[i, :], dtype=torch.float32).unsqueeze(0)
            next_state = models[i].update(x_state=c_state, car_input=c_input).detach().cpu().numpy().squeeze()
            cars_state[i] = next_state
            state_traj[i, :, t] = next_state
            
        elif model == "Lagrangian":
            models[i].update(safe_input[i, :])
            state_traj[i, :, t] = models[i].car_state
            cars_state[i] = models[i].car_state



# --------------------------------------------
# ------------ 3. Plot the results -----------
# --------------------------------------------
if plot_fig:
    if full_trajectory:
        fig, (ax_full, ax_dyn) = plt.subplots(nrows=1, ncols=2, figsize=(6, 3), dpi=300)
    else:
        fig, ax_dyn = plt.subplots(nrows=1, ncols=1, figsize=(4, 4), dpi=300)

    plt.subplots_adjust(bottom=0.25)
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

    cars_dots = []
    cars_orient_arrows = []
    cars_traj_lines = []

    arrow_length = 0.35     # make this smaller/larger as you like
    dot_size = 0.5            # marker size
    traj_linewidth = 0.8

    # Initialization
    for i in range(car_number):
        if full_trajectory:
            # Full trajectory panel
            ax_full.plot(
                state_traj[i, 0, :],
                state_traj[i, 1, :],
                color=colors[i % 7],
                alpha=0.5,
                linewidth=0.8
            )
            ax_full.plot(
                state_traj[i, 0, simulation_horizon - 1],
                state_traj[i, 1, simulation_horizon - 1],
                colors[i % 7] + 'x',
                markersize=4
            )

        # Goal position
        ax_dyn.plot(
            car_state_final[0],
            car_state_final[1],
            colors[i % 7] + 'o',
            markersize=2,
            markerfacecolor='none'
        )

        # Dot for current position
        dot = ax_dyn.plot([], [], 'o', color=colors[i % 7], markersize=dot_size)[0]
        cars_dots.append(dot)

        # Trajectory line
        traj_line = ax_dyn.plot([], [], color=colors[i % 7], lw=traj_linewidth, alpha=0.8)[0]
        cars_traj_lines.append(traj_line)

        # Small orientation arrow
        # orient_arrow = ax_dyn.arrow(
        #     0, 0, 0, 0,
        #     color=colors[i % 7],
        #     width=0.008,
        #     head_width=0.08,
        #     head_length=0.10,
        #     length_includes_head=True
        # )
        # cars_orient_arrows.append(orient_arrow)

    # Figure settings
    if full_trajectory:
        ax_full.set_xlim([-1, 11])
        ax_full.set_ylim([-1, 11])
        ax_full.set_title("Full Trajectory")
        ax_full.set_aspect('equal', adjustable='box')
        # ax_full.hlines(y=[0, 10], xmin=0, xmax=10, colors=['k'])
        # ax_full.vlines(x=[0, 10], ymin=0, ymax=10, colors=['k'])
        # ax_full.hlines(y=[1, 9], xmin=1, xmax=9, colors=['r'], alpha=0.5)
        # ax_full.vlines(x=[1, 9], ymin=1, ymax=9, colors=['r'], alpha=0.5)

    ax_dyn.set_xlim([-1, 11])
    ax_dyn.set_ylim([-1, 11])
    ax_dyn.set_title("Current State")
    ax_dyn.set_aspect('equal', adjustable='box')
    ax_dyn.hlines(y=[0, 10], xmin=0, xmax=10, colors=['k'])
    ax_dyn.vlines(x=[0, 10], ymin=0, ymax=10, colors=['k'])
    ax_dyn.hlines(y=[1, 9], xmin=1, xmax=9, colors=['r'], alpha=0.5)
    ax_dyn.vlines(x=[1, 9], ymin=1, ymax=9, colors=['r'], alpha=0.5)

    
    ax_slider = plt.axes([0.25, 0.1, 0.5, 0.03])
    slider = Slider(ax_slider, 'Time', 0, 6 * simulation_horizon - 1, valinit=0, valstep=1)

    def update(frame):
        t = int(frame)

        real_t = t / 60.0 - 0.001

        # Find which two data samples we are between
        idx_low = max(int(np.floor(real_t / 0.1)), 0)
        idx_high = min(idx_low + 1, simulation_horizon - 1)

        # Linear interpolation factor
        alpha = np.mod(real_t, 0.1) / 0.1

        # Remove old arrows before drawing new ones
        # for arr in cars_orient_arrows:
        #     arr.remove()
        # cars_orient_arrows.clear()

        artists = []

        for i in range(car_number):
            # Interpolated position
            x_i = state_traj[i, 0, idx_low] + alpha * (state_traj[i, 0, idx_high] - state_traj[i, 0, idx_low])
            y_i = state_traj[i, 1, idx_low] + alpha * (state_traj[i, 1, idx_high] - state_traj[i, 1, idx_low])

            # Interpolated angle, avoiding wrap jump
            if abs(state_traj[i, 2, idx_low] - state_traj[i, 2, idx_high]) < np.pi / 2:
                theta_i = state_traj[i, 2, idx_low] + alpha * (state_traj[i, 2, idx_high] - state_traj[i, 2, idx_low])
            else:
                theta_i = state_traj[i, 2, idx_low] + alpha * (
                    np.sign(state_traj[i, 2, idx_low]) * 2 * np.pi
                    + state_traj[i, 2, idx_high]
                    - state_traj[i, 2, idx_low]
                )

            # Update dot
            cars_dots[i].set_data([x_i], [y_i])

            # Update trajectory line up to current point
            x_hist = list(state_traj[i, 0, :idx_low + 1])
            y_hist = list(state_traj[i, 1, :idx_low + 1])

            # append interpolated current point
            if idx_high > idx_low:
                x_hist.append(x_i)
                y_hist.append(y_i)

            cars_traj_lines[i].set_data(x_hist, y_hist)

            # Draw small orientation arrow
            # dx = arrow_length * np.cos(theta_i)
            # dy = arrow_length * np.sin(theta_i)

            # orient_arrow = ax_dyn.arrow(
            #     x_i, y_i, dx, dy,
            #     color=colors[i % 7],
            #     width=0.008,
            #     head_width=0.08,
            #     head_length=0.10,
            #     length_includes_head=True
            # )
            # cars_orient_arrows.append(orient_arrow)

            # artists.extend([cars_dots[i], cars_traj_lines[i], orient_arrow])

        ax_dyn.set_title("Current State")
        slider.valtext.set_text(f'{real_t:.2f} s')

        return artists

    slider.on_changed(update)
    update(0)

    if animate:
        ani = FuncAnimation(
            fig,
            update,
            frames= 4* simulation_horizon - 1,
            interval= t / 5.0 - 0.001,
            blit=False
        )
        writer = PillowWriter(fps=50)
        ani.save("car_trip.gif", writer=writer, dpi=200)
        ani.event_source.stop()
        del ani

    plt.show()