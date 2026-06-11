import numpy as np
import torch
from pathlib import Path

from algorithms import NN_Controller
from parameters import params
from mlp_model import MLPModel


# ---------- Helpers ----------

def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def create_grid_states(goal_state):
    grid_size = 8
    spacing = 1.0
    headings = [0.0, np.pi/2, np.pi, 3*np.pi/2]

    half = grid_size // 2
    offsets = np.arange(-half, half + 1) * spacing

    init_states = []

    for dx in offsets:
        for dy in offsets:
            for theta in headings:
                if not (np.linalg.norm(np.array([dx,dy])) <= 1):
                    s = goal_state.copy()
                    s[0] += dx
                    s[1] += dy
                    s[2] = theta
                    init_states.append(s)

    cars_state_init = np.array(init_states, dtype=np.float32)
    goal_states = np.repeat(goal_state[None, :], len(init_states), axis=0)

    return cars_state_init, goal_states


# ---------- Controller ----------

def compute_control_batch(controller, cars_state, goal_states):
    car_number = cars_state.shape[0]
    safe_input = np.zeros((car_number, 2), dtype=np.float32)

    for i in range(car_number):
        cars_state_t = torch.tensor(cars_state[i], dtype=torch.float32)
        cars_state_final_t = torch.tensor(goal_states[i], dtype=torch.float32)

        error = cars_state_final_t - cars_state_t

        dx_i = error[0]
        dy_i = error[1]
        theta = torch.tensor(cars_state[i, 2], dtype=torch.float32)

        dx_local = torch.cos(theta) * dx_i + torch.sin(theta) * dy_i
        dy_local = -torch.sin(theta) * dx_i + torch.cos(theta) * dy_i

        sin_dtheta = torch.sin(error[2])
        cos_dtheta = torch.cos(error[2])

        error_body = torch.cat([
            dx_local.unsqueeze(0).unsqueeze(0),
            dy_local.unsqueeze(0).unsqueeze(0),
            sin_dtheta.unsqueeze(0).unsqueeze(0),
            cos_dtheta.unsqueeze(0).unsqueeze(0),
            error[3:].unsqueeze(0)
        ], dim=1)

        with torch.no_grad():
            u = controller.policy(error_body)

        safe_input[i, :] = u[0].detach().cpu().numpy()

    return safe_input


# ---------- Simulator ----------

def simulate_one_step_batch(models, cars_state, cars_state_init, safe_input, t):
    car_number = cars_state.shape[0]

    for i in range(car_number):
        if t == 0:
            c_state = torch.tensor(cars_state_init[i], dtype=torch.float32).unsqueeze(0)
        else:
            c_state = torch.tensor(cars_state[i], dtype=torch.float32).unsqueeze(0)

        c_input = torch.tensor(safe_input[i, :], dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            next_state = models[i].update(
                x_state=c_state,
                car_input=c_input
            ).detach().cpu().numpy().squeeze()

        cars_state[i] = next_state

    return cars_state


# ---------- Metrics ----------

def compute_metrics(cars_state, goal_states, threshold):
    threshold_pos, threshold_theta = threshold
    err = np.zeros((cars_state.shape[0], 3), dtype=np.float32)

    err[:, 0] = goal_states[:, 0] - cars_state[:, 0]
    err[:, 1] = goal_states[:, 1] - cars_state[:, 1]

    for i in range(cars_state.shape[0]):
        err[i, 2] = wrap_angle(goal_states[i, 2] - cars_state[i, 2])

    err_norm_pos = np.linalg.norm(err[:, 0:2], axis=1)
    err_theta = err[:, 2]

    success_pos = err_norm_pos <= threshold_pos
    success_theta = err_theta <= threshold_theta

    success_count_pos = int(np.sum(success_pos))
    success_count_theta = int(np.sum(success_theta))
    
    # success_rate_pos = success_count_pos / len(err_norm)
    
    rmse_pos = float(np.sqrt(np.mean(err_norm_pos**2)))
    rmse_theta = float(np.sqrt(np.mean(err_theta**2)))

    return {
        "success_count_pos": success_count_pos,
        "success_count_angle": success_count_theta,
        "rmse_pos": rmse_pos,
        "rmse_theta": rmse_theta,
        "mean_error_pos": float(np.mean(err_norm_pos)),
        "mean_abs_error_theta": float(np.mean(np.abs(err_theta))),
        "max_error_pos": float(np.max(err_norm_pos)),
        "max_error_theta": float(np.max(err_theta)),
    }


def evaluate_model(type, model_path):
    goal_state = np.array([5.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    # Create grid
    cars_state_init, goal_states = create_grid_states(goal_state)
    cars_state = cars_state_init.copy()
    # cars_state_init[:, 3] = 0.5

    car_number = cars_state.shape[0]
    print(f"Number of cars: {car_number}") 

    # Init controller
    controller = NN_Controller(type=type, params=params)
    controller.load_weigths(model_path)
    # Init models per car
    models = [MLPModel(cars_state_init[i], params) for i in range(car_number)]

    # Simulation settings
    sim_time = 18.0
    dt = 0.1
    n_steps = int(sim_time / dt)

    # Simulation loop
    for t in range(n_steps):
        safe_input = compute_control_batch(controller, cars_state, goal_states)
        cars_state = simulate_one_step_batch(models, cars_state, cars_state_init, safe_input, t)

    # Evaluate
    
    threshold = [0.2, 5 / 360 * 2*np.pi] #norm of pos_error = 0.2, 5 degree error
    metrics = compute_metrics(cars_state, goal_states, threshold)

    print("\n=== Evaluation Results ===")
    print(f"Success (pos / angle): {metrics['success_count_pos']} / {metrics['success_count_angle']}")
    print(f"RMSE   (pos / theta):  {metrics['rmse_pos']:.4f} / {metrics['rmse_theta']:.4f}")
    print(f"Mean   (pos / theta):  {metrics['mean_error_pos']:.4f} / {metrics['mean_abs_error_theta']:.4f}")
    print(f"Max    (pos / theta):  {metrics['max_error_pos']:.4f} / {metrics['max_error_theta']:.4f}")

# ---------- Main ----------

if __name__ == "__main__":

    model_dir = Path("NN_controllers/MLP")
    # model_dir = Path("NN_controllers/GRU")

    for path in model_dir.glob("*.pth"):
        print(f"\nEvaluating: {path.name}")
        
        try:
            evaluate_model("MLP", str(path))
        except Exception as e:
            print(f"Failed on {path.name}: {e}")