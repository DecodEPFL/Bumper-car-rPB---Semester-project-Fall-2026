import os
import sys

import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Circle, Ellipse
from matplotlib.widgets import Slider

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, BASE_DIR)

from experiment_params import getCarFinalParams, getCarInitParams, getLossParams
from config import device
from controllers import PerfBoostController
from controllers.MLP import ZeroController
from loss_functions import BumpercarLoss
from plants import BumpercarSystem, car_params
from plants.bumpercar.bumpercar_dataset import BumpercarDataset


# Add your trained pRB checkpoint here. Prefer a checkpoint with MLP weights, e.g.
# "experiments/robots/saved_results/perf_boost_XX_XX_XX_XX_XX/checkpoints/checkpoint_latest.pt"
# TRAINED_PBR_MODEL_PATH = "experiments/bumpercar/trained_pRB/controller_smoother_traj.pt"
# TRAINED_PBR_MODEL_PATH = "experiments/bumpercar/trained_pRB/controller_zero_collisions.pt"
# TRAINED_PBR_MODEL_PATH = "experiments/bumpercar/trained_pRB/checkpoint_epoch_00110 (2) copy.pt"
TRAINED_PBR_MODEL_PATH = "experiments/bumpercar/trained_pRB/checkpoint_epoch_00095.pt"


EVALUATE_MODEL = True
EVAL_HORIZON = 100
EVAL_NUM_ROLLOUTS = 100
EVAL_NUM_TEST_ROLLOUTS = 500
EVAL_RANDOM_SEED = 5

SIM_USE_GENERATED_SAMPLE = True
SIM_SAMPLE_INDEX = 62
SIM_RANDOM_SEED = 12
OBSTACLE_RADIUS = 0.625



def make_generated_sample_data(horizon, sample_index=0, random_seed=11):
    train_data, test_data = make_eval_data(
        horizon=horizon,
        num_rollouts=max(sample_index + 1, 1),
        num_test_rollouts=1,
        random_seed=random_seed,
    )

    data = train_data[sample_index:sample_index + 1]
    nx = data.shape[-1] // 2
    xbar = data[0, 0, nx:]

    return data.to(device), xbar.to(device)


def make_crossing_data(horizon, n_agents=2, ):
    nx = 7 * n_agents
    data = torch.zeros(1, horizon, 2 * nx)

    x0_bumpercar, x_final_bumpercar, _ , _, _, _ = getCarInitParams(device)
    
    data[:, 0, :nx] = x0_bumpercar
    data[:, :, nx:] = x_final_bumpercar.view(1, 1, -1)
    return data.to(device), x_final_bumpercar


def make_eval_data(horizon, num_rollouts, num_test_rollouts, random_seed):
    x0_bumpercar, _ , _ , _, car_init_radius, std_init_theta = getCarInitParams(device)
    x_final_limit, y_final_limit, final_car_min_dist = getCarFinalParams()
    
    dataset = BumpercarDataset(
        random_seed=random_seed,
        horizon=horizon,
        x0=x0_bumpercar,
        car_init_radius=car_init_radius,
        x_final_limit=x_final_limit,
        y_final_limit=y_final_limit,
        final_car_min_dist=final_car_min_dist,
        std_init_theta = std_init_theta,
        n_agents=2,
    )
    train_data, test_data = dataset.get_data(
        num_train_samples=num_rollouts,
        num_test_samples=num_test_rollouts,
    )
    return train_data.to(device), test_data.to(device)


def make_loss_fn(train_data, n_agents=2):
    _, _, obstacle_centers, obstacle_covs, _, _ = getCarInitParams(device)    
    Q, Qs, alpha_col, alpha_obst, alpha_u, position_deadzone, steady_state_velocity_radius, min_dist = getLossParams(device)
    
    return BumpercarLoss(
        Q=Q,
        Qs=Qs,
        alpha_u=alpha_u,
        xbar=train_data[0, :, 14:],
        loss_bound=None,
        sat_bound=None,
        alpha_col=alpha_col,
        alpha_obst=alpha_obst,
        obstacle_centers=obstacle_centers,
        obstacle_covs=obstacle_covs,
        min_dist=min_dist,
        n_agents=n_agents,
        position_deadzone=position_deadzone,
        steady_state_velocity_radius=steady_state_velocity_radius,
    )


def obstacle_collision_indices(x_log, obstacle_centers, radius=OBSTACLE_RADIUS, n_agents=2):
    collision_indices = set()
    radius_sq = radius ** 2

    for agent_idx in range(n_agents):
        base = 7 * agent_idx
        positions = x_log[:, :, base:base + 2]

        for obstacle_idx, center in enumerate(obstacle_centers):
            center = center.to(device=positions.device, dtype=positions.dtype).view(1, 1, 2)
            distance_sq = torch.sum((positions - center) ** 2, dim=-1)
            colliding = distance_sq <= radius_sq

            for rollout_idx, _ in colliding.nonzero(as_tuple=False):
                collision_indices.add(int(rollout_idx))

    return sorted(collision_indices)


def checkpoint_args(checkpoint):
    args = checkpoint.get("args", {})
    return args if isinstance(args, dict) else vars(args)


def load_controller(system, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    args = checkpoint_args(checkpoint)

    controller = PerfBoostController(
        noiseless_forward=system.noiseless_forward,
        input_init=system.x_init,
        output_init=system.u_init,
        dim_internal=args.get("dim_internal", 8),
        dim_nl=args.get("dim_nl", 8),
        initialization_std=args.get("cont_init_std", 0.1),
        output_amplification=1,
        ren_internal_state_init=None,
    ).to(device)

    if "controller_state_dict" in checkpoint:
        controller.load_state_dict(checkpoint["controller_state_dict"], strict=False)
    elif "ren_state_dict" in checkpoint:
        controller.c_ren.load_state_dict(checkpoint["ren_state_dict"], strict=False)
        if "mlp_state_dict" in checkpoint:
            controller.MLP.load_state_dict(checkpoint["mlp_state_dict"], strict=False)
    else:
        ren_state = {
            key: value for key, value in checkpoint.items()
            if key in controller.c_ren.state_dict()
        }
        controller.c_ren.load_state_dict(ren_state, strict=False)

    controller.eval()
    controller.reset()
    return controller


def evaluate_controller():
    if not TRAINED_PBR_MODEL_PATH:
        raise ValueError("Set TRAINED_PBR_MODEL_PATH before evaluating.")

    system = BumpercarSystem(
        params=car_params,
        x_init=None,
        u_init=None,
    ).to(device)
    controller = load_controller(system, TRAINED_PBR_MODEL_PATH)
    train_data, test_data = make_eval_data(
        horizon=EVAL_HORIZON,
        num_rollouts=EVAL_NUM_ROLLOUTS,
        num_test_rollouts=EVAL_NUM_TEST_ROLLOUTS,
        random_seed=EVAL_RANDOM_SEED
    )
    loss_fn = make_loss_fn(train_data, n_agents=system.n_agents)

    print(f"[INFO] evaluating {TRAINED_PBR_MODEL_PATH}")
    _, _, obstacle_centers, _, _, _ = getCarInitParams(device)
    with torch.no_grad():
        x_log, e_log, u_log = system.rollout(controller, train_data, train=False)
        train_loss = loss_fn.forward(x_log, u_log, e_log).item()
        train_collisions = loss_fn.count_collisions(x_log)
        train_obstacle_collision_indices = obstacle_collision_indices(
            x_log,
            obstacle_centers,
            n_agents=system.n_agents,
        )

        x_log, e_log, u_log = system.rollout(controller, test_data, train=False)
        test_loss = loss_fn.forward(x_log, u_log, e_log).item()
        test_collisions = loss_fn.count_collisions(x_log)
        test_obstacle_collision_indices = obstacle_collision_indices(
            x_log,
            obstacle_centers,
            n_agents=system.n_agents,
        )

    print(f"Train loss: {train_loss:.4f} -- Number of collisions = {train_collisions:.0f}")
    print(f"Test loss: {test_loss:.4f} -- Number of collisions = {test_collisions:.0f}")
    print(f"Train obstacle collision indices: {train_obstacle_collision_indices}")
    print(f"Test obstacle collision indices: {test_obstacle_collision_indices}")


def simulate(horizon=400, use_generated_sample=SIM_USE_GENERATED_SAMPLE, sample_index=SIM_SAMPLE_INDEX):
    system = BumpercarSystem(
        params=car_params,
        x_init=None,
        u_init=None,
    ).to(device)

    if TRAINED_PBR_MODEL_PATH:
        controller = load_controller(system, TRAINED_PBR_MODEL_PATH)
        title = "Trained pRB controller"
    else:
        controller = ZeroController(ref_dim=2 * system.n_agents).to(device)
        title = "PID only - set TRAINED_PBR_MODEL_PATH to use pRB"

    if use_generated_sample:
        data, xbar = make_generated_sample_data(
            horizon=horizon,
            sample_index=sample_index,
            random_seed=SIM_RANDOM_SEED,
        )
        title = f"{title} - generated sample {sample_index}"
    else:
        data, xbar = make_crossing_data(horizon=horizon, n_agents=system.n_agents)
        title = f"{title} - fixed crossing"

    with torch.no_grad():
        x_log, _, _ = system.rollout(controller, data, train=False)

    _, _, obstacle_centers, _, _, _ = getCarInitParams(device)
    sim_obstacle_collision_indices = obstacle_collision_indices(
        x_log,
        obstacle_centers,
        n_agents=system.n_agents,
    )
    # print(f"Sim obstacle collision indices: {sim_obstacle_collision_indices}")

    return x_log[0].detach().cpu(), xbar.cpu(), title


def draw_car(ax, x, y, theta, color):
    length = 0.55
    width = 0.32
    dx = torch.tensor([length / 2, length / 2, -length / 2, -length / 2])
    dy = torch.tensor([width / 2, -width / 2, -width / 2, width / 2])
    c = torch.cos(theta)
    s = torch.sin(theta)
    px = x + c * dx - s * dy
    py = y + s * dx + c * dy
    return ax.fill(px, py, color=color, alpha=0.75, edgecolor="black", linewidth=1.0)[0]


def draw_obstacles(ax, obstacle_centers, obstacle_covs):
    for center, _ in zip(obstacle_centers, obstacle_covs):
        center = center.detach().cpu().flatten()

        circle = Circle(
            xy=(center[0].item(), center[1].item()),
            radius=OBSTACLE_RADIUS,
            facecolor="0.35",
            edgecolor="black",
            alpha=0.25,
            linewidth=1.0,
            zorder=0,
        )
        ax.add_patch(circle)


def show_simulation():
    x_log, xbar, title = simulate()
    _, _, obstacle_centers, obstacle_covs, _, _ = getCarInitParams(device)
    n_agents = 2
    colors = ["tab:blue", "tab:orange"]

    fig, ax = plt.subplots(figsize=(7, 7))
    plt.subplots_adjust(bottom=0.16)
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-4.0, 6.0)
    ax.grid(True, alpha=0.3)
    draw_obstacles(ax, obstacle_centers, obstacle_covs)

    path_lines = []
    car_patches = []
    target_markers = []
    start_markers = []

    for i in range(n_agents):
        base = 7 * i
        color = colors[i]
        (path_line,) = ax.plot([], [], color=color, linewidth=2)
        path_lines.append(path_line)
        car_patches.append(draw_car(ax, x_log[0, base], x_log[0, base + 1], x_log[0, base + 2], color))
        target_markers.append(ax.plot(xbar[base], xbar[base + 1], marker="*", markersize=14, color=color)[0])
        start_markers.append(ax.plot(x_log[0, base], x_log[0, base + 1], marker="o", markersize=7, color=color, fillstyle="none")[0])

    time_text = ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top")

    slider_ax = fig.add_axes([0.15, 0.05, 0.72, 0.035])
    time_slider = Slider(
        ax=slider_ax,
        label="time",
        valmin=0,
        valmax=x_log.shape[0] - 1,
        valinit=0,
        valstep=1,
    )

    def update(frame):
        t = int(frame)
        for i in range(n_agents):
            base = 7 * i
            path_lines[i].set_data(x_log[:t + 1, base], x_log[:t + 1, base + 1])
            car_patches[i].remove()
            car_patches[i] = draw_car(
                ax,
                x_log[t, base],
                x_log[t, base + 1],
                x_log[t, base + 2],
                colors[i],
            )

        dist = torch.linalg.norm(x_log[t, 0:2] - x_log[t, 7:9]).item()
        time_text.set_text(f"step {t}   distance {dist:.2f} m")
        fig.canvas.draw_idle()

    time_slider.on_changed(update)
    update(0)
    plt.show()


if __name__ == "__main__":
    if EVALUATE_MODEL:
        evaluate_controller()
    show_simulation()
