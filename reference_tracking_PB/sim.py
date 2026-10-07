#!/usr/bin/env python3
"""
Simulation and benchmark evaluation harness for 2 autonomous bumper cars.
Evaluates the trained pRB controller with the integrated CBF-QP safety filter.
Uses physical obstacle radius (0.80 m) and balanced inter-vehicle distance (2.05 m).
"""

import os
import sys
import csv

from matplotlib import animation
import matplotlib.pyplot as plt
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, Polygon
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


# Model checkpoint path
TRAINED_PBR_MODEL_PATH = os.path.join(
    BASE_DIR, "experiments/bumpercar/trained_pRB/rPB_best.pt"
)

EVALUATE_MODEL = True
EVAL_HORIZON = 500
EVAL_NUM_ROLLOUTS = 100
EVAL_NUM_TEST_ROLLOUTS = 500
EVAL_RANDOM_SEED = 2

SIM_USE_GENERATED_SAMPLE = True
SIM_DATA_SPLIT = "train"
SIM_SAMPLE_INDEX = 12
SIM_RANDOM_SEED = EVAL_RANDOM_SEED

# Corrected physical collision radius (0.45 m obstacle + 0.35 m car footprint = 0.80 m)
# Restores the physical 1.15 m open corridor between c1 (-1.375) and c2 (+1.375)
OBSTACLE_RADIUS = 0.80

# Baseline inter-vehicle distance preserving central crossing
CAR_SAFE_DISTANCE = 2.05

REPORT_FIGURE_PATH = os.path.join(
    BASE_DIR, "experiments/bumpercar/report_trajectory.svg"
)
REPORT_COLLISION_FIGURE_PATH = os.path.join(
    BASE_DIR, "experiments/bumpercar/report_collision.svg"
)
REPORT_INIT_FIGURE_PATH = os.path.join(
    BASE_DIR, "experiments/bumpercar/report_setup.svg"
)
REPORT_SNAPSHOT_FIGURE_PATH = os.path.join(
    BASE_DIR, "experiments/bumpercar/report_snapshots.svg"
)
TRAJECTORY_GIF_PATH = os.path.join(
    BASE_DIR, "experiments/bumpercar/trajectory.gif"
)
ROLLOUT_METRICS_PATH = os.path.join(
    BASE_DIR, "experiments/bumpercar/rollout_metrics.csv"
)
REPORT_FINAL_RADIUS = 1.0

DT = 0.04


def make_generated_sample_data(
    horizon: int,
    sample_index: int = 0,
    random_seed: int = 11,
    split: str = "train",
):
    train_data, test_data = make_eval_data(
        horizon=horizon,
        num_rollouts=max(sample_index + 1, 1),
        num_test_rollouts=max(sample_index + 1, 1),
        random_seed=random_seed,
    )

    if split == "train":
        source_data = train_data
    elif split == "test":
        source_data = test_data
    else:
        raise ValueError(f"Unknown generated sample split: {split}")

    data = source_data[sample_index : sample_index + 1]
    nx = data.shape[-1] // 2
    xbar = data[0, 0, nx:]

    return data.to(device), xbar.to(device)


def make_crossing_data(horizon: int, n_agents: int = 2):
    nx = 7 * n_agents
    data = torch.zeros(1, horizon, 2 * nx)

    x0_bumpercar, x_final_bumpercar, _, _, _, _ = getCarInitParams(device)

    data[:, 0, :nx] = x0_bumpercar
    data[:, :, nx:] = x_final_bumpercar.view(1, 1, -1)

    return data.to(device), x_final_bumpercar


def make_eval_data(
    horizon: int,
    num_rollouts: int,
    num_test_rollouts: int,
    random_seed: int,
):
    x0_bumpercar, x_final_bumpercar, _, _, car_init_radius, std_init_theta = (
        getCarInitParams(device)
    )
    x_final_limit, y_final_limit, final_car_min_dist = getCarFinalParams()

    dataset = BumpercarDataset(
        random_seed=random_seed,
        horizon=horizon,
        x0=x0_bumpercar,
        x_final=x_final_bumpercar,
        car_init_radius=car_init_radius,
        x_final_limit=x_final_limit,
        y_final_limit=y_final_limit,
        final_car_min_dist=final_car_min_dist,
        std_init_theta=std_init_theta,
        n_agents=2,
    )
    train_data, test_data = dataset.get_data(
        num_train_samples=num_rollouts,
        num_test_samples=num_test_rollouts,
    )
    return train_data.to(device), test_data.to(device)


def make_loss_fn(train_data: torch.Tensor, n_agents: int = 2):
    _, _, obstacle_centers, obstacle_covs, _, _ = getCarInitParams(device)
    (
        Q,
        Q_final,
        Qs,
        alpha_col,
        alpha_obst,
        alpha_u,
        position_deadzone,
        steady_state_velocity_radius,
        min_dist,
    ) = getLossParams(device)

    return BumpercarLoss(
        Q=Q,
        Q_final=Q_final,
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


def obstacle_collision_indices(
    x_log: torch.Tensor,
    obstacle_centers: list,
    radius: float = OBSTACLE_RADIUS,
    n_agents: int = 2,
):
    """
    Identifies rollouts that penetrate within the physical safe obstacle radius.
    """
    collision_indices = set()
    radius_sq = radius**2

    for agent_idx in range(n_agents):
        base = 7 * agent_idx
        positions = x_log[:, :, base : base + 2]

        for obstacle_idx, center in enumerate(obstacle_centers):
            center = center.to(
                device=positions.device, dtype=positions.dtype
            ).view(1, 1, 2)
            distance_sq = torch.sum((positions - center) ** 2, dim=-1)
            colliding = distance_sq <= radius_sq

            for rollout_idx, _ in colliding.nonzero(as_tuple=False):
                collision_indices.add(int(rollout_idx))

    return sorted(collision_indices)


def checkpoint_args(checkpoint):
    args = checkpoint.get("args", {})
    return args if isinstance(args, dict) else vars(args)


def load_controller(system: BumpercarSystem, checkpoint_path: str):
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
            key: value
            for key, value in checkpoint.items()
            if key in controller.c_ren.state_dict()
        }
        controller.c_ren.load_state_dict(ren_state, strict=False)

    controller.eval()
    controller.reset()
    return controller


def print_cbf_diagnostics(system: BumpercarSystem, title: str):
    """
    Helper function to print CBF solver telemetry and feasibility counters safely.
    """
    if hasattr(system, "cbf_filter") and system.cbf_filter is not None:
        if hasattr(system.cbf_filter, "get_diagnostics"):
            stats = system.cbf_filter.get_diagnostics()
            print(f"\n[CBF TELEMETRY - {title}] Evaluated {stats['total_steps']} steps:")
            print(
                f"  Active Interventions: {stats['active_interventions']} "
                f"({stats['active_ratio_pct']:.2f}%)"
            )
            print(
                f"  Infeasible QP Steps:   {stats['infeasible_steps']} "
                f"({stats['infeasible_ratio_pct']:.2f}%)"
            )
            print(f"  Emergency Fallbacks:   {stats['emergency_fallbacks']}")
            if hasattr(system.cbf_filter, "reset_diagnostics"):
                system.cbf_filter.reset_diagnostics()
        else:
            print(
                f"\n[CBF TELEMETRY - {title}] CBF Filter active (telemetry methods not found in cbf_qp.py)."
            )


def evaluate_controller():
    if not TRAINED_PBR_MODEL_PATH:
        raise ValueError("Set TRAINED_PBR_MODEL_PATH before evaluating.")

    _, _, obstacle_centers, _, _, _ = getCarInitParams(device)

    system = BumpercarSystem(
        params=car_params,
        x_init=None,
        u_init=None,
        dt=DT,
        r_obs_safe=OBSTACLE_RADIUS,
        d_car_safe=CAR_SAFE_DISTANCE,
        obstacle_centers=obstacle_centers,
    ).to(device)

    controller = load_controller(system, TRAINED_PBR_MODEL_PATH)
    train_data, test_data = make_eval_data(
        horizon=EVAL_HORIZON,
        num_rollouts=EVAL_NUM_ROLLOUTS,
        num_test_rollouts=EVAL_NUM_TEST_ROLLOUTS,
        random_seed=EVAL_RANDOM_SEED,
    )
    loss_fn = make_loss_fn(train_data, n_agents=system.n_agents)

    print(f"[INFO] Evaluating {TRAINED_PBR_MODEL_PATH}")
    with torch.no_grad():
        # Evaluate Train Set
        x_log, e_log, u_log = system.rollout(controller, train_data, train=False)
        train_loss = loss_fn.forward(x_log, u_log, e_log).item()
        train_collisions = loss_fn.count_collisions(x_log)
        train_obstacle_collision_indices = obstacle_collision_indices(
            x_log,
            obstacle_centers,
            radius=OBSTACLE_RADIUS,
            n_agents=system.n_agents,
        )
        print_cbf_diagnostics(system, "TRAIN")

        # Evaluate Test Set
        x_log, e_log, u_log = system.rollout(controller, test_data, train=False)
        test_loss = loss_fn.forward(x_log, u_log, e_log).item()
        test_collisions = loss_fn.count_collisions(x_log)
        test_obstacle_collision_indices = obstacle_collision_indices(
            x_log,
            obstacle_centers,
            radius=OBSTACLE_RADIUS,
            n_agents=system.n_agents,
        )
        print_cbf_diagnostics(system, "TEST")

    print("\n---------------- EVALUATION SUMMARY ----------------")
    print(
        f"Train loss: {train_loss:.4f} -- Number of collisions = {train_collisions:.0f}"
    )
    print(
        f"Test loss:  {test_loss:.4f} -- Number of collisions = {test_collisions:.0f}"
    )
    print(
        f"Train obstacle collision rollouts ({len(train_obstacle_collision_indices)}/{EVAL_NUM_ROLLOUTS}): {train_obstacle_collision_indices}"
    )
    print(
        f"Test obstacle collision rollouts  ({len(test_obstacle_collision_indices)}/{EVAL_NUM_TEST_ROLLOUTS}): {test_obstacle_collision_indices}"
    )
    print("----------------------------------------------------\n")


def simulate(
    horizon: int = 400,
    use_generated_sample: bool = SIM_USE_GENERATED_SAMPLE,
    sample_index: int = SIM_SAMPLE_INDEX,
):
    _, _, obstacle_centers, _, _, _ = getCarInitParams(device)

    system = BumpercarSystem(
        params=car_params,
        x_init=None,
        u_init=None,
        dt=DT,
        r_obs_safe=OBSTACLE_RADIUS,
        d_car_safe=CAR_SAFE_DISTANCE,
        obstacle_centers=obstacle_centers,
    ).to(device)

    if TRAINED_PBR_MODEL_PATH:
        controller = load_controller(system, TRAINED_PBR_MODEL_PATH)
        title = "Trained pRB controller with CBF filter"
    else:
        controller = ZeroController(ref_dim=2 * system.n_agents).to(device)
        title = "PID only - set TRAINED_PBR_MODEL_PATH to use pRB"

    if use_generated_sample:
        data, xbar = make_generated_sample_data(
            horizon=horizon,
            sample_index=sample_index,
            random_seed=SIM_RANDOM_SEED,
            split=SIM_DATA_SPLIT,
        )
        title = f"{title} - {SIM_DATA_SPLIT} generated sample {sample_index}"
    else:
        data, xbar = make_crossing_data(horizon=horizon, n_agents=system.n_agents)
        title = f"{title} - fixed crossing"

    with torch.no_grad():
        x_log, _, dxRef_log = system.rollout(controller, data, train=False)

    print_cbf_diagnostics(system, f"SAMPLE {sample_index}")
    return x_log[0].detach().cpu(), xbar.cpu(), dxRef_log, title


def draw_car(ax, x, y, theta, color, alpha=0.75):
    length = 0.30
    width = 0.16
    dx = torch.tensor([length / 2, length / 2, -length / 2, -length / 2])
    dy = torch.tensor([width / 2, -width / 2, -width / 2, width / 2])
    c = torch.cos(theta)
    s = torch.sin(theta)
    px = x + c * dx - s * dy
    py = y + s * dx + c * dy
    return ax.fill(
        px, py, color=color, alpha=alpha, edgecolor="black", linewidth=1.0
    )[0]


def draw_pose_arrow(ax, x, y, theta, color, length=0.45, alpha=1.0):
    dx = length * torch.cos(theta).item()
    dy = length * torch.sin(theta).item()
    arrow = FancyArrowPatch(
        (x.item(), y.item()),
        (x.item() + dx, y.item() + dy),
        arrowstyle="-|>",
        mutation_scale=12,
        color=color,
        linewidth=1.2,
        alpha=alpha,
        zorder=5,
    )
    ax.add_patch(arrow)
    return arrow


def draw_obstacles(ax, obstacle_centers, obstacle_covs):
    """
    Renders obstacles using the realistic radius of 0.80 m, showing the open corridor.
    """
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


def draw_sample_regions(ax, colors):
    x0_bumpercar, x_final, _, _, car_init_radius, _ = getCarInitParams(device)

    for i, color in enumerate(colors):
        base = 7 * i
        init_circle = Circle(
            xy=(x0_bumpercar[base].item(), x0_bumpercar[base + 1].item()),
            radius=car_init_radius,
            facecolor=color,
            edgecolor=color,
            alpha=0.10,
            linewidth=1.2,
            zorder=0,
        )
        final_circle = Circle(
            xy=(x_final[base].item(), x_final[base + 1].item()),
            radius=REPORT_FINAL_RADIUS,
            facecolor=color,
            edgecolor=color,
            alpha=0.08,
            linestyle="--",
            linewidth=1.2,
            zorder=0,
        )
        ax.add_patch(init_circle)
        ax.add_patch(final_circle)


def show_simulation():
    x_log, xbar, dx_log, title = simulate()
    _, _, obstacle_centers, obstacle_covs, _, _ = getCarInitParams(device)
    _, _, _, _, _, _, _, _, min_dist = getLossParams(device)
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
    safety_patches = []
    target_markers = []
    start_markers = []

    for i in range(n_agents):
        base = 7 * i
        color = colors[i]
        (path_line,) = ax.plot([], [], color=color, linewidth=2)
        path_lines.append(path_line)
        car_patches.append(
            draw_car(
                ax, x_log[0, base], x_log[0, base + 1], x_log[0, base + 2], color
            )
        )
        safety_circle = Circle(
            xy=(x_log[0, base].item(), x_log[0, base + 1].item()),
            radius=min_dist / 2,
            facecolor=color,
            edgecolor=color,
            alpha=0.08,
            linewidth=1.0,
        )
        ax.add_patch(safety_circle)
        safety_patches.append(safety_circle)
        target_markers.append(
            ax.plot(
                xbar[base], xbar[base + 1], marker="*", markersize=14, color=color
            )[0]
        )
        start_markers.append(
            ax.plot(
                x_log[0, base],
                x_log[0, base + 1],
                marker="o",
                markersize=7,
                color=color,
                fillstyle="none",
            )[0]
        )

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
            path_lines[i].set_data(
                x_log[: t + 1, base], x_log[: t + 1, base + 1]
            )
            safety_patches[i].center = (
                x_log[t, base].item(),
                x_log[t, base + 1].item(),
            )
            car_patches[i].remove()
            car_patches[i] = draw_car(
                ax,
                x_log[t, base],
                x_log[t, base + 1],
                x_log[t, base + 2],
                colors[i],
            )

        dist = torch.linalg.norm(x_log[t, 0:2] - x_log[t, 7:9]).item()
        is_collision = dist < min_dist
        time_text.set_color("tab:red" if is_collision else "black")
        status = "collision" if is_collision else "clear"
        time_text.set_text(
            f"step {t}   distance {dist:.2f} m   {status} < {min_dist:.2f} m"
        )
        fig.canvas.draw_idle()

    time_slider.on_changed(update)
    update(0)
    plt.show()


if __name__ == "__main__":
    if EVALUATE_MODEL and TRAINED_PBR_MODEL_PATH:
        evaluate_controller()
    elif EVALUATE_MODEL:
        print(
            "[INFO] Skipping trained-model evaluation because TRAINED_PBR_MODEL_PATH is empty."
        )
    show_simulation()