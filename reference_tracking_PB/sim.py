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


# Add your trained pRB checkpoint here. Prefer a checkpoint with MLP weights, e.g.
TRAINED_PBR_MODEL_PATH="experiments/bumpercar/trained_pRB/rPB_best.pt"

EVALUATE_MODEL = True
EVAL_HORIZON = 500
EVAL_NUM_ROLLOUTS = 100
EVAL_NUM_TEST_ROLLOUTS = 500
EVAL_RANDOM_SEED = 2

SIM_USE_GENERATED_SAMPLE = True
SIM_DATA_SPLIT = "train"
SIM_SAMPLE_INDEX = 12
SIM_RANDOM_SEED = EVAL_RANDOM_SEED
OBSTACLE_RADIUS = 1.5
REPORT_FIGURE_PATH = "experiments/bumpercar/report_trajectory.svg"
REPORT_COLLISION_FIGURE_PATH = "experiments/bumpercar/report_collision.svg"
REPORT_INIT_FIGURE_PATH = "experiments/bumpercar/report_setup.svg"
REPORT_SNAPSHOT_FIGURE_PATH = "experiments/bumpercar/report_snapshots.svg"
TRAJECTORY_GIF_PATH = "experiments/bumpercar/trajectory.gif"
ROLLOUT_METRICS_PATH = "experiments/bumpercar/rollout_metrics.csv"
REPORT_FINAL_RADIUS = 1.0

DT = 0.04



def make_generated_sample_data(horizon, sample_index=0, random_seed=11, split="train"):
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

    data = source_data[sample_index:sample_index + 1]
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
    x0_bumpercar, x_final_bumpercar, _ , _, car_init_radius, std_init_theta = getCarInitParams(device)
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
    Q, Q_final, Qs, alpha_col, alpha_obst, alpha_u, position_deadzone, steady_state_velocity_radius, min_dist = getLossParams(device)
    
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
        dt=DT,
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
        dt = DT,
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
            split=SIM_DATA_SPLIT,
        )
        title = f"{title} - {SIM_DATA_SPLIT} generated sample {sample_index}"
    else:
        data, xbar = make_crossing_data(horizon=horizon, n_agents=system.n_agents)
        title = f"{title} - fixed crossing"

    with torch.no_grad():
        x_log, _, dxRef_log = system.rollout(controller, data, train=False)

    _, _, obstacle_centers, _, _, _ = getCarInitParams(device)
    sim_obstacle_collision_indices = obstacle_collision_indices(
        x_log,
        obstacle_centers,
        n_agents=system.n_agents,
    )
    # print(f"Sim obstacle collision indices: {sim_obstacle_collision_indices}")

    return x_log[0].detach().cpu(), xbar.cpu(), dxRef_log, title


def evaluate_rollout_metrics(
    num_runs,
    horizon=400,
    split="test",
    random_seed=EVAL_RANDOM_SEED,
    save_path=ROLLOUT_METRICS_PATH,
):
    system = BumpercarSystem(
        params=car_params,
        x_init=None,
        u_init=None,
        dt=DT,
    ).to(device)

    if TRAINED_PBR_MODEL_PATH:
        controller = load_controller(system, TRAINED_PBR_MODEL_PATH)
        controller_name = "trained_pRB"
    else:
        controller = ZeroController(ref_dim=2 * system.n_agents).to(device)
        controller_name = "PID_only"

    train_data, test_data = make_eval_data(
        horizon=horizon,
        num_rollouts=num_runs,
        num_test_rollouts=num_runs,
        random_seed=random_seed,
    )
    if split == "train":
        data = train_data
    elif split == "test":
        data = test_data
    else:
        raise ValueError(f"Unknown split: {split}")

    with torch.no_grad():
        x_log, _, u_log = system.rollout(controller, data, train=False)

    nx = system.state_dim
    pos_idx = system.pos_indices()
    n_agents = system.n_agents
    _, _, obstacle_centers, _, _, _ = getCarInitParams(device)
    _, _, _, _, _, _, _, _, min_dist = getLossParams(device)

    final_pos = x_log[:, -1, pos_idx].reshape(num_runs, n_agents, 2)
    final_ref = data[:, -1, nx + torch.tensor(pos_idx, device=data.device)].reshape(num_runs, n_agents, 2)
    steady_state_error = torch.linalg.norm(final_pos - final_ref, dim=-1)

    rollout_horizon = x_log.shape[1]
    positions = x_log[:, :, pos_idx].reshape(num_runs, rollout_horizon, n_agents, 2)
    car_distance = torch.linalg.norm(positions[:, :, 0, :] - positions[:, :, 1, :], dim=-1)
    closest_car_distance = car_distance.min(dim=1).values

    centers = torch.stack([
        center.to(device=positions.device, dtype=positions.dtype).flatten()
        for center in obstacle_centers
    ])
    obstacle_distances = torch.linalg.norm(
        positions.unsqueeze(3) - centers.view(1, 1, 1, -1, 2),
        dim=-1,
    )
    closest_obstacle_distance_by_center = obstacle_distances.min(dim=1).values
    closest_obstacle_distance = closest_obstacle_distance_by_center.min(dim=-1).values
    metrics = {
        "controller": controller_name,
        "split": split,
        "horizon": horizon,
        "dt": DT,
        "x_log": x_log.detach().cpu(),
        "u_log": u_log.detach().cpu(),
        "steady_state_error": steady_state_error.detach().cpu(),
        "closest_car_distance": closest_car_distance.detach().cpu(),
        "closest_obstacle_distance": closest_obstacle_distance.detach().cpu(),
        "closest_obstacle_distance_by_center": closest_obstacle_distance_by_center.detach().cpu(),
    }

    def rmse(values, dim=0):
        return torch.sqrt(torch.mean(values.detach().cpu() ** 2, dim=dim))

    def format_list(values):
        if values.dim() == 0:
            return f"{values.item():.3f}"
        return [round(v, 3) for v in values.flatten().tolist()]

    print(f"[INFO] evaluated {num_runs} {split} rollouts with {controller_name}")
    print("[SUMMARY] Steady-state error per car [m]")
    print(f"  RMSE: {format_list(rmse(metrics['steady_state_error']))}")
    print(f"  Std:  {format_list(metrics['steady_state_error'].std(dim=0, unbiased=False))}")
    print(f"  Worst/highest: {format_list(metrics['steady_state_error'].max(dim=0).values)}")

    print("[SUMMARY] Closest car-car distance [m]")
    print(f"  RMS: {format_list(rmse(metrics['closest_car_distance']))}")
    print(f"  Std: {format_list(metrics['closest_car_distance'].std(unbiased=False))}")
    print(f"  Safety-worst/min: {metrics['closest_car_distance'].min().item():.3f}")
    print(f"  Highest/max:      {metrics['closest_car_distance'].max().item():.3f}")
    too_close_count = (metrics["closest_car_distance"] < min_dist).sum().item()
    print(f"  Runs closer than {min_dist:.2f} m: {too_close_count}/{num_runs}")

    print("[SUMMARY] Closest car-obstacle-center distance per car [m]")
    print(f"  RMS: {format_list(rmse(metrics['closest_obstacle_distance']))}")
    print(f"  Std: {format_list(metrics['closest_obstacle_distance'].std(dim=0, unbiased=False))}")
    print(f"  Safety-worst/min: {format_list(metrics['closest_obstacle_distance'].min(dim=0).values)}")
    print(f"  Highest/max:      {format_list(metrics['closest_obstacle_distance'].max(dim=0).values)}")

    print("[SUMMARY] Closest car-obstacle-center distance per car/obstacle [m]")
    print(f"  RMS: {format_list(rmse(metrics['closest_obstacle_distance_by_center']))}")
    print(f"  Std: {format_list(metrics['closest_obstacle_distance_by_center'].std(dim=0, unbiased=False))}")
    print(f"  Safety-worst/min: {format_list(metrics['closest_obstacle_distance_by_center'].min(dim=0).values)}")
    print(f"  Highest/max:      {format_list(metrics['closest_obstacle_distance_by_center'].max(dim=0).values)}")

    if save_path is not None:
        output_dir = os.path.dirname(save_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(save_path, "w", newline="") as csvfile:
            fieldnames = (
                ["run", "closest_car_distance"]
                + [f"steady_state_error_car_{i + 1}" for i in range(n_agents)]
                + [f"closest_obstacle_distance_car_{i + 1}" for i in range(n_agents)]
                + [
                    f"closest_obstacle_distance_car_{i + 1}_obstacle_{j + 1}"
                    for i in range(n_agents)
                    for j in range(len(obstacle_centers))
                ]
            )
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for run_idx in range(num_runs):
                row = {
                    "run": run_idx,
                    "closest_car_distance": closest_car_distance[run_idx].item(),
                }
                for i in range(n_agents):
                    row[f"steady_state_error_car_{i + 1}"] = steady_state_error[run_idx, i].item()
                    row[f"closest_obstacle_distance_car_{i + 1}"] = closest_obstacle_distance[run_idx, i].item()
                    for j in range(len(obstacle_centers)):
                        row[f"closest_obstacle_distance_car_{i + 1}_obstacle_{j + 1}"] = (
                            closest_obstacle_distance_by_center[run_idx, i, j].item()
                        )
                writer.writerow(row)

        print(f"[INFO] saved rollout metrics to {save_path}")

    return metrics


def draw_car(ax, x, y, theta, color, alpha=0.75):
    length = 0.30
    width = 0.16
    dx = torch.tensor([length / 2, length / 2, -length / 2, -length / 2])
    dy = torch.tensor([width / 2, -width / 2, -width / 2, width / 2])
    c = torch.cos(theta)
    s = torch.sin(theta)
    px = x + c * dx - s * dy
    py = y + s * dx + c * dy
    return ax.fill(px, py, color=color, alpha=alpha, edgecolor="black", linewidth=1.0)[0]


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


def collision_lens_points(center_a, center_b, radius, n_points=80):
    ax, ay = center_a
    bx, by = center_b
    dx = bx - ax
    dy = by - ay
    distance = (dx**2 + dy**2) ** 0.5

    if distance <= 0 or distance >= 2 * radius:
        return []

    angle = torch.atan2(torch.tensor(dy), torch.tensor(dx)).item()
    half_angle = torch.acos(torch.tensor(distance / (2 * radius))).item()

    angles_a = torch.linspace(angle - half_angle, angle + half_angle, n_points)
    angles_b = torch.linspace(angle + torch.pi - half_angle, angle + torch.pi + half_angle, n_points)

    arc_a = [(ax + radius * torch.cos(t).item(), ay + radius * torch.sin(t).item()) for t in angles_a]
    arc_b = [(bx + radius * torch.cos(t).item(), by + radius * torch.sin(t).item()) for t in reversed(angles_b)]
    return arc_a + arc_b


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


def save_report_figure(path=REPORT_FIGURE_PATH):
    x_log, xbar, dx_log, _ = simulate()
    _, _, obstacle_centers, obstacle_covs, _, _ = getCarInitParams(device)
    n_agents = 2
    colors = ["tab:blue", "tab:orange"]

    fig, ax = plt.subplots(figsize=(6.2, 7.0))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-4.0, 6.0)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, alpha=0.25)

    draw_sample_regions(ax, colors)
    draw_obstacles(ax, obstacle_centers, obstacle_covs)

    for i in range(n_agents):
        base = 7 * i
        color = colors[i]
        ax.plot(x_log[:, base], x_log[:, base + 1], color=color, linewidth=2.0, linestyle="--")
        ax.plot(x_log[0, base], x_log[0, base + 1], marker="o", markersize=6, color=color, fillstyle="none")
        ax.plot(xbar[base], xbar[base + 1], marker="*", markersize=12, color=color)
        draw_car(ax, x_log[-1, base], x_log[-1, base + 1], x_log[-1, base + 2], color)
        draw_pose_arrow(ax, x_log[-1, base], x_log[-1, base + 1], x_log[-1, base + 2], color)

    legend_handles = [
        Line2D([0], [0], color="0.25", linewidth=2.0, linestyle="--", label="Trajectory"),
        Line2D([0], [0], marker="o", color="0.25", linestyle="None", markersize=7, fillstyle="none", label="Initial position"),
        Line2D([0], [0], marker="*", color="0.25", linestyle="None", markersize=12, label="Final target"),
        Line2D([0], [0], marker="o", color="0.25", linestyle="None", markersize=13, fillstyle="none", label="Sample region"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        frameon=True,
        framealpha=0.95,
        fontsize=8,
        handlelength=1.6,
        borderpad=0.35,
        labelspacing=0.35,
        handletextpad=0.55,
        markerscale=0.75,
    )

    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] saved report figure to {path}")


def save_snapshot_report_figure(path=REPORT_SNAPSHOT_FIGURE_PATH):
    x_log, xbar, dx_log, _ = simulate()
    _, _, obstacle_centers, obstacle_covs, _, _ = getCarInitParams(device)
    n_agents = 2
    colors = ["tab:blue", "tab:orange"]
    snapshot_idxs = [0, 30, 60, 120, 399]
    snapshot_labels = ["1", "2", "3", "4", "5"]
    snapshot_alphas = [0.30, 0.50, 0.65, 0.8, 0.9]

    fig, ax = plt.subplots(figsize=(6.2, 7.0))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-4.0, 6.0)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, alpha=0.25)

    draw_sample_regions(ax, colors)
    draw_obstacles(ax, obstacle_centers, obstacle_covs)

    for i in range(n_agents):
        base = 7 * i
        color = colors[i]
        ax.plot(x_log[:, base], x_log[:, base + 1], color=color, linewidth=1.8, linestyle="--", alpha=0.45)
        ax.plot(x_log[0, base], x_log[0, base + 1], marker="o", markersize=6, color=color, fillstyle="none")
        ax.plot(xbar[base], xbar[base + 1], marker="*", markersize=12, color=color)

        for snapshot_idx, label, alpha in zip(snapshot_idxs, snapshot_labels, snapshot_alphas):
            x = x_log[snapshot_idx, base]
            y = x_log[snapshot_idx, base + 1]
            theta = x_log[snapshot_idx, base + 2]
            draw_car(ax, x, y, theta, color, alpha=alpha)
            draw_pose_arrow(ax, x, y, theta, color, length=0.34, alpha=alpha)
            label_offset = 0.26 if i == 0 else -0.26
            ax.text(
                x.item() + label_offset * torch.cos(theta + torch.pi / 2).item(),
                y.item() + label_offset * torch.sin(theta + torch.pi / 2).item(),
                label,
                color=color,
                fontsize=8,
                fontweight="bold",
                ha="center",
                va="center",
                bbox={
                    "boxstyle": "circle,pad=0.18",
                    "facecolor": "white",
                    "edgecolor": color,
                    "linewidth": 0.9,
                    "alpha": min(alpha + 0.10, 1.0),
                },
            )

    legend_handles = [
        Line2D([0], [0], color="0.25", linewidth=1.8, linestyle="--", label="Trajectory"),
        Line2D([0], [0], marker="o", color="0.25", linestyle="None", markersize=7, fillstyle="none", label="Initial position"),
        Line2D([0], [0], marker="*", color="0.25", linestyle="None", markersize=12, label="Final target"),
        Line2D([0], [0], marker="s", color="0.25", linestyle="None", markersize=8, alpha=0.70, label="Snapshots 1-5"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        frameon=True,
        framealpha=0.95,
        fontsize=8,
        handlelength=1.6,
        borderpad=0.35,
        labelspacing=0.35,
        handletextpad=0.55,
        markerscale=0.75,
    )

    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] saved snapshot report figure to {path}")


def save_collision_report_figure(path=REPORT_COLLISION_FIGURE_PATH):
    _, _, obstacle_centers, obstacle_covs, _, _ = getCarInitParams(device)
    _, _, _, _, _, _, _, _, min_dist = getLossParams(device)
    x0_bumpercar, x_final, _, _, _, _ = getCarInitParams(device)
    n_agents = 2
    colors = ["tab:blue", "tab:orange"]
    safety_radius = min_dist / 2

    collision_fractions = torch.tensor([0.425, 0.425])

    fig, ax = plt.subplots(figsize=(6.2, 7.0))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-4.0, 6.0)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, alpha=0.25)

    draw_sample_regions(ax, colors)
    draw_obstacles(ax, obstacle_centers, obstacle_covs)

    centers = []
    for i in range(n_agents):
        base = 7 * i
        color = colors[i]
        start = x0_bumpercar[base:base + 2]
        final = x_final[base:base + 2]
        direction = final - start
        collision = start + collision_fractions[i] * direction
        theta = torch.atan2(direction[1], direction[0])
        centers.append((collision[0].item(), collision[1].item()))

        ax.plot(
            [start[0].item(), final[0].item()],
            [start[1].item(), final[1].item()],
            color=color,
            linewidth=2.0,
            linestyle="--",
            alpha=0.85,
        )
        ax.plot(start[0], start[1], marker="o", markersize=6, color=color, fillstyle="none")
        ax.plot(final[0], final[1], marker="*", markersize=12, color=color)
        safety_circle = Circle(
            xy=(collision[0].item(), collision[1].item()),
            radius=safety_radius,
            facecolor="tab:red",
            edgecolor="tab:red",
            alpha=0.14,
            linewidth=1.4,
            zorder=2,
        )
        ax.add_patch(safety_circle)
        draw_car(ax, collision[0], collision[1], theta, color)
        draw_pose_arrow(ax, collision[0], collision[1], theta, color)

    overlap = collision_lens_points(centers[0], centers[1], safety_radius)
    if overlap:
        ax.add_patch(
            Polygon(
                overlap,
                closed=True,
                facecolor="tab:red",
                edgecolor="tab:red",
                alpha=0.45,
                linewidth=1.2,
                zorder=3,
            )
        )

    legend_handles = [
        Line2D([0], [0], color="0.25", linewidth=2.0, linestyle="--", label="Trajectory"),
        Line2D([0], [0], marker="o", color="0.25", linestyle="None", markersize=7, fillstyle="none", label="Initial position"),
        Line2D([0], [0], marker="*", color="0.25", linestyle="None", markersize=12, label="Final target"),
        Line2D([0], [0], marker="o", color="tab:red", linestyle="None", markersize=13, fillstyle="none", label="1 m safety radius"),
        Line2D([0], [0], color="tab:red", linewidth=6, alpha=0.45, label="Collision overlap"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=True, framealpha=0.95)

    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] saved collision report figure to {path}")


def save_init_report_figure(path=REPORT_INIT_FIGURE_PATH):
    x0_bumpercar, x_final, obstacle_centers, obstacle_covs, _, _ = getCarInitParams(device)
    n_agents = 2
    colors = ["tab:blue", "tab:orange"]

    fig, ax = plt.subplots(figsize=(6.2, 7.0))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-4.0, 6.0)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, alpha=0.25)

    draw_sample_regions(ax, colors)
    draw_obstacles(ax, obstacle_centers, obstacle_covs)

    for i in range(n_agents):
        base = 7 * i
        color = colors[i]
        ax.plot(
            x0_bumpercar[base],
            x0_bumpercar[base + 1],
            marker="o",
            markersize=7,
            color=color,
            fillstyle="none",
        )
        ax.plot(
            x_final[base],
            x_final[base + 1],
            marker="*",
            markersize=13,
            color=color,
        )
        draw_car(
            ax,
            x0_bumpercar[base],
            x0_bumpercar[base + 1],
            x0_bumpercar[base + 2],
            color,
        )
        draw_pose_arrow(
            ax,
            x0_bumpercar[base],
            x0_bumpercar[base + 1],
            x0_bumpercar[base + 2],
            color,
        )

    legend_handles = [
        Line2D([0], [0], marker="o", color="0.25", linestyle="None", markersize=7, fillstyle="none", label="Initial position"),
        Line2D([0], [0], marker="*", color="0.25", linestyle="None", markersize=12, label="Final target"),
        Line2D([0], [0], marker="o", color="0.25", linestyle="None", markersize=13, fillstyle="none", label="Sample region"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=True, framealpha=0.95)

    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] saved init report figure to {path}")


def save_trajectory_gif(path=TRAJECTORY_GIF_PATH, frame_stride=4, fps=20):
    x_log, xbar, dx_log, title = simulate()
    _, _, obstacle_centers, obstacle_covs, _, _ = getCarInitParams(device)
    _, _, _, _, _, _, _, _, min_dist = getLossParams(device)
    n_agents = 2
    colors = ["tab:blue", "tab:orange"]
    frames = list(range(0, x_log.shape[0], frame_stride))
    if frames[-1] != x_log.shape[0] - 1:
        frames.append(x_log.shape[0] - 1)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-4.0, 6.0)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, alpha=0.3)
    draw_sample_regions(ax, colors)
    draw_obstacles(ax, obstacle_centers, obstacle_covs)

    path_lines = []
    car_patches = []
    safety_patches = []
    time_text = ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top")

    for i in range(n_agents):
        base = 7 * i
        color = colors[i]
        (path_line,) = ax.plot([], [], color=color, linewidth=2)
        path_lines.append(path_line)
        car_patches.append(draw_car(ax, x_log[0, base], x_log[0, base + 1], x_log[0, base + 2], color))
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
        ax.plot(xbar[base], xbar[base + 1], marker="*", markersize=14, color=color)
        ax.plot(x_log[0, base], x_log[0, base + 1], marker="o", markersize=7, color=color, fillstyle="none")

    def update(frame):
        artists = []
        for i in range(n_agents):
            base = 7 * i
            path_lines[i].set_data(x_log[:frame + 1, base], x_log[:frame + 1, base + 1])
            safety_patches[i].center = (x_log[frame, base].item(), x_log[frame, base + 1].item())
            car_patches[i].remove()
            car_patches[i] = draw_car(
                ax,
                x_log[frame, base],
                x_log[frame, base + 1],
                x_log[frame, base + 2],
                colors[i],
            )
            artists.extend([path_lines[i], safety_patches[i], car_patches[i]])

        dist = torch.linalg.norm(x_log[frame, 0:2] - x_log[frame, 7:9]).item()
        is_collision = dist < min_dist
        time_text.set_color("tab:red" if is_collision else "black")
        time_text.set_text(f"step {frame}   distance {dist:.2f} m")
        artists.append(time_text)
        return artists

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=frames,
        interval=1000 / fps,
        blit=False,
        repeat=True,
    )

    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    anim.save(path, writer="pillow", fps=fps)
    plt.close(fig)
    print(f"[INFO] saved trajectory GIF to {path}")


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
        car_patches.append(draw_car(ax, x_log[0, base], x_log[0, base + 1], x_log[0, base + 2], color))
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
            safety_patches[i].center = (x_log[t, base].item(), x_log[t, base + 1].item())
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
        time_text.set_text(f"step {t}   distance {dist:.2f} m   {status} < {min_dist:.2f} m")
        fig.canvas.draw_idle()

    time_slider.on_changed(update)
    update(0)
    plt.show()


if __name__ == "__main__":
    # evaluate_rollout_metrics(num_runs=500, horizon=400, save_path="experiments/bumpercar/distances.csv")
    if EVALUATE_MODEL and TRAINED_PBR_MODEL_PATH:
        evaluate_controller()
    elif EVALUATE_MODEL:
        print("[INFO] skipping trained-model evaluation because TRAINED_PBR_MODEL_PATH is empty.")
    show_simulation()

