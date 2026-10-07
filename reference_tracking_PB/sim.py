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

import experiment_params
from experiment_params import getCarFinalParams, getCarInitParams, getLossParams, getExtraLossParams, getGoalSequence, getMissions, n_agents as N_AGENTS, args as ARGS
from config import device
from controllers import PerfBoostController
from controllers.MLP import ZeroController
from loss_functions import BumpercarLoss
from plants import BumpercarSystem, car_params
from plants.bumpercar.bumpercar_dataset import BumpercarDataset
from utils.plot_functions import agent_color, route_points


# Checkpoint to evaluate: --checkpoint <path>, default the provided one. Use a full checkpoint (with MLP weights).
TRAINED_PBR_MODEL_PATH = ARGS.checkpoint or os.path.join(BASE_DIR, "experiments/bumpercar/trained_pRB/rPB_best.pt")

# Training settings of the evaluated run override sim.py's own flags: same car count, start/goal spreads,
# loss weights and scenario as in training. Evaluation seed, horizon and violation distances stay sim.py's own.
TRAINING_SETTINGS = ["n_agents", "start_radius", "start_heading_range", "goal_heading_range",
                     "alpha_u", "q_final_factor", "alpha_col", "alpha_obst", "min_dist", "obst_safe_dist",
                     "alpha_heading", "alpha_heading_final", "heading_radius", "goal_window_steps", "alpha_offset_final",
                     "alpha_bounds", "alpha_smooth", "turn_speed"]
checkpoint = torch.load(TRAINED_PBR_MODEL_PATH, map_location="cpu", weights_only=False)
stored_args = checkpoint.get("args", {})
stored_args = stored_args if isinstance(stored_args, dict) else vars(stored_args)
for name in TRAINING_SETTINGS:
    if name in stored_args:
        setattr(ARGS, name, stored_args[name])
missing = [name for name in TRAINING_SETTINGS if name not in stored_args]
if missing:
    print(f"[WARN] checkpoint does not store {missing}; using sim.py's flags for these")
experiment_params.n_agents = N_AGENTS = ARGS.n_agents
if ARGS.current_scenario:
    print("[INFO] --current-scenario: evaluating on the current SCENARIOS and OBSTACLES, not the checkpoint's")
elif "scenario" in checkpoint:
    experiment_params.SCENARIOS[N_AGENTS] = {key: value for key, value in checkpoint["scenario"].items() if key != "obstacles"}   # start, goal(s)
    experiment_params.OBSTACLES = checkpoint["scenario"]["obstacles"]
else:
    print("[WARN] checkpoint does not store its scenario; using the current SCENARIOS and OBSTACLES")
# horizon of the training run: metrics and figures cover it (the "final" values are at its end); the loss evaluation
# runs 100 steps longer. Goal sequence: switch between goals after the same number of steps as in training (the last
# goal keeps the rest of a longer evaluation horizon)
TRAIN_HORIZON = stored_args.get("horizon", ARGS.horizon)
TRAIN_LEG_STEPS = TRAIN_HORIZON // len(getGoalSequence())

EVALUATE_MODEL = True
EVAL_HORIZON = TRAIN_HORIZON + 100
EVAL_NUM_ROLLOUTS = 100
EVAL_NUM_TEST_ROLLOUTS = 500
EVAL_RANDOM_SEED = 2

SIM_USE_GENERATED_SAMPLE = True
SIM_DATA_SPLIT = "train"
SIM_SAMPLE_INDEX = 12
SIM_RANDOM_SEED = EVAL_RANDOM_SEED
OBSTACLE_RADIUS = 1.5
EVAL_MIN_DIST = 2.0   # car-car safety distance for metrics and figures (independent of the training loss)
ARENA_BOUNDS = ((-3.5, 3.5), (-4.0, 6.0))   # arena edges (x, y) [m] for the "left the arena" metric (as in BumpercarLoss)
REPORT_FINAL_RADIUS = 1.0

DT = 0.04


def get_output_dir(checkpoint_path=None):
    """
    Folder for the figures, GIF and CSV of one checkpoint:
    - checkpoint of a training run (.../<run>/checkpoints/<name>.pt) -> <run>/sim/<name>/
    - any other checkpoint (e.g. the provided rPB_best.pt)             -> saved_results/sim_<name>/
    """
    checkpoint_path = os.path.abspath(checkpoint_path or TRAINED_PBR_MODEL_PATH)
    checkpoint_dir = os.path.dirname(checkpoint_path)
    name = os.path.splitext(os.path.basename(checkpoint_path))[0]
    if os.path.basename(checkpoint_dir) == "checkpoints":
        run_dir = os.path.dirname(checkpoint_dir)
        return os.path.join(run_dir, "sim", name)
    return os.path.join(BASE_DIR, "experiments", "bumpercar", "saved_results", f"sim_{name}")


def output_file(filename):
    return os.path.join(get_output_dir(), filename)



def closest_pair_distance(positions):
    """positions [..., n_agents, 2] -> distance of the closest pair of cars [...]."""
    n = positions.shape[-2]
    distances = torch.cdist(positions, positions)                                  # [..., n, n]
    off_diagonal = ~torch.eye(n, dtype=torch.bool, device=positions.device)
    return distances[..., off_diagonal].min(dim=-1).values


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
    ref = data[0, :, nx:]                       # reference over time: the goals in visiting order

    return data.to(device), ref.to(device)


def make_crossing_data(horizon, n_agents=N_AGENTS):
    nx = 7 * n_agents
    data = torch.zeros(1, horizon, 2 * nx)

    x0_bumpercar, _, _, _, _, _ = getCarInitParams(device)

    data[:, 0, :nx] = x0_bumpercar
    # reference: the nominal goals in visiting order, each during its leg (same schedule as in training)
    goals = getGoalSequence()
    leg_starts, _, _ = BumpercarDataset.goal_schedule(horizon, len(goals), TRAIN_LEG_STEPS)
    leg_ends = leg_starts[1:] + [horizon]
    for goal, start, end in zip(goals, leg_starts, leg_ends):
        data[:, start:end, nx:] = goal.view(1, 1, -1)

    return data.to(device), data[0, :, nx:].to(device)


def make_eval_data(horizon, num_rollouts, num_test_rollouts, random_seed):
    x0_bumpercar, x_final_bumpercar, _ , _, car_init_radius, start_heading_range = getCarInitParams(device)
    x_final_limit, y_final_limit, final_car_min_dist = getCarFinalParams()
    poses, missions = getMissions()

    dataset = BumpercarDataset(
        random_seed=random_seed,
        horizon=horizon,
        x0=x0_bumpercar,
        x_final=x_final_bumpercar,
        car_init_radius=car_init_radius,
        x_final_limit=x_final_limit,
        y_final_limit=y_final_limit,
        final_car_min_dist=final_car_min_dist,
        start_heading_range=start_heading_range,
        n_agents=N_AGENTS,
        goal_heading_range=ARGS.goal_heading_range * torch.pi / 180,   # pass the training run's --goal-heading-range
        poses=poses,
        missions=missions,                      # from the checkpoint's scenario: one mission drawn per sample
        leg_steps=TRAIN_LEG_STEPS,
    )
    train_data, test_data = dataset.get_data(
        num_train_samples=num_rollouts,
        num_test_samples=num_test_rollouts,
    )
    return train_data.to(device), test_data.to(device)


def mission_labels(data, n_agents=N_AGENTS):
    """Mission of each sample, recovered from its data: per car the nearest named pose (by position) of its start and
    of its goal in every leg, e.g. "BL-TR-BL | BR-TL-BR" (car 1 | car 2)."""
    poses, _ = getMissions()
    names = list(poses)
    centers = torch.tensor([poses[name][0:2] for name in names], dtype=data.dtype, device=data.device)
    nx = 7 * n_agents
    leg_starts, _, _ = BumpercarDataset.goal_schedule(data.shape[1], len(getGoalSequence()), TRAIN_LEG_STEPS)
    points = [data[:, 0, :nx]] + [data[:, start, nx:] for start in leg_starts]    # start, then the goal of every leg
    labels = []
    for s in range(data.shape[0]):
        cars = []
        for i in range(n_agents):
            sequence = [names[torch.linalg.norm(centers - point[s, 7 * i:7 * i + 2], dim=-1).argmin()] for point in points]
            cars.append("-".join(sequence))
        labels.append(" | ".join(cars))
    return labels


def make_loss_fn(train_data, n_agents=N_AGENTS):
    _, _, obstacle_centers, obstacle_covs, _, _ = getCarInitParams(device)    
    _, goal_leg_ends, final_leg_start = BumpercarDataset.goal_schedule(train_data.shape[1], len(getGoalSequence()), TRAIN_LEG_STEPS)
    Q, Q_final, Qs, alpha_col, alpha_obst, alpha_u, position_deadzone, steady_state_velocity_radius, min_dist = getLossParams(device)
    
    return BumpercarLoss(
        Q=Q,
        Q_final=Q_final,
        Qs=Qs,
        alpha_u=alpha_u,
        xbar=train_data[0, :, 7 * n_agents:],
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
        **getExtraLossParams(),
        goal_leg_ends=goal_leg_ends,
        final_leg_start=final_leg_start,
    )


def obstacle_collision_indices(x_log, obstacle_centers, radius=OBSTACLE_RADIUS, n_agents=N_AGENTS):
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
        ren_internal_state_init=None,
        dt=system.dt,
        ref_trace_tau=args["ren_ref_trace_tau"],
        m2_type=args["m2"],
        m2_gain_bound=args["m2_gain_bound"],
    ).to(device)
    # all tensors must match: a checkpoint of another controller structure fails here instead of loading partly
    controller.load_state_dict(checkpoint["controller_state_dict"])

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
        n_agents=N_AGENTS,
        dt=DT,
        turn_speed=ARGS.turn_speed,
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
        train_collisions = loss_fn.count_collisions(x_log, min_dist=EVAL_MIN_DIST)
        train_obstacle_collision_indices = obstacle_collision_indices(
            x_log,
            obstacle_centers,
            n_agents=system.n_agents,
        )

        x_log, e_log, u_log = system.rollout(controller, test_data, train=False)
        test_loss = loss_fn.forward(x_log, u_log, e_log).item()
        test_collisions = loss_fn.count_collisions(x_log, min_dist=EVAL_MIN_DIST)
        test_obstacle_collision_indices = obstacle_collision_indices(
            x_log,
            obstacle_centers,
            n_agents=system.n_agents,
        )

    print(f"Train loss: {train_loss:.4f} -- Number of collisions = {train_collisions:.0f}")
    print(f"Test loss: {test_loss:.4f} -- Number of collisions = {test_collisions:.0f}")
    print(f"Train obstacle collision indices: {train_obstacle_collision_indices}")
    print(f"Test obstacle collision indices: {test_obstacle_collision_indices}")




def simulate(horizon=TRAIN_HORIZON, use_generated_sample=SIM_USE_GENERATED_SAMPLE, sample_index=SIM_SAMPLE_INDEX):
    system = BumpercarSystem(
        params=car_params,
        x_init=None,
        u_init=None,
        n_agents=N_AGENTS,
        dt = DT,
        turn_speed=ARGS.turn_speed,
    ).to(device)

    if TRAINED_PBR_MODEL_PATH:
        controller = load_controller(system, TRAINED_PBR_MODEL_PATH)
        title = "Trained pRB controller"
    else:
        controller = ZeroController(ref_dim=2 * system.n_agents).to(device)
        title = "PID only - set TRAINED_PBR_MODEL_PATH to use pRB"

    if use_generated_sample:
        data, ref = make_generated_sample_data(
            horizon=horizon,
            sample_index=sample_index,
            random_seed=SIM_RANDOM_SEED,
            split=SIM_DATA_SPLIT,
        )
        title = f"{title} - {SIM_DATA_SPLIT} generated sample {sample_index}"
    else:
        data, ref = make_crossing_data(horizon=horizon, n_agents=system.n_agents)
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

    return x_log[0].detach().cpu(), ref.cpu(), dxRef_log, title


def evaluate_rollout_metrics(
    num_runs,
    horizon=TRAIN_HORIZON,
    split="test",
    random_seed=EVAL_RANDOM_SEED,
    save_path="auto",   # "auto": <output dir>/rollout_metrics.csv, None: don't save
):
    if save_path == "auto":
        save_path = output_file("rollout_metrics.csv")
    system = BumpercarSystem(
        params=car_params,
        x_init=None,
        u_init=None,
        n_agents=N_AGENTS,
        dt=DT,
        turn_speed=ARGS.turn_speed,
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
    min_dist = EVAL_MIN_DIST

    final_pos = x_log[:, -1, pos_idx].reshape(num_runs, n_agents, 2)
    final_ref = data[:, -1, nx + torch.tensor(pos_idx, device=data.device)].reshape(num_runs, n_agents, 2)
    steady_state_error = torch.linalg.norm(final_pos - final_ref, dim=-1)

    # final heading error per car [deg], wrapped to [0, 180]
    final_theta = x_log[:, -1, 2::7]                                                 # [runs, n_agents]
    target_theta = data[:, -1, nx + 2::7]                                            # [runs, n_agents] theta*
    heading_difference = target_theta - final_theta
    final_heading_error = torch.rad2deg(torch.atan2(torch.sin(heading_difference), torch.cos(heading_difference))).abs()

    # goal sequence: position and heading error at the end of every earlier goal's leg (the last goal: above)
    _, goal_leg_ends, _ = BumpercarDataset.goal_schedule(data.shape[1], len(getGoalSequence()), TRAIN_LEG_STEPS)
    earlier_goal_position_errors, earlier_goal_heading_errors = [], []
    for end in goal_leg_ends[:-1]:
        position = x_log[:, end, pos_idx].reshape(num_runs, n_agents, 2)
        goal_position = data[:, end, nx + torch.tensor(pos_idx, device=data.device)].reshape(num_runs, n_agents, 2)
        difference = data[:, end, nx + 2::7] - x_log[:, end, 2::7]
        earlier_goal_position_errors.append(torch.linalg.norm(position - goal_position, dim=-1))      # [runs, n_agents]
        earlier_goal_heading_errors.append(torch.rad2deg(torch.atan2(torch.sin(difference), torch.cos(difference))).abs())

    rollout_horizon = x_log.shape[1]
    positions = x_log[:, :, pos_idx].reshape(num_runs, rollout_horizon, n_agents, 2)
    car_distance = closest_pair_distance(positions)                                  # [runs, T]
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

    # arena: how far each car's center got past the arena edge (0 = stayed inside), per run and car
    (x_min, x_max), (y_min, y_max) = ARENA_BOUNDS
    past_edge = torch.stack([
        x_min - positions[..., 0], positions[..., 0] - x_max,
        y_min - positions[..., 1], positions[..., 1] - y_max,
    ]).amax(dim=0).clamp(min=0)                                                     # [runs, T, n_agents]
    arena_excursion = past_edge.amax(dim=1)                                         # [runs, n_agents]

    # parked cars: while a car's goal is exactly its start (parked), how far it got from its start [m] (0 = never parked)
    start_position = data[:, 0, pos_idx].reshape(num_runs, 1, n_agents, 2)
    goal_position = data[:, :, nx + torch.tensor(pos_idx, device=data.device)].reshape(num_runs, rollout_horizon, n_agents, 2)
    parked = (goal_position == start_position).all(dim=-1)                         # [runs, T, n_agents]
    parked_drift = (torch.linalg.norm(positions - start_position, dim=-1) * parked).amax(dim=1)   # [runs, n_agents]
    is_parked = parked.any(dim=1)                                                   # [runs, n_agents]

    # mission of each run (from its start and goals)
    missions = mission_labels(data, n_agents)
    metrics = {
        "controller": controller_name,
        "split": split,
        "horizon": horizon,
        "dt": DT,
        "x_log": x_log.detach().cpu(),
        "u_log": u_log.detach().cpu(),
        "steady_state_error": steady_state_error.detach().cpu(),
        "final_heading_error_deg": final_heading_error.detach().cpu(),
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

    print("[SUMMARY] Final heading error per car [deg]")
    heading_errors = metrics["final_heading_error_deg"]
    print(f"  RMS:  {format_list(rmse(heading_errors))}")
    print(f"  Std:  {format_list(heading_errors.std(dim=0, unbiased=False))}")
    print(f"  Worst/highest: {format_list(heading_errors.max(dim=0).values)}")
    print(f"  Runs below 10 deg [%]: {format_list((heading_errors < 10).float().mean(dim=0) * 100)}")

    for k, (position_error, heading_error) in enumerate(zip(earlier_goal_position_errors, earlier_goal_heading_errors)):
        print(f"[SUMMARY] Goal {k + 1} of {len(goal_leg_ends)} (end of its leg, step {goal_leg_ends[k]}) per car")
        print(f"  Position error RMSE [m]: {format_list(rmse(position_error))}, worst {format_list(position_error.max(dim=0).values)}")
        print(f"  Heading error RMS [deg]: {format_list(rmse(heading_error))}, worst {format_list(heading_error.max(dim=0).values)}")
        print(f"  Runs below 10 deg [%]:   {format_list((heading_error.cpu() < 10).float().mean(dim=0) * 100)}")

    print("[SUMMARY] Arena (x in [%g, %g], y in [%g, %g]) per car" % (x_min, x_max, y_min, y_max))
    print(f"  Runs leaving the arena: {format_list((arena_excursion.cpu() > 0).sum(dim=0).float())} of {num_runs}")
    print(f"  Farthest past the edge [m]: {format_list(arena_excursion.max(dim=0).values)}")

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

    # per mission: final position RMSE and heading RMS per car, largest drift of a parked car, closest car-car and
    # car-obstacle distance (worst run); car 1 / car 2 / ...
    print("[SUMMARY] Per mission (car 1 / car 2): final position RMSE [m] | final heading RMS [deg] | parked drift max [m] "
          f"| closest car-car min [m] (runs < {min_dist:g} m) | closest obstacle min [m]")
    for mission in sorted(set(missions)):
        runs = torch.tensor([label == mission for label in missions])
        drift = ["-" if not is_parked[runs, i].any() else f"{parked_drift[runs, i].max().item():.2f}" for i in range(n_agents)]
        print(f"  {mission}  (n={runs.sum().item():3d}): "
              f"pos {'/'.join(f'{v:.2f}' for v in rmse(steady_state_error[runs]).tolist())} | "
              f"heading {'/'.join(f'{v:.0f}' for v in rmse(final_heading_error[runs]).tolist())} | "
              f"parked {'/'.join(drift)} | "
              f"car-car {closest_car_distance[runs].min().item():.2f} ({(closest_car_distance[runs] < min_dist).sum().item()}) | "
              f"obstacle {'/'.join(f'{v:.2f}' for v in closest_obstacle_distance[runs].min(dim=0).values.tolist())}")

    if save_path is not None:
        output_dir = os.path.dirname(save_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(save_path, "w", newline="") as csvfile:
            fieldnames = (
                ["run", "mission", "closest_car_distance"]
                + [f"parked_drift_car_{i + 1}" for i in range(n_agents)]
                + [f"steady_state_error_car_{i + 1}" for i in range(n_agents)]
                + [f"final_heading_error_deg_car_{i + 1}" for i in range(n_agents)]
                + [f"goal_{k + 1}_{name}_car_{i + 1}" for k in range(len(earlier_goal_position_errors))
                   for name in ("position_error", "heading_error_deg") for i in range(n_agents)]
                + [f"arena_excursion_car_{i + 1}" for i in range(n_agents)]
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
                    "mission": missions[run_idx],
                    "closest_car_distance": closest_car_distance[run_idx].item(),
                }
                for i in range(n_agents):
                    row[f"parked_drift_car_{i + 1}"] = parked_drift[run_idx, i].item() if is_parked[run_idx, i] else ""
                    row[f"steady_state_error_car_{i + 1}"] = steady_state_error[run_idx, i].item()
                    row[f"final_heading_error_deg_car_{i + 1}"] = final_heading_error[run_idx, i].item()
                    row[f"arena_excursion_car_{i + 1}"] = arena_excursion[run_idx, i].item()
                    row[f"closest_obstacle_distance_car_{i + 1}"] = closest_obstacle_distance[run_idx, i].item()
                    for k in range(len(earlier_goal_position_errors)):
                        row[f"goal_{k + 1}_position_error_car_{i + 1}"] = earlier_goal_position_errors[k][run_idx, i].item()
                        row[f"goal_{k + 1}_heading_error_deg_car_{i + 1}"] = earlier_goal_heading_errors[k][run_idx, i].item()
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


def draw_goals(ax, goals, car, color, star_size=12):
    """Goals of one car (flat states, in visiting order): a star with its target-heading arrow each,
    numbered 1, 2, ... when there are several. Returns the marker of the last goal."""
    base = 7 * car
    for k, goal in enumerate(goals):
        draw_pose_arrow(ax, goal[base], goal[base + 1], goal[base + 2], color, length=0.4)
        marker = ax.plot(goal[base], goal[base + 1], marker="*", markersize=star_size, color=color)[0]
        if len(goals) > 1:
            ax.annotate(str(k + 1), (float(goal[base]), float(goal[base + 1])), textcoords="offset points",
                        xytext=(7, -10), color=color, fontsize=8, fontweight="bold")
    return marker


def draw_route(ax, ref, car, color, star_size=12, start=None):
    """Route of one car in this sample (reference over time ref): every goal as a star with its target-heading arrow;
    with start = this sample's start state, also its start-heading arrow. Returns the marker of the last goal."""
    earlier_goals, goal = route_points(ref)
    base = 7 * car
    if start is not None:
        draw_pose_arrow(ax, start[base], start[base + 1], start[base + 2], color, length=0.4)
    return draw_goals(ax, earlier_goals + [goal], car, color, star_size)


def route_legend():
    """Legend note for a goal sequence: the goals are numbered in visiting order."""
    if len(getGoalSequence()) < 2:
        return []
    return [Line2D([0], [0], linestyle="None", label="Goals numbered in visiting order")]


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
        for earlier_goal in getGoalSequence()[:-1]:   # earlier goals of a sequence, same spread as the last goal
            ax.add_patch(Circle(
                xy=(earlier_goal[base].item(), earlier_goal[base + 1].item()),
                radius=REPORT_FINAL_RADIUS,
                facecolor=color,
                edgecolor=color,
                alpha=0.05,
                linestyle=":",
                linewidth=1.0,
                zorder=0,
            ))


def save_report_figure(path=None):
    path = path or output_file("report_trajectory.svg")
    x_log, ref, dx_log, _ = simulate()
    _, _, obstacle_centers, obstacle_covs, _, _ = getCarInitParams(device)
    n_agents = N_AGENTS
    colors = [agent_color(i) for i in range(n_agents)]

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
        draw_route(ax, ref, i, color, star_size=12, start=x_log[0])
        draw_car(ax, x_log[-1, base], x_log[-1, base + 1], x_log[-1, base + 2], color)
        draw_pose_arrow(ax, x_log[-1, base], x_log[-1, base + 1], x_log[-1, base + 2], color)

    legend_handles = [
        Line2D([0], [0], color="0.25", linewidth=2.0, linestyle="--", label="Trajectory"),
        Line2D([0], [0], marker="o", color="0.25", linestyle="None", markersize=7, fillstyle="none", label="Start (this sample)"),
        Line2D([0], [0], marker=r"$\rightarrow$", color="0.25", linestyle="None", markersize=12, label="Heading (start / target)"),
        Line2D([0], [0], marker="*", color="0.25", linestyle="None", markersize=12, label="Goal (this sample)"),
        *route_legend(),
        Line2D([0], [0], marker="o", color="0.25", linestyle="None", markersize=13, fillstyle="none", label="Sampling region (nominal ± spread)"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper left", bbox_to_anchor=(1.02, 1.0),   # outside the axes: never covers goals or trajectories
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


def save_snapshot_report_figure(path=None):
    path = path or output_file("report_snapshots.svg")
    x_log, ref, dx_log, _ = simulate()
    _, _, obstacle_centers, obstacle_covs, _, _ = getCarInitParams(device)
    n_agents = N_AGENTS
    colors = [agent_color(i) for i in range(n_agents)]
    T = x_log.shape[0]
    snapshot_idxs = [0, int(0.075 * T), int(0.15 * T), int(0.3 * T), T - 1]   # 0, 30, 60, 120, 399 for T = 400
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
        draw_route(ax, ref, i, color, star_size=12, start=x_log[0])

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
        Line2D([0], [0], marker="o", color="0.25", linestyle="None", markersize=7, fillstyle="none", label="Start (this sample)"),
        Line2D([0], [0], marker=r"$\rightarrow$", color="0.25", linestyle="None", markersize=12, label="Heading (start / target)"),
        Line2D([0], [0], marker="*", color="0.25", linestyle="None", markersize=12, label="Goal (this sample)"),
        *route_legend(),
        Line2D([0], [0], marker="s", color="0.25", linestyle="None", markersize=8, alpha=0.70, label="Snapshots 1-5"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper left", bbox_to_anchor=(1.02, 1.0),   # outside the axes: never covers goals or trajectories
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


def save_collision_report_figure(path=None):
    path = path or output_file("report_collision.svg")
    _, _, obstacle_centers, obstacle_covs, _, _ = getCarInitParams(device)
    min_dist = EVAL_MIN_DIST
    x0_bumpercar, x_final, _, _, _, _ = getCarInitParams(device)
    n_agents = 2   # illustration of the two-car crossing (first two cars only)
    colors = [agent_color(i) for i in range(n_agents)]
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
        Line2D([0], [0], color="0.25", linewidth=2.0, linestyle="--", label="Straight path (illustration)"),
        Line2D([0], [0], marker="o", color="0.25", linestyle="None", markersize=7, fillstyle="none", label="Start (nominal)"),
        Line2D([0], [0], marker="*", color="0.25", linestyle="None", markersize=12, label="Goal (nominal)"),
        Line2D([0], [0], marker="o", color="tab:red", linestyle="None", markersize=13, fillstyle="none", label="1 m safety radius"),
        Line2D([0], [0], color="tab:red", linewidth=6, alpha=0.45, label="Collision overlap"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True, framealpha=0.95)   # outside the axes

    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] saved collision report figure to {path}")


def save_mission_nominal_figure(path=None, horizon=TRAIN_HORIZON):
    """One rollout per mission without random offsets (nominal poses): trajectories in a grid, with start (square),
    goals (stars with target heading) and final heading (hollow arrow); final heading error per car in the title."""
    path = path or output_file("missions_nominal.svg")
    poses, missions = getMissions()
    _, _, obstacle_centers, obstacle_covs, _, _ = getCarInitParams(device)
    n_agents, nx = N_AGENTS, 7 * N_AGENTS
    system = BumpercarSystem(params=car_params, x_init=None, u_init=None, n_agents=n_agents, dt=DT,
                             turn_speed=ARGS.turn_speed).to(device)
    controller = load_controller(system, TRAINED_PBR_MODEL_PATH)

    # nominal data: start pose, then the goal of every leg (legs as in training)
    def state(names):
        return torch.tensor([v for name in names for v in (*poses[name], 0.0, 0.0, 0.0, 0.0)])
    leg_starts, _, _ = BumpercarDataset.goal_schedule(horizon, len(missions[0][0]) - 1, TRAIN_LEG_STEPS)
    leg_ends = leg_starts[1:] + [horizon]
    data = torch.zeros(len(missions), horizon, 2 * nx)
    for m, mission in enumerate(missions):
        data[m, 0, :nx] = state([car[0] for car in mission])
        for leg, (start, end) in enumerate(zip(leg_starts, leg_ends)):
            data[m, start:end, nx:] = state([car[leg + 1] for car in mission])
    data = data.to(device)
    with torch.no_grad():
        x_log, e_log, _ = system.rollout(controller, data, train=False)
    heading_error = torch.rad2deg(torch.atan2(torch.sin(e_log[:, -1, 2::7]), torch.cos(e_log[:, -1, 2::7]))).abs()

    n_cols = 6
    n_rows = (len(missions) + n_cols - 1) // n_cols
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(3.0 * n_cols, 4.2 * n_rows), squeeze=False)
    (x_min, x_max), (y_min, y_max) = ARENA_BOUNDS
    print("[SUMMARY] Nominal rollout per mission: final heading error [deg] per car")
    for m, (ax, mission) in enumerate(zip(axs.flat, missions)):
        draw_obstacles(ax, obstacle_centers, obstacle_covs)
        ax.add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, fill=False, color="black", linewidth=0.8))
        for i in range(n_agents):
            color = agent_color(i)
            x = x_log[m, :, 7 * i:7 * i + 3].cpu()
            ax.plot(x[:, 0], x[:, 1], color=color, linewidth=1.2)
            ax.plot(x[0, 0], x[0, 1], "s", color=color, markersize=6)
            for name in dict.fromkeys(mission[i][1:]):                         # each goal of this car once
                gx, gy, gtheta = (torch.tensor(v) for v in poses[name])
                ax.plot(gx, gy, "*", color=color, markersize=11)
                draw_pose_arrow(ax, gx, gy, gtheta, color)
            arrow = draw_pose_arrow(ax, x[-1, 0], x[-1, 1], x[-1, 2], color)
            arrow.set_facecolor("none")
        label = " | ".join("-".join(car) for car in mission)
        errors = "/".join(f"{v:.0f}" for v in heading_error[m].tolist())
        ax.set_title(f"{label}\nheading error {errors} deg", fontsize=8)
        ax.set_aspect("equal")
        ax.set_xlim(x_min - 0.2, x_max + 0.2)
        ax.set_ylim(y_min - 0.2, y_max + 0.2)
        ax.set_xticks([])
        ax.set_yticks([])
        print(f"  {label}: {errors}")
    for ax in list(axs.flat)[len(missions):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"[INFO] saved nominal mission figure to {path}")


def save_init_report_figure(path=None):
    path = path or output_file("report_setup.svg")
    # the scenario as designed: nominal start and goal(s) (poses with heading arrows),
    # and the sampling regions around them; the rollout figures show the poses of their own sample
    start, goal, obstacle_centers, obstacle_covs, _, _ = getCarInitParams(device)
    goals = getGoalSequence()
    n_agents = N_AGENTS
    colors = [agent_color(i) for i in range(n_agents)]

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
        # start: car and heading arrow at the nominal start pose
        ax.plot(start[base], start[base + 1], marker="o", markersize=7, color=color, fillstyle="none")
        draw_car(ax, start[base], start[base + 1], start[base + 2], color)
        draw_pose_arrow(ax, start[base], start[base + 1], start[base + 2], color)
        # goals: star and target-heading arrow at the nominal poses (numbered for a goal sequence)
        draw_goals(ax, goals, i, color, star_size=13)

    legend_handles = [
        Line2D([0], [0], marker="o", color="0.25", linestyle="None", markersize=7, fillstyle="none", label="Start (nominal)"),
        Line2D([0], [0], marker="*", color="0.25", linestyle="None", markersize=12, label="Goal (nominal)"),
        *route_legend(),
        Line2D([0], [0], marker=r"$\rightarrow$", color="0.25", linestyle="None", markersize=12, label="Heading (nominal start / target)"),
        Line2D([0], [0], marker="o", color="0.25", linestyle="None", markersize=13, fillstyle="none", label="Sampling region (nominal ± spread)"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True, framealpha=0.95)   # outside the axes

    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] saved init report figure to {path}")


def save_trajectory_gif(path=None, frame_stride=4, fps=20):
    path = path or output_file("trajectory.gif")
    x_log, ref, dx_log, title = simulate()
    _, _, obstacle_centers, obstacle_covs, _, _ = getCarInitParams(device)
    min_dist = EVAL_MIN_DIST
    n_agents = N_AGENTS
    colors = [agent_color(i) for i in range(n_agents)]
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
        draw_route(ax, ref, i, color, star_size=14, start=x_log[0])
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

        dist = closest_pair_distance(x_log[frame].view(n_agents, 7)[:, :2]).item()
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
    x_log, ref, dx_log, title = simulate()
    _, _, obstacle_centers, obstacle_covs, _, _ = getCarInitParams(device)
    min_dist = EVAL_MIN_DIST
    n_agents = N_AGENTS
    colors = [agent_color(i) for i in range(n_agents)]

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
        target_markers.append(draw_route(ax, ref, i, color, star_size=14, start=x_log[0]))
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

        dist = closest_pair_distance(x_log[t].view(n_agents, 7)[:, :2]).item()
        is_collision = dist < min_dist
        time_text.set_color("tab:red" if is_collision else "black")
        status = "collision" if is_collision else "clear"
        time_text.set_text(f"step {t}   distance {dist:.2f} m   {status} < {min_dist:.2f} m")
        fig.canvas.draw_idle()

    time_slider.on_changed(update)
    update(0)
    plt.show()


if __name__ == "__main__":
    print(f"[INFO] checkpoint: {TRAINED_PBR_MODEL_PATH}")
    print(f"[INFO] outputs:    {get_output_dir()}")
    if EVALUATE_MODEL and TRAINED_PBR_MODEL_PATH:
        evaluate_controller()
    elif EVALUATE_MODEL:
        print("[INFO] skipping trained-model evaluation because TRAINED_PBR_MODEL_PATH is empty.")

    # report tables (steady-state error, car-car and obstacle distances) + CSV, and the figures
    evaluate_rollout_metrics(num_runs=500, horizon=TRAIN_HORIZON)
    save_mission_nominal_figure()
    save_init_report_figure()
    save_report_figure()
    save_snapshot_report_figure()
    save_trajectory_gif()

    show_simulation()

