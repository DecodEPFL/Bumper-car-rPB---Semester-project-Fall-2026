"""
Evaluation of a trained rPB controller:  python sim.py --checkpoint <run>/checkpoints/best.pt

Everything printed is also saved as summary.txt, next to the figures in <run>/sim/<checkpoint>/.
Per-car values are written car 1/car 2. Test samples come from the training families, with another seed than training.
  1. Nominal crossing and back: one fixed run, the same for every checkpoint (comparable between runs).
  2. All test samples: final errors at the end of the training horizon and safety.
  3. The same per family.
  4. Goal-and-back: the same samples; at the end of the training horizon every car's goal becomes its start pose
     (the controller resets the car's memory at a goal change, as at the start of a training rollout).
  5. Longer horizon: random test samples run for twice the training horizon - what happens after the training horizon.
  Reference chains (GIF and CSV only): every car gets its next goal as soon as it has arrived.
Figures: nominal.png and families.png (4 samples per family): static trajectories; GIFs of the nominal crossing and back,
of goal-and-back per family (2 x 2 samples) and of chains; dxref over time (nominal, per family, goal-and-back).
CSVs: rollout_metrics.csv (per test run), chain_metrics.csv.
"""
import os
import sys
import csv
import math

from matplotlib import animation
import matplotlib.pyplot as plt
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, BASE_DIR)

import experiment_params
from experiment_params import getCarInitParams, getFamilies, getLossParams, getExtraLossParams, args as ARGS
from config import device
from controllers import PerfBoostController
from loss_functions import BumpercarLoss
from plants import BumpercarSystem, car_params
from plants.bumpercar.bumpercar_dataset import BumpercarDataset
from utils.plot_functions import agent_color


# Checkpoint to evaluate: --checkpoint <path>, default the provided one.
TRAINED_PBR_MODEL_PATH = ARGS.checkpoint or os.path.join(BASE_DIR, "experiments/bumpercar/trained_pRB/rPB_best.pt")

# Training settings of the evaluated run override sim.py's own flags: same car count, families, loss weights and
# scenario as in training. Evaluation seeds, horizons and violation distances stay sim.py's own.
TRAINING_SETTINGS = ["n_agents", "families",
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
    if "families" not in checkpoint["scenario"]:
        sys.exit("[ERROR] this checkpoint was trained before the scenario families (its scenario has no zones/families). "
                 "Evaluate it on the current families with --current-scenario.")
    experiment_params.SCENARIOS[N_AGENTS] = {key: value for key, value in checkpoint["scenario"].items() if key != "obstacles"}
    experiment_params.OBSTACLES = checkpoint["scenario"]["obstacles"]
else:
    print("[WARN] checkpoint does not store its scenario; using the current SCENARIOS and OBSTACLES")

DT = 0.04
TRAIN_HORIZON = stored_args.get("horizon", ARGS.horizon)   # steps; final errors at its end, as in the loss
EVAL_NUM_RUNS = 500                      # test samples (sections 2-4)
EVAL_RANDOM_SEED = 2                     # test samples (training uses --random-seed, 7 by default)
RANDOM_SEED = 3                          # long-horizon selection, chain goals
LONG_NUM_RUNS = 100                      # section 5: random test samples run for twice the training horizon
CHAIN_NUM_RUNS = 50                      # reference chains (chain_metrics.csv); the GIF shows the first GIF_PANELS
CHAIN_LEGS = 4                           # goals per car in a chain
ARRIVAL_POSITION = 0.10                  # [m]   success: within this distance of the goal ...
ARRIVAL_HEADING = 10.0                   # [deg] ... and this heading error
ARRIVAL_SPEED = 0.05                     # [m/s] chains: and slower than this, then the car gets its next goal
STANDSTILL_STEPS = 50                    # section 5: a car whose speed stayed ~0 over the last 50 steps (2 s) stands still
OBSTACLE_RADIUS = 1.5                    # [m] obstacle hit: car centre closer than this to an obstacle centre
EVAL_MIN_DIST = 2.0                      # [m] car-car safety distance (car radius 1 m)
ARENA_BOUNDS = ((-3.5, 3.5), (-4.0, 6.0))   # arena edges (x, y) [m] (as in BumpercarLoss)
GIF_PANELS = 4                           # samples per GIF (2 x 2 grid)
GIF_STRIDE, GIF_FPS = 5, 25              # every 5th step at 25 frames/s: every GIF runs at 5x real time
GIF_DPI = 100                            # GIF resolution


def get_output_dir(checkpoint_path=None):
    """
    Folder for the summary, figures and CSVs of one checkpoint, named after its file and epoch (best.pt changes during
    training, so every evaluated epoch gets its own folder):
    - checkpoint of a training run (.../<run>/checkpoints/<name>.pt) -> <run>/sim/<name>_ep<epoch>/
    - any other checkpoint (e.g. the provided rPB_best.pt)             -> saved_results/sim_<name>_ep<epoch>/
    """
    checkpoint_path = os.path.abspath(checkpoint_path or TRAINED_PBR_MODEL_PATH)
    checkpoint_dir = os.path.dirname(checkpoint_path)
    name = os.path.splitext(os.path.basename(checkpoint_path))[0]
    if checkpoint.get("epoch") is not None:
        name = f"{name}_ep{checkpoint['epoch']}"
    if os.path.basename(checkpoint_dir) == "checkpoints":
        return os.path.join(os.path.dirname(checkpoint_dir), "sim", name)
    return os.path.join(BASE_DIR, "experiments", "bumpercar", "saved_results", f"sim_{name}")


def output_file(filename):
    path = os.path.join(get_output_dir(), filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


# ---------------------------------------------------------------------------------------------------------------------
# set-up: system, controller, data, loss
# ---------------------------------------------------------------------------------------------------------------------

def make_system():
    return BumpercarSystem(params=car_params, x_init=None, u_init=None, n_agents=N_AGENTS, dt=DT,
                           turn_speed=ARGS.turn_speed).to(device)


def load_controller(system, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    args = checkpoint.get("args", {})
    args = args if isinstance(args, dict) else vars(args)
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


def make_dataset(horizon, random_seed):
    zones, families, sampling = getFamilies()
    return BumpercarDataset(random_seed=random_seed, horizon=horizon, zones=zones, families=families,
                            sampling=sampling, n_agents=N_AGENTS)


def make_loss_fn(data):
    _, _, obstacle_centers, obstacle_covs = getCarInitParams(device)
    Q, Q_final, Qs, alpha_col, alpha_obst, alpha_u, position_deadzone, steady_state_velocity_radius, min_dist = getLossParams(device)
    return BumpercarLoss(
        Q=Q, Q_final=Q_final, Qs=Qs, alpha_u=alpha_u, xbar=data[0, :, 7 * N_AGENTS:], loss_bound=None, sat_bound=None,
        alpha_col=alpha_col, alpha_obst=alpha_obst, obstacle_centers=obstacle_centers, obstacle_covs=obstacle_covs,
        min_dist=min_dist, n_agents=N_AGENTS, position_deadzone=position_deadzone,
        steady_state_velocity_radius=steady_state_velocity_radius, **getExtraLossParams(),
    )


def obstacle_centers_tensor():
    _, _, obstacle_centers, _ = getCarInitParams(device)
    return torch.stack([center.flatten() for center in obstacle_centers])            # [n_obstacles, 2]


def rollout(system, controller, data):
    """Closed-loop rollout without gradients: states, errors (reference - state) and dxref over time."""
    with torch.no_grad():
        x_log, e_log, u_log = system.rollout(controller, data)
    return {"x_log": x_log, "e_log": e_log, "u_log": u_log}


# ---------------------------------------------------------------------------------------------------------------------
# zones and new goals during a run (family labels, chains)
# ---------------------------------------------------------------------------------------------------------------------

def family_zones(dataset):
    """Zones the evaluated families use."""
    return sorted({zone for variants in dataset.families.values() for variant in variants for pair in variant for zone in pair})


def zone_of(dataset, position):
    """Name of the zone (of those the families use) that contains position (x, y); None if none does."""
    for name in family_zones(dataset):
        x_min, x_max, y_min, y_max = dataset.zones[name]
        if x_min <= position[0] <= x_max and y_min <= position[1] <= y_max:
            return name
    return None


def family_labels(data, dataset):
    """Family of every sample, from the zones of each car's start and first goal ("other" if no variant matches)."""
    nx = 7 * N_AGENTS
    lookup = {tuple(tuple(pair) for pair in variant): name
              for name, variants in dataset.families.items() for variant in variants}
    labels = []
    for s in range(data.shape[0]):
        key = tuple((zone_of(dataset, data[s, 0, 7 * i:7 * i + 2].cpu()), zone_of(dataset, data[s, 0, nx + 7 * i:nx + 7 * i + 2].cpu()))
                    for i in range(N_AGENTS))
        labels.append(lookup.get(key, "other"))
    return labels


def sample_goal(dataset, zone, start, others, max_tries=200):
    """
    One new goal pose (x, y, theta) for one car during a run, by the training rules: uniform in zone, at least min_move
    from the car's position start (x, y), at least car_spacing from every point of others [m, 2] (the other cars' goals
    and positions), heading uniform over the feasible directions. None if no position is found in max_tries draws.
    """
    for _ in range(max_tries):
        goal = dataset.sample_position(dataset.zones[zone])
        if torch.linalg.norm(goal - start) < dataset.min_move:
            continue
        if len(others) > 0 and torch.linalg.norm(others - goal, dim=-1).min() < dataset.car_spacing:
            continue
        return torch.cat((goal, dataset.sample_feasible_heading(goal).view(1)))
    return None


def make_test_data(dataset):
    """
    EVAL_NUM_RUNS test samples over twice the training horizon: the sampled goal during the first training horizon,
    back to the start pose (position and heading) during the second (goal-and-back).
    """
    T, nx = TRAIN_HORIZON, 7 * N_AGENTS
    data = dataset._generate_data(EVAL_NUM_RUNS)
    data[:, T:, nx:] = data[:, 0:1, :nx]                                               # the start pose as the return goal
    return data.to(device)


def make_nominal_data():
    """Nominal crossing of the scenario (fixed start and goal poses, no sampling) and back, as one sample
    [1, 2 * TRAIN_HORIZON, 2*nx]: the nominal goal during the first training horizon, then back to the nominal start
    pose (position and heading)."""
    T, nx = TRAIN_HORIZON, 7 * N_AGENTS
    x0, x_final, _, _ = getCarInitParams(device)
    data = torch.zeros(1, 2 * T, 2 * nx)
    data[:, 0, :nx] = x0
    data[:, :T, nx:] = x_final.view(1, 1, -1)
    data[:, T:, nx:] = x0.view(1, 1, -1)
    return data.to(device)


# ---------------------------------------------------------------------------------------------------------------------
# metrics per run and car
# ---------------------------------------------------------------------------------------------------------------------

def wrapped_degrees(angle):
    """|angle| wrapped to [0, 180] degrees."""
    return torch.rad2deg(torch.atan2(torch.sin(angle), torch.cos(angle))).abs()


def errors_at(x_log, data, step):
    """Position error [m] and heading error [deg] of every car at step: [S, n] each."""
    S, nx = x_log.shape[0], 7 * N_AGENTS
    state = x_log[:, step].reshape(S, N_AGENTS, 7)
    goal = data[:, step, nx:].reshape(S, N_AGENTS, 7)
    return torch.linalg.norm(goal[..., 0:2] - state[..., 0:2], dim=-1), wrapped_degrees(goal[..., 2] - state[..., 2])


def at_goal(x_log, goal_log, with_speed=False):
    """Every step and car: within ARRIVAL_POSITION and ARRIVAL_HEADING of its goal (and slower than ARRIVAL_SPEED)."""
    S, T = x_log.shape[0], x_log.shape[1]
    state = x_log.reshape(S, T, N_AGENTS, 7)
    goal = goal_log.reshape(S, T, N_AGENTS, 7)
    inside = ((torch.linalg.norm(goal[..., 0:2] - state[..., 0:2], dim=-1) < ARRIVAL_POSITION)
              & (wrapped_degrees(goal[..., 2] - state[..., 2]) < ARRIVAL_HEADING))
    if with_speed:
        inside = inside & (state[..., 3].abs() < ARRIVAL_SPEED)
    return inside                                                                     # [S, T, n]


def safety(x_log, start, end):
    """Between steps start and end: closest car-car distance per run [S], closest obstacle-centre distance per run and
    car [S, n]."""
    S = x_log.shape[0]
    positions = x_log[:, start:end].reshape(S, end - start, N_AGENTS, 7)[..., 0:2]   # [S, T, n, 2]
    distances = torch.cdist(positions, positions) + 1e6 * torch.eye(N_AGENTS, device=positions.device)
    obstacle = torch.linalg.norm(positions.unsqueeze(3) - obstacle_centers_tensor().view(1, 1, 1, -1, 2), dim=-1)
    return {"car_distance": distances.amin((1, 2, 3)), "obstacle_distance": obstacle.amin((1, 3))}


def leg_metrics(run, data, start, end):
    """Final errors at the last step of a leg [S, n] and safety over the leg."""
    pos, head = errors_at(run["x_log"], data, end - 1)
    return {"pos": pos, "head": head, **safety(run["x_log"], start, end)}


def leg_loss(loss_fn, run, start, end, runs):
    """Loss (as in training) of the leg from step start to end over the runs in runs, and its terms."""
    with torch.no_grad():
        loss = loss_fn.forward(run["x_log"][runs, start:end], run["u_log"][runs, start:end], run["e_log"][runs, start:end])
    return loss.item(), dict(loss_fn.last_terms)


# ---------------------------------------------------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------------------------------------------------

def per_car(values, fmt):
    """Values per car [n] as "a/b" ("-" for nan)."""
    return "/".join("-" if math.isnan(v) else fmt.format(v) for v in values.flatten().tolist())


def rms(values):
    return values.float().pow(2).mean(0).sqrt()


def print_table(title, guide, header, rows):
    """Table with a title, a few guide lines (what it shows, how to read it), a header and rows of strings."""
    widths = [max(len(str(row[k])) for row in [header] + rows) + 2 for k in range(len(header))]
    print(f"\n[{title}]")
    for line in guide:
        print(f"  {line}")
    for row in [header] + rows:
        print("  " + str(row[0]).ljust(widths[0]) + "".join(str(cell).rjust(width) for cell, width in zip(row[1:], widths[1:])))


RESULT_HEADER = ["runs", "success [%]", "pos RMSE [cm]", "heading RMS [deg]", "car-car < 2 m [runs]", "obstacle hits [runs]"]


def result_cells(metrics, mask):
    """Table cells of the runs in mask: success, position RMSE, heading RMS per car; runs with a car-car distance below
    EVAL_MIN_DIST; runs in which any car hit an obstacle."""
    pos, head = metrics["pos"][mask], metrics["head"][mask]
    success = ((pos < ARRIVAL_POSITION) & (head < ARRIVAL_HEADING)).float().mean(0) * 100
    return [str(int(mask.sum())), per_car(success, "{:.0f}"), per_car(rms(pos) * 100, "{:.1f}"), per_car(rms(head), "{:.1f}"),
            str(int((metrics["car_distance"][mask] < EVAL_MIN_DIST).sum())),
            str(int((metrics["obstacle_distance"][mask] < OBSTACLE_RADIUS).any(-1).sum()))]


def result_rows(metrics, groups):
    """One row per group (name, mask)."""
    return [[name] + result_cells(metrics, mask) for name, mask in groups]


def family_groups(labels, families):
    names = [name for name in families if name in labels] + (["other"] if "other" in labels else [])
    return [(name, torch.tensor([label == name for label in labels])) for name in names]


SUCCESS_GUIDE = (f"success = within {ARRIVAL_POSITION * 100:.0f} cm and {ARRIVAL_HEADING:.0f} deg of the goal at the end; per car: car 1/car 2. "
                 f"car-car < {EVAL_MIN_DIST:g} m and obstacle hits (car centre < {OBSTACLE_RADIUS:g} m from an obstacle centre) count runs.")


def print_nominal(nominal):
    rows = []
    for leg, n in zip(("crossing", "back"), nominal["legs"]):
        hit = n["obstacle_distance"] < OBSTACLE_RADIUS
        rows.append([leg, per_car(n["pos"] * 100, "{:.1f}"), per_car(n["head"], "{:.1f}"),
                     "/".join("yes" if h else "no" for h in hit.tolist()), per_car(n["obstacle_distance"], "{:.2f}"),
                     "yes" if n["car_distance"] < EVAL_MIN_DIST else "no", f"{n['car_distance']:.2f}"])
    print_table("1. NOMINAL CROSSING AND BACK", [
        f"One fixed run (the same for every checkpoint): the nominal crossing for {TRAIN_HORIZON * DT:.0f} s, then back to the nominal",
        f"start poses for another {TRAIN_HORIZON * DT:.0f} s. Final errors at the end of each leg, per car: car 1/car 2."],
        ["leg", "pos error [cm]", "heading error [deg]", "obstacle hit", "closest obstacle [m]", "car-car < 2 m",
         "closest car-car [m]"], rows)


def print_test_results(test):
    T = TRAIN_HORIZON
    leg1 = test["legs"][0]
    everything = torch.ones(EVAL_NUM_RUNS, dtype=torch.bool)
    loss, terms = test["loss"][0]
    print_table("2. ALL TEST SAMPLES", [
        f"{EVAL_NUM_RUNS} test samples of the training families, final errors at the end of the training horizon ({T * DT:.0f} s).",
        SUCCESS_GUIDE,
        f"Loss (as in training): {loss:.2f}  [" + ", ".join(f"{name} {value:.2f}" for name, value in terms.items()) + "]"],
        [""] + RESULT_HEADER, result_rows(leg1, [("all", everything)]))
    print_table("3. PER FAMILY", ["The same, per family."],
                ["family"] + RESULT_HEADER, result_rows(leg1, family_groups(test["labels"], test["families"])))


def print_goal_and_back(test):
    T = TRAIN_HORIZON
    leg2 = test["legs"][1]
    rows = []
    for leg, (loss, terms) in enumerate(test["loss"]):
        rows.append([f"leg {leg + 1}", f"{loss:.2f}"] + [f"{value:.2f}" for value in terms.values()])
    print_table("4. GOAL-AND-BACK", [
        f"The same samples; at {T * DT:.0f} s every car's goal becomes its start pose (position and heading): leg 2, another {T * DT:.0f} s.",
        "The controller resets the car's memory at its goal change -> leg 2 should look like a fresh training rollout.",
        "Leg 1 = sections 2 and 3. Loss as in training, each leg on its own:"],
        ["leg", "loss"] + list(test["loss"][0][1].keys()), rows)
    everything = torch.ones(EVAL_NUM_RUNS, dtype=torch.bool)
    print_table("4. GOAL-AND-BACK: return leg", [SUCCESS_GUIDE + " Errors at the end of leg 2."],
                ["family"] + RESULT_HEADER,
                result_rows(leg2, [("all", everything)] + family_groups(test["labels"], test["families"])))


def print_long_horizon(long):
    T = TRAIN_HORIZON
    m = long
    rows = [
        [f"at the end of the training horizon ({T * DT:.0f} s): success [%] / pos RMSE [cm] / heading RMS [deg]",
         " / ".join([per_car(m["success_T"], "{:.0f}"), per_car(rms(m["pos_T"]) * 100, "{:.1f}"), per_car(rms(m["head_T"]), "{:.1f}")])],
        [f"at twice the training horizon ({2 * T * DT:.0f} s): success [%] / pos RMSE [cm] / heading RMS [deg]",
         " / ".join([per_car(m["success_end"], "{:.0f}"), per_car(rms(m["pos_end"]) * 100, "{:.1f}"), per_car(rms(m["head_end"]), "{:.1f}")])],
        ["arrived only after the training horizon [cars]", per_car(m["late"].sum(0).float(), "{:.0f}")],
        ["at the goal at the training horizon, but not at the end [cars]", per_car(m["left"].sum(0).float(), "{:.0f}")],
        ["not at the goal at the end, still moving [cars]", per_car(m["moving"].sum(0).float(), "{:.0f}")],
        [f"not at the goal at the end, standing still for {STANDSTILL_STEPS * DT:.0f} s (will not get there) [cars]",
         per_car(m["standing"].sum(0).float(), "{:.0f}")],
        [f"remaining |dxref| over the last {STANDSTILL_STEPS * DT:.0f} s: median / max [m]",
         per_car(m["dxref_end"].median(0).values, "{:.3f}") + " / " + per_car(m["dxref_end"].max(0).values, "{:.3f}")],
        ["after the training horizon: car-car < 2 m / obstacle hits [runs]",
         f"{int((m['car_distance'] < EVAL_MIN_DIST).sum())} / {int((m['obstacle_distance'] < OBSTACLE_RADIUS).any(-1).sum())}"],
    ]
    print_table("5. LONGER HORIZON", [
        f"{LONG_NUM_RUNS} random test samples, the goal kept for twice the training horizon ({2 * T * DT:.0f} s): what happens once the",
        "training horizon is over, as on the real system (rPB; per car: car 1/car 2)."],
        ["", "value"], rows)


# ---------------------------------------------------------------------------------------------------------------------
# the tests
# ---------------------------------------------------------------------------------------------------------------------

def run_nominal(system, controller):
    """Nominal crossing and back: logs, and per leg the final errors per car and safety."""
    T = TRAIN_HORIZON
    data = make_nominal_data()
    run = rollout(system, controller, data)
    legs = [leg_metrics(run, data, 0, T), leg_metrics(run, data, T, 2 * T)]
    return {"data": data, **run, "legs": [{key: value[0] for key, value in metrics.items()} for metrics in legs]}


def run_test(system, controller):
    """Sections 2-4: the test samples with the way back to their start, one rollout over both legs."""
    T = TRAIN_HORIZON
    dataset = make_dataset(2 * T, EVAL_RANDOM_SEED)
    data = make_test_data(dataset)
    run = rollout(system, controller, data)
    loss_fn = make_loss_fn(data)
    everything = torch.ones(EVAL_NUM_RUNS, dtype=torch.bool)
    return {"data": data, "dataset": dataset, "labels": family_labels(data, dataset), "families": list(dataset.families),
            "run": run, "legs": [leg_metrics(run, data, 0, T), leg_metrics(run, data, T, 2 * T)],
            "loss": [leg_loss(loss_fn, run, 0, T, everything), leg_loss(loss_fn, run, T, 2 * T, everything)]}


def run_long_horizon(system, controller, test):
    """Section 5: LONG_NUM_RUNS random test samples with their first goal kept for twice the training horizon."""
    T, nx = TRAIN_HORIZON, 7 * N_AGENTS
    runs = torch.randperm(EVAL_NUM_RUNS, generator=torch.Generator().manual_seed(RANDOM_SEED))[:LONG_NUM_RUNS].sort().values
    data = test["data"][runs].clone()
    data[:, :, nx:] = data[:, :1, nx:]                                                 # first goal over both horizons
    run = rollout(system, controller, data)
    L = LONG_NUM_RUNS
    inside = at_goal(run["x_log"], data[:, :, nx:]).cpu()                              # [L, 2T, n]
    speed = run["x_log"][:, -STANDSTILL_STEPS:].reshape(L, STANDSTILL_STEPS, N_AGENTS, 7)[..., 3].abs().cpu()
    at_end, at_T = inside[:, -1], inside[:, T - 1]
    pos_T, head_T = errors_at(run["x_log"], data, T - 1)
    pos_end, head_end = errors_at(run["x_log"], data, 2 * T - 1)
    return {"runs": runs, "pos_T": pos_T.cpu(), "head_T": head_T.cpu(), "pos_end": pos_end.cpu(), "head_end": head_end.cpu(),
            "success_T": at_T.float().mean(0) * 100, "success_end": at_end.float().mean(0) * 100,
            "late": ~at_T & at_end, "left": at_T & ~at_end,
            "moving": ~at_end & (speed[:, -1] > ARRIVAL_SPEED), "standing": ~at_end & (speed < 1e-3).all(1),
            "dxref_end": torch.linalg.norm(run["u_log"][:, -STANDSTILL_STEPS:].reshape(L, STANDSTILL_STEPS, N_AGENTS, 2), dim=-1).amax(1).cpu(),
            **{key: value.cpu() for key, value in safety(run["x_log"], T, 2 * T).items()}}


def run_chain(system, controller, data, dataset):
    """
    Reference chains: every car drives CHAIN_LEGS goals, alternating bottom <-> top (random side, by the training rules).
    A car gets its next goal as soon as it has arrived (within ARRIVAL_POSITION and ARRIVAL_HEADING, slower than
    ARRIVAL_SPEED) or after a timeout of one training horizon, independently of the other car. The rollout loop is the
    one of BumpercarSystem.rollout, with the reference decided step by step.
    Returns the logs [S, steps, .] and one record per leg (run, car, leg, first and last step, how it ended).
    """
    torch.manual_seed(RANDOM_SEED)
    S, nx, n = data.shape[0], 7 * N_AGENTS, N_AGENTS
    steps = CHAIN_LEGS * TRAIN_HORIZON
    zones = family_zones(dataset)
    top_zones = [name for name in zones if dataset.zones[name][2] > 0]
    bottom_zones = [name for name in zones if dataset.zones[name][3] < 0]

    controller.reset()
    system.positionPID.reset(batch_size=S, device=data.device, dtype=data.dtype)
    x = data[:, 0:1, :nx].clone()
    u = system.u_init.detach().clone().repeat(S, 1, 1)
    v = torch.zeros_like(u)
    w = torch.zeros_like(x)
    goal = data[:, 0, nx:].clone()                                                   # [S, nx] current goal per car
    leg = torch.zeros(S, n, dtype=torch.long)
    leg_start = torch.zeros(S, n, dtype=torch.long)
    pos_indices = system.pos_indices()
    x_steps, u_steps, goal_steps, records = [], [], [], []
    with torch.no_grad():
        for t in range(steps):
            xbar = goal[:, pos_indices].unsqueeze(1)
            theta_ref = goal[:, 2::7].unsqueeze(1)
            x, v = system.forward(t=t, x=x, v=v, u=u, w=w, xbar=xbar)
            u = controller(x, v, xbar, theta_ref)
            x_steps.append(x)
            u_steps.append(u)
            goal_steps.append(goal.clone().unsqueeze(1))

            # end of a leg: arrived or timeout (the last leg ends with the run)
            arrived = at_goal(x, goal.unsqueeze(1), with_speed=True)[:, 0].cpu()     # [S, n]
            timed_out = (t + 1 - leg_start) >= TRAIN_HORIZON
            switch = (arrived | timed_out) & (leg < CHAIN_LEGS - 1)
            for s, i in switch.nonzero().tolist():
                # next goal: the other band, random side; apart from the other cars' goals and positions
                state = x[s, 0].view(n, 7).cpu()
                target = goal[s].view(n, 7).cpu()
                candidates = list(bottom_zones if target[i, 1] > 0 else top_zones)
                candidates = [candidates[k] for k in torch.randperm(len(candidates)).tolist()]
                others_idx = [j for j in range(n) if j != i]
                others = torch.cat([target[others_idx, 0:2], state[others_idx, 0:2]])
                new_goal = None
                for zone in candidates:
                    new_goal = sample_goal(dataset, zone, state[i, 0:2], others)
                    if new_goal is not None:
                        break
                if new_goal is None:
                    continue                                                          # no free goal now: try again next step
                records.append({"run": s, "car": i, "leg": int(leg[s, i]) + 1, "start": int(leg_start[s, i]), "end": t + 1,
                                "ended_by": "arrival" if arrived[s, i] else "timeout"})
                goal[s, 7 * i:7 * i + 7] = torch.cat((new_goal, torch.zeros(4))).to(goal.device)
                leg[s, i] += 1
                leg_start[s, i] = t + 1
    for s in range(S):
        for i in range(n):
            records.append({"run": s, "car": i, "leg": int(leg[s, i]) + 1, "start": int(leg_start[s, i]), "end": steps,
                            "ended_by": "end of run"})
    x_log, u_log, goal_log = torch.cat(x_steps, 1), torch.cat(u_steps, 1), torch.cat(goal_steps, 1)

    # per leg: first step at the goal (position, heading, speed) and the errors at the leg's last step
    inside = at_goal(x_log, goal_log, with_speed=True).cpu()                        # [S, steps, n]
    for record in records:
        s, i, start, end = record["run"], record["car"], record["start"], record["end"]
        hits = inside[s, start:end, i].nonzero()
        record["time_to_goal"] = (hits[0, 0].item() + 1) * DT if len(hits) else float("inf")
        state = x_log[s, end - 1].view(n, 7)[i].cpu()
        target = goal_log[s, end - 1].view(n, 7)[i].cpu()
        record["pos_error"] = torch.linalg.norm(target[0:2] - state[0:2]).item()
        record["heading_error"] = wrapped_degrees(target[2] - state[2]).item()
        record["goal"] = target[0:3].tolist()
    return {"x_log": x_log, "u_log": u_log, "goal_log": goal_log, "records": records}


# ---------------------------------------------------------------------------------------------------------------------
# CSVs
# ---------------------------------------------------------------------------------------------------------------------

def save_rollout_metrics(test, long):
    """One row per test run: family, start and goal poses, final errors of both legs (leg 2: back to the start pose),
    long-horizon errors, distances."""
    path = output_file("rollout_metrics.csv")
    data, nx = test["data"], 7 * N_AGENTS
    long_index = {run: k for k, run in enumerate(long["runs"].tolist())}
    with open(path, "w", newline="") as csvfile:
        writer = None
        for s in range(data.shape[0]):
            row = {"run": s, "family": test["labels"][s]}
            for leg, metrics in enumerate(test["legs"]):
                row[f"leg_{leg + 1}_closest_car_distance"] = round(metrics["car_distance"][s].item(), 3)
            for i in range(N_AGENTS):
                c = f"_car_{i + 1}"
                for name, pose in (("start", data[s, 0, 7 * i:7 * i + 3]), ("goal", data[s, 0, nx + 7 * i:nx + 7 * i + 3])):
                    row.update({f"{name}_x{c}": round(pose[0].item(), 3), f"{name}_y{c}": round(pose[1].item(), 3),
                                f"{name}_heading_deg{c}": round(math.degrees(pose[2].item()), 1)})
                for leg, metrics in enumerate(test["legs"]):
                    row[f"leg_{leg + 1}_pos_error_cm{c}"] = round(metrics["pos"][s, i].item() * 100, 2)
                    row[f"leg_{leg + 1}_heading_error_deg{c}"] = round(metrics["head"][s, i].item(), 2)
                    row[f"leg_{leg + 1}_closest_obstacle{c}"] = round(metrics["obstacle_distance"][s, i].item(), 3)
                k = long_index.get(s)
                row[f"long_pos_error_cm{c}"] = "" if k is None else round(long["pos_end"][k, i].item() * 100, 2)
                row[f"long_heading_error_deg{c}"] = "" if k is None else round(long["head_end"][k, i].item(), 2)
            if writer is None:
                writer = csv.DictWriter(csvfile, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
    print(f"[INFO] saved {path}")


def save_chain_metrics(chain):
    path = output_file("chain_metrics.csv")
    fields = ["run", "car", "leg", "start_s", "end_s", "ended_by", "time_to_goal_s", "pos_error_cm", "heading_error_deg",
              "goal_x", "goal_y", "goal_heading_deg"]
    with open(path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for r in sorted(chain["records"], key=lambda r: (r["run"], r["car"], r["leg"])):
            writer.writerow({"run": r["run"], "car": r["car"] + 1, "leg": r["leg"], "start_s": round(r["start"] * DT, 2),
                             "end_s": round(r["end"] * DT, 2), "ended_by": r["ended_by"], "time_to_goal_s": r["time_to_goal"],
                             "pos_error_cm": round(r["pos_error"] * 100, 2), "heading_error_deg": round(r["heading_error"], 2),
                             "goal_x": round(r["goal"][0], 3), "goal_y": round(r["goal"][1], 3),
                             "goal_heading_deg": round(math.degrees(r["goal"][2]), 1)})
    print(f"[INFO] saved {path}")


# ---------------------------------------------------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------------------------------------------------

def car_polygon(x, y, theta):
    """Corners of the car marker: an arrow-shaped body (0.64 m long, 0.36 m wide) pointing in the driving direction."""
    body_x = torch.tensor([0.34, 0.08, -0.30, -0.30, 0.08])
    body_y = torch.tensor([0.0, 0.18, 0.18, -0.18, -0.18])
    c, s = math.cos(float(theta)), math.sin(float(theta))
    return torch.stack((float(x) + c * body_x - s * body_y, float(y) + s * body_x + c * body_y), -1).tolist()


def draw_car(ax, x, y, theta, color):
    patch = Polygon(car_polygon(x, y, theta), closed=True, facecolor=color, edgecolor="black", linewidth=1.0, zorder=7)
    ax.add_patch(patch)
    return patch


def draw_pose_arrow(ax, x, y, theta, color, length=0.5, alpha=1.0, linewidth=2.0, head=14):
    x, y, theta = float(x), float(y), float(theta)
    arrow = FancyArrowPatch((x, y), (x + length * math.cos(theta), y + length * math.sin(theta)), arrowstyle="-|>",
                            mutation_scale=head, color=color, linewidth=linewidth, alpha=alpha, zorder=6)
    ax.add_patch(arrow)
    return arrow


def draw_arena(ax):
    """Arena edge, sampling zones of the evaluated families (dotted) and the obstacles; no ticks."""
    zones, families, _ = getFamilies()
    used = sorted({zone for variants in families.values() for variant in variants for pair in variant for zone in pair})
    for name in used:
        x_min, x_max, y_min, y_max = zones[name]
        ax.add_patch(Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, fill=False, linestyle=":", color="0.55", linewidth=0.8))
    for center in obstacle_centers_tensor().cpu():
        ax.add_patch(Circle((center[0].item(), center[1].item()), OBSTACLE_RADIUS, facecolor="0.35", edgecolor="black", alpha=0.25))
    (x_min, x_max), (y_min, y_max) = ARENA_BOUNDS
    ax.add_patch(Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, fill=False, color="black", linewidth=0.8))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(x_min - 0.1, x_max + 0.1)
    ax.set_ylim(y_min - 0.1, y_max + 0.1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def errors_text(pos, head):
    """Final errors per car "car 1: <pos> cm, <heading>° | car 2: ..." from pos [m], head [deg] per car."""
    return " | ".join(f"car {i + 1}: {pos[i] * 100:.0f} cm, {head[i]:.0f}°" for i in range(N_AGENTS))


GIF_LEGEND = [Line2D([0], [0], marker=r"$\rightarrow$", color="0.3", linestyle="None", markersize=12, label="car (points where it drives)"),
              Line2D([0], [0], marker="o", color="0.3", linestyle="None", fillstyle="none", label="start + start heading"),
              Line2D([0], [0], marker="*", color="0.3", linestyle="None", markersize=10, label="current goal + target heading"),
              Line2D([0], [0], marker="o", color="tab:red", alpha=0.3, linestyle="None", markersize=12,
                     label=f"1 m car radius (red: cars closer than {EVAL_MIN_DIST:g} m)")]


def save_grid_gif(panels, filename, title):
    """
    Animation of 1 to 4 runs side by side (2 x 2 grid). panels: list of {"x_log": [T, nx], "goal_log": [T, nx],
    "title": str}. Per car: path, car marker pointing in its driving direction, 1 m radius (red while the cars are closer
    than EVAL_MIN_DIST), start pose, current goal with its target heading (changes at a goal change).
    """
    n_panels = len(panels)
    cols = 1 if n_panels == 1 else 2
    rows = math.ceil(n_panels / cols)
    fig, axs = plt.subplots(rows, cols, figsize=(4.4 * cols, 5.8 * rows + 0.6), squeeze=False)
    T = panels[0]["x_log"].shape[0]
    frames = list(range(0, T, GIF_STRIDE)) + ([T - 1] if (T - 1) % GIF_STRIDE else [])
    drawn = []
    for ax, panel in zip(axs.flat, panels):
        x_log, goal_log = panel["x_log"].cpu(), panel["goal_log"].cpu()
        draw_arena(ax)
        ax.set_title(panel["title"], fontsize=8)
        cars = []
        for i in range(N_AGENTS):
            base, color = 7 * i, agent_color(i)
            ax.plot(x_log[0, base], x_log[0, base + 1], marker="o", markersize=6, color=color, fillstyle="none")
            draw_pose_arrow(ax, x_log[0, base], x_log[0, base + 1], x_log[0, base + 2], color, alpha=0.6)
            # artists are created once and only moved per frame; the goal arrow is redrawn only when the goal changes
            cars.append({"path": ax.plot([], [], color=color, linewidth=1.8)[0],
                         "radius": ax.add_patch(Circle((0, 0), EVAL_MIN_DIST / 2, facecolor=color, edgecolor=color, alpha=0.1)),
                         "goal": ax.plot([], [], marker="*", markersize=11, color=color, linestyle="None", zorder=8)[0],
                         "goal_arrow": None, "shown_goal": None,
                         "car": draw_car(ax, x_log[0, base], x_log[0, base + 1], x_log[0, base + 2], color)})
        drawn.append((ax, x_log, goal_log, cars))
    for ax in list(axs.flat)[n_panels:]:
        ax.axis("off")
    suptitle = fig.suptitle("", fontsize=10)
    fig.legend(handles=GIF_LEGEND, loc="lower center", ncol=2, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))

    def update(frame):
        for ax, x_log, goal_log, cars in drawn:
            positions = x_log[frame].view(N_AGENTS, 7)[:, 0:2]
            too_close = (torch.cdist(positions, positions) + 1e6 * torch.eye(N_AGENTS)).min().item() < EVAL_MIN_DIST
            for i, car in enumerate(cars):
                base, color = 7 * i, agent_color(i)
                car["path"].set_data(x_log[:frame + 1, base], x_log[:frame + 1, base + 1])
                car["radius"].center = (x_log[frame, base].item(), x_log[frame, base + 1].item())
                car["radius"].set_facecolor("tab:red" if too_close else color)
                car["radius"].set_alpha(0.3 if too_close else 0.1)
                car["car"].set_xy(car_polygon(x_log[frame, base], x_log[frame, base + 1], x_log[frame, base + 2]))
                goal = tuple(goal_log[frame, base:base + 3].tolist())
                if goal != car["shown_goal"]:                                         # first frame or a goal change
                    if car["goal_arrow"] is not None:
                        car["goal_arrow"].remove()
                    car["goal"].set_data([goal[0]], [goal[1]])
                    car["goal_arrow"] = draw_pose_arrow(ax, goal[0], goal[1], goal[2], color)
                    car["shown_goal"] = goal
        suptitle.set_text(f"{title}   t = {frame * DT:.1f} s")
        return []

    anim = animation.FuncAnimation(fig, update, frames=frames, interval=1000 / GIF_FPS, blit=False)
    path = output_file(filename)
    anim.save(path, writer="pillow", fps=GIF_FPS, dpi=GIF_DPI)
    plt.close(fig)
    print(f"[INFO] saved {path}")


def save_dxref_figure(rows, filename, title):
    """dxref per car over time: one row of panels per entry of rows = [(label, u_log [T, 2n], marks [s]), ...],
    one panel per car; dashed red lines at the times in marks."""
    fig, axs = plt.subplots(len(rows), N_AGENTS, figsize=(6.5 * N_AGENTS, 2.4 * len(rows) + 0.8), squeeze=False, sharex=True)
    for row, (label, u, marks) in enumerate(rows):
        t = torch.arange(u.shape[0]) * DT
        for i in range(N_AGENTS):
            ax, du = axs[row, i], u[:, 2 * i:2 * i + 2].cpu()
            ax.plot(t, du[:, 0], label="dX", linewidth=1.0)
            ax.plot(t, du[:, 1], label="dY", linewidth=1.0)
            for mark in marks:
                ax.axvline(mark, color="tab:red", linestyle="--", linewidth=0.8)
            ax.set_title(f"{label} - car {i + 1}", fontsize=9)
            ax.set_ylabel("offset [m]")
            ax.grid(True, alpha=0.3)
    for i in range(N_AGENTS):
        axs[-1, i].set_xlabel("time [s]")
    axs[0, -1].legend(fontsize=8, loc="upper right")
    fig.suptitle(title)
    fig.tight_layout()
    path = output_file(filename)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"[INFO] saved {path}")


def save_trajectory_png(rows, filename, title):
    """
    Static trajectories: rows of panels, each panel {"x_log": [T, nx], "goal_log": [T, nx], "title": str}. Per car: path,
    start pose, every goal with its target heading (numbered if there are several), car marker at its final pose.
    """
    n_rows, n_cols = len(rows), max(len(row) for row in rows)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(3.4 * n_cols, 4.6 * n_rows + 0.6), squeeze=False)
    for r, row in enumerate(rows):
        for c in range(n_cols):
            ax = axs[r, c]
            if c >= len(row):
                ax.axis("off")
                continue
            panel = row[c]
            x_log, goal_log = panel["x_log"].cpu(), panel["goal_log"].cpu()
            draw_arena(ax)
            ax.set_title(panel["title"], fontsize=8)
            for i in range(N_AGENTS):
                base, color = 7 * i, agent_color(i)
                ax.plot(x_log[:, base], x_log[:, base + 1], color=color, linewidth=1.4)
                ax.plot(x_log[0, base], x_log[0, base + 1], marker="o", markersize=6, color=color, fillstyle="none")
                draw_pose_arrow(ax, x_log[0, base], x_log[0, base + 1], x_log[0, base + 2], color, alpha=0.6)
                g = goal_log[:, base:base + 3]
                changes = [0] + ((g[1:] != g[:-1]).any(-1).nonzero().flatten() + 1).tolist()
                for k, step in enumerate(changes):
                    draw_pose_arrow(ax, g[step, 0], g[step, 1], g[step, 2], color)
                    ax.plot(g[step, 0], g[step, 1], marker="*", markersize=10, color=color, zorder=8)
                    if len(changes) > 1:
                        ax.annotate(str(k + 1), (g[step, 0].item(), g[step, 1].item()), textcoords="offset points",
                                    xytext=(6, -10), color=color, fontsize=8, fontweight="bold")
                draw_car(ax, x_log[-1, base], x_log[-1, base + 1], x_log[-1, base + 2], color)
    handles = GIF_LEGEND[:3] + [Line2D([0], [0], color="0.3", linewidth=1.4, label="path (car at its final pose)")]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=8, frameon=False)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    path = output_file(filename)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"[INFO] saved {path}")


def gif_runs(test):
    """The first GIF_PANELS test runs of every family: {family: [run, ...]}."""
    examples = {}
    for s, label in enumerate(test["labels"]):
        if len(examples.setdefault(label, [])) < GIF_PANELS:
            examples[label].append(s)
    return examples


def save_figures(nominal, test, chain):
    T, nx = TRAIN_HORIZON, 7 * N_AGENTS
    rpb = nominal

    def nominal_text(leg):
        n = rpb["legs"][leg]
        hits = "/".join("yes" if hit else "no" for hit in (n["obstacle_distance"] < OBSTACLE_RADIUS).tolist())
        car_car = "yes" if n["car_distance"] < EVAL_MIN_DIST else "no"
        return (f"{errors_text(n['pos'], n['head'])}\nobstacle hit (car 1/car 2): {hits}; car-car < {EVAL_MIN_DIST:g} m: {car_car} "
                f"(closest {n['car_distance']:.2f} m)")
    save_trajectory_png([[{"x_log": rpb["x_log"][0, :T], "goal_log": nominal["data"][0, :T, nx:], "title": nominal_text(0)}]],
                        "nominal.png", f"Nominal crossing (rPB, {T * DT:.0f} s)")
    save_grid_gif([{"x_log": rpb["x_log"][0], "goal_log": nominal["data"][0, :, nx:],
                    "title": f"crossing: {nominal_text(0)}\nback: {nominal_text(1)}"}],
                  "goal_and_back_nominal.gif", f"Nominal crossing and back (rPB), goal change at {T * DT:.0f} s")
    save_dxref_figure([("nominal crossing and back", rpb["u_log"][0], [T * DT])], "dxref_nominal.png",
                      "rPB offset dxref, nominal crossing and back (red: goal change)")

    examples = gif_runs(test)
    run, data = test["run"], test["data"]
    legs = test["legs"]
    save_trajectory_png([[{"x_log": run["x_log"][s, :T], "goal_log": data[s, :T, nx:],
                           "title": f"{family} - run {s}\n{errors_text(legs[0]['pos'][s], legs[0]['head'][s])}"} for s in runs]
                         for family, runs in examples.items()],
                        "families.png", f"Test samples per family (rPB, {T * DT:.0f} s); titles: final errors")
    for family, runs in examples.items():
        save_grid_gif([{"x_log": run["x_log"][s], "goal_log": data[s, :, nx:],
                        "title": f"run {s}, goal change at {T * DT:.0f} s\nleg 1: {errors_text(legs[0]['pos'][s], legs[0]['head'][s])}\n"
                                 f"leg 2: {errors_text(legs[1]['pos'][s], legs[1]['head'][s])}"} for s in runs],
                      f"goal_and_back_{family}.gif", f"{family}, goal-and-back (rPB)")
    save_dxref_figure([(f"{family} run {runs[0]}", run["u_log"][runs[0], :T], []) for family, runs in examples.items()],
                      "dxref_families.png", "rPB offset dxref, first GIF sample of every family")
    save_dxref_figure([(f"{family} run {runs[0]}", run["u_log"][runs[0]], [T * DT]) for family, runs in examples.items()],
                      "dxref_goal_and_back.png", "rPB offset dxref, goal-and-back (red: goal change, the car's memory is reset)")

    panels = []
    for s in range(min(GIF_PANELS, chain["x_log"].shape[0])):
        legs_of = [sorted([r for r in chain["records"] if r["run"] == s and r["car"] == i], key=lambda r: r["leg"]) for i in range(N_AGENTS)]
        reached = " | ".join(f"car {i + 1}: " + "".join("✓" if math.isfinite(r["time_to_goal"]) else "✗" for r in legs_of[i])
                             for i in range(N_AGENTS))
        panels.append({"x_log": chain["x_log"][s], "goal_log": chain["goal_log"][s], "title": f"chain {s}: goals reached\n{reached}"})
    save_grid_gif(panels, "trajectory_chains.gif", "Reference chains (rPB): next goal when the car has arrived")


class Tee:
    """Writes to the terminal and keeps a copy (for summary.txt)."""
    def __init__(self, stream):
        self.stream, self.text = stream, []

    def write(self, text):
        self.stream.write(text)
        self.text.append(text)

    def flush(self):
        self.stream.flush()


if __name__ == "__main__":
    tee = Tee(sys.stdout)
    sys.stdout = tee
    trained_families = stored_args.get("families")
    print(f"[INFO] checkpoint: {TRAINED_PBR_MODEL_PATH}")
    print(f"[INFO] outputs:    {get_output_dir()}")
    print(f"[RUN]  epoch {checkpoint.get('epoch')}, validation loss {checkpoint.get('validation_loss')}, "
          f"{N_AGENTS} cars, training horizon {TRAIN_HORIZON} steps ({TRAIN_HORIZON * DT:.0f} s), "
          f"trained on families: {'all of its scenario' if trained_families is None else ', '.join(trained_families)}")

    system = make_system()
    controller = load_controller(system, TRAINED_PBR_MODEL_PATH)
    nominal = run_nominal(system, controller)
    test = run_test(system, controller)
    long = run_long_horizon(system, controller, test)
    chain = run_chain(system, controller, test["data"][:CHAIN_NUM_RUNS, :1], test["dataset"])

    print_nominal(nominal)
    print_test_results(test)
    print_goal_and_back(test)
    print_long_horizon(long)

    print()
    save_rollout_metrics(test, long)
    save_chain_metrics(chain)
    save_figures(nominal, test, chain)

    sys.stdout = tee.stream
    summary_path = output_file("summary.txt")
    with open(summary_path, "w") as f:
        f.write("".join(tee.text))
    print(f"[INFO] saved {summary_path}")
