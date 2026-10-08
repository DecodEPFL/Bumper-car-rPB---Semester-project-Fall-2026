import torch

from experiments.bumpercar.arg_parser import argument_parser

args = argument_parser()
n_agents = args.n_agents

# Scenario per number of cars: one reference (goal pose per car) per rollout.
# Training data: every sample draws a family, then one of its variants (both uniformly); a variant gives every car a
# (start zone, goal zone). Start and goal positions are uniform in their zones, start and goal headings uniform over
# the feasible directions (only if the approach point behind the pose is free, see "sampling": a car starts where it
# last arrived, so starts follow the goal rules). Cars start at rest.
# "start" / "goal": nominal (x, y, theta) per car, only for the plots and the fixed sim.py figures.
SCENARIOS = {
    2: {
        # zones for car centres: (x_min, x_max, y_min, y_max). Arena x in [-3.5, 3.5], y in [-4, 6]; centres 1 m (car
        # radius) from the walls; left/right leave a 1 m strip at the gap (|x| < 0.5) -> a crossing always changes side;
        # bottom ends at y = -1.7 and top starts at y = 2 -> every zone point is >= 1.7 m from the obstacle centres
        "zones": {
            "top": (-2.5, 2.5, 2.0, 5.0), "bottom": (-2.5, 2.5, -3.0, -1.7),
            "top_left": (-2.5, -0.5, 2.0, 5.0), "top_right": (0.5, 2.5, 2.0, 5.0),
            "bottom_left": (-2.5, -0.5, -3.0, -1.7), "bottom_right": (0.5, 2.5, -3.0, -1.7),
        },
        # families: list of variants, each one (start zone, goal zone) per car. The cars share all networks and see
        # each other only relatively, so one car order per variant is enough
        "families": {
            "crossing_up": [[("bottom_left", "top_right"), ("bottom_right", "top_left")]],
            "crossing_down": [[("top_left", "bottom_right"), ("top_right", "bottom_left")]],
            "head_on": [[("bottom_left", "top_left"), ("top_right", "bottom_right")],          # own side, opposite directions:
                        [("bottom_right", "top_right"), ("top_left", "bottom_left")]],         # up on the left / on the right
            "same_side": [[("bottom_left", "top_left"), ("bottom_right", "top_right")],         # both up
                          [("top_left", "bottom_left"), ("top_right", "bottom_right")]],        # both down
            "local": [[("top_left", "top_right"), ("bottom_right", "bottom_left")],              # side change within the
                      [("top_right", "top_left"), ("bottom_left", "bottom_right")]],             # own zone, no gap
        },
        # sampling rules: allowed area for car centres (x_min, x_max, y_min, y_max) and minimum distance to the obstacle
        # centres; start and goal heading only if the approach point approach_length behind the pose is in the allowed area;
        # starts, goals, and every goal vs every other car's start at least car_spacing apart; every goal at least
        # min_move from its own start (2 m: from rest the base P-controller only starts the car for goals >= ~2 m away,
        # its drive command is below the model's start-up threshold otherwise)
        "sampling": {"allowed": (-2.5, 2.5, -3.0, 5.0), "obstacle_clearance": 1.7, "approach_length": 1.0,
                     "car_spacing": 2.2, "min_move": 2.0},
        # nominal crossing up (plots)
        "start": [(-2.0, -2.5, 0.0), (2.0, -2.5, -torch.pi)],
        "goal": [(2.0, 4.5, torch.pi / 2), (-2.0, 4.5, torch.pi / 2)],
    },
    # TODO: placeholder 3-car scenario (car 3 drives top -> bottom through the gap); define the real one.
    3: {
        "start": [(-2.0, -2.5, 0.0), (2.0, -2.5, -torch.pi), (0.0, 4.5, -torch.pi / 2)],
        "goal": [(2.0, 4.5, torch.pi / 2), (-2.0, 4.5, torch.pi / 2), (0.0, -3.5, -torch.pi / 2)],
    },
}

# Gaussian obstacles: centers (x, y) and diagonal covariance, shared by all scenarios
OBSTACLES = {"centers": [(-2.0, 0.0), (2.0, 0.0)], "cov": (0.44, 0.44)}


def _poses_to_state(poses):
    """[(x, y, theta), ...] -> flat state [x, y, theta, 0, 0, 0, 0] per car."""
    return torch.tensor([v for (x, y, th) in poses for v in (x, y, th, 0.0, 0.0, 0.0, 0.0)])


def getFamilies():
    """Sampling zones, the selected families (--families, default all of the scenario) and the sampling rules
    (with the obstacle centres) of the scenario."""
    scenario = SCENARIOS[n_agents]
    if "families" not in scenario:
        raise ValueError(f"No families defined for n_agents={n_agents}; add zones and families to SCENARIOS")
    names = args.families if args.families is not None else list(scenario["families"])
    unknown = [name for name in names if name not in scenario["families"]]
    assert not unknown, f"unknown families {unknown}; the scenario has {list(scenario['families'])}"
    families = {name: scenario["families"][name] for name in names}
    for variants in families.values():
        assert all(len(variant) == n_agents for variant in variants), "one (start zone, goal zone) per car"
    sampling = {**scenario["sampling"], "obstacle_centers": OBSTACLES["centers"]}
    return scenario["zones"], families, sampling


def getCarInitParams(device):
    """Nominal start and goal state of the scenario (plots and fixed sim.py figures) and the obstacles."""
    if n_agents not in SCENARIOS:
        raise ValueError(f"No scenario defined for n_agents={n_agents}; add one to SCENARIOS in experiment_params.py")
    scenario = SCENARIOS[n_agents]
    x0_bumpercar = _poses_to_state(scenario["start"])
    x_final = _poses_to_state(scenario["goal"])

    obstacle_centers = [torch.tensor([center], device=device) for center in OBSTACLES["centers"]]
    obstacle_covs = [torch.tensor([OBSTACLES["cov"]], device=device)] * len(obstacle_centers)

    return x0_bumpercar, x_final, obstacle_centers, obstacle_covs

def getLossParams(device):
    Q_agent = torch.diag(torch.tensor([
        2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ], device=device)) / 15
    
    Q = torch.kron(torch.eye(n_agents, device=device), Q_agent)
    Q_final = Q * args.q_final_factor
    Qs = torch.kron(torch.eye(n_agents), torch.eye(1)).to(device)

    position_deadzone = 0.01
    steady_state_velocity_radius = 0.1
    min_dist = args.min_dist   # None under --no-col-av; BumpercarLoss requires None together with alpha_col=None

    return (
        Q,
        Q_final,
        Qs,
        args.alpha_col,
        args.alpha_obst,
        args.alpha_u,
        position_deadzone,
        steady_state_velocity_radius,
        min_dist
    )


def getExtraLossParams():
    """Heading, terminal offset and obstacle cutoff settings for BumpercarLoss (defaults = off)."""
    return dict(
        alpha_heading=args.alpha_heading,
        alpha_heading_final=args.alpha_heading_final,
        heading_radius=args.heading_radius,
        goal_window_steps=args.goal_window_steps,
        alpha_offset_final=args.alpha_offset_final,
        obst_safe_dist=args.obst_safe_dist,
        alpha_bounds=args.alpha_bounds,
        alpha_smooth=args.alpha_smooth,
    )
