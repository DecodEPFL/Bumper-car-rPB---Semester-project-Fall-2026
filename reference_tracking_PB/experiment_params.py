import torch

from experiments.bumpercar.arg_parser import argument_parser

args = argument_parser()
n_agents = args.n_agents

# Nominal (x, y, theta) of each car's start and goal, per number of cars.
# The other 4 states (v_f, beta_f, beta_r, delta) start and end at 0.
SCENARIOS = {
    2: {
        "start": [(-2.0, -2.5, 0.0), (2.0, -2.5, -torch.pi)],
        # one goal: one (x, y, theta) per car. Or a list of goals visited in order, each a list with one pose per car;
        # every goal is a pose (position and heading at the end of its leg), the horizon is split into equal legs.
        # Goal-and-back:
        "goal": [[(2.0, 4.5, torch.pi / 2), (-2.0, 4.5, torch.pi / 2)],        # goal 1: car 1, car 2
                 [(-2.0, -2.5, 0.0), (2.0, -2.5, -torch.pi)]],                # goal 2: back to the start poses
        # single goal: "goal": [(2.0, 4.5, torch.pi / 2), (-2.0, 4.5, torch.pi / 2)],
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


def _goal_list(scenario):
    """Goals of a scenario in visiting order, each a list with one (x, y, theta) per car.
    "goal" is either one goal (one pose per car) or a list of goals (each a list with one pose per car)."""
    goal = scenario["goal"]
    single_goal = isinstance(goal[0][0], (int, float))       # [(x, y, theta), ...]: one pose per car
    return [goal] if single_goal else list(goal)


def getCarInitParams(device):
    if n_agents not in SCENARIOS:
        raise ValueError(f"No scenario defined for n_agents={n_agents}; add one to SCENARIOS in experiment_params.py")
    x0_bumpercar = _poses_to_state(SCENARIOS[n_agents]["start"])
    x_final = _poses_to_state(_goal_list(SCENARIOS[n_agents])[-1])    # the last goal

    obstacle_centers = [torch.tensor([center], device=device) for center in OBSTACLES["centers"]]
    obstacle_covs = [torch.tensor([OBSTACLES["cov"]], device=device)] * len(obstacle_centers)
    
    car_init_radius = args.start_radius                              # [m] start position spread
    start_heading_range = args.start_heading_range * torch.pi / 180   # [rad] start heading spread, +-

    return x0_bumpercar, x_final, obstacle_centers, obstacle_covs, car_init_radius, start_heading_range

def getGoalSequence():
    """Goals of the scenario in visiting order: one flat state [x, y, theta, 0, 0, 0, 0] per car for each goal
    (a single goal: a list of one)."""
    return [_poses_to_state(goal) for goal in _goal_list(SCENARIOS[n_agents])]


def getCarFinalParams():
    x_final_limit = (-2.5, 2.5) 
    y_final_limit = (2, 5)
    final_car_min_dist = 2
    
    return x_final_limit, y_final_limit, final_car_min_dist

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
        heading_final_steps=args.heading_final_steps,
        alpha_offset_final=args.alpha_offset_final,
        obst_safe_dist=args.obst_safe_dist,
    )
