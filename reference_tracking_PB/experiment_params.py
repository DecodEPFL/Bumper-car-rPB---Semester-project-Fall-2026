import torch

from experiments.bumpercar.arg_parser import argument_parser

args = argument_parser()
n_agents = args.n_agents

# Nominal (x, y, theta) of each car's start and goal, per number of cars.
# The other 4 states (v_f, beta_f, beta_r, delta) start and end at 0.
SCENARIOS = {
    2: {
        # named poses (x, y, theta), shared by all cars: bottom/top, left/right
        "poses": {"BL": (-2.0, -2.5, 0.0), "BR": (2.0, -2.5, -torch.pi),
                  "TL": (-2.0, 4.5, torch.pi / 2), "TR": (2.0, 4.5, torch.pi / 2)},
        # missions: per car (start, goal of leg 1, goal of leg 2) as pose names; the horizon is split into equal legs.
        # All missions are drawn equally often. The first mission is the nominal scenario (plots, report figures).
        "missions": [
            [("BL", "TR", "BL"), ("BR", "TL", "BR")],   # crossing goal-and-back (nominal)
        ],
        # single goal: crossing to the top, facing up
        # "missions": [[("BL", "TR"), ("BR", "TL")]],
        # "missions": [
        #     [("BL", "TR", "BL"), ("BR", "TL", "BR")],   # crossing goal-and-back (nominal)
        #     [("BL", "TL", "BL"), ("BR", "TR", "BR")],   # parallel goal-and-back (each car stays on its side)
        #     [("BL", "TR", "BL"), ("BR", "BR", "BR")],   # one car drives goal-and-back, the other is parked
        #     [("TR", "BL", "BL"), ("TL", "BR", "BR")],   # return only (start at the top)
        #     [("TR", "BL", "BL"), ("TL", "TL", "TL")],   # one car returns alone, the other is parked at the top
        #     [("BL", "BL", "TR"), ("BR", "TL", "TL")],   # staggered: one car waits a leg, then drives up
        # ],
        # single scenario instead of missions: "start": one pose per car; "goal": one pose per car, or a list of goals
        # visited in order (each a list with one pose per car), e.g.
        # "start": [(-2.0, -2.5, 0.0), (2.0, -2.5, -torch.pi)],
        # "goal": [[(2.0, 4.5, torch.pi / 2), (-2.0, 4.5, torch.pi / 2)], [(-2.0, -2.5, 0.0), (2.0, -2.5, -torch.pi)]],
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


def getMissions():
    """Named poses {name: (x, y, theta)} and the missions of the scenario: each mission is a list with one
    (start, goal of leg 1, goal of leg 2, ...) of pose names per car, as listed in the scenario.
    A scenario without "missions" (only "start" and "goal") is one mission."""
    scenario = SCENARIOS[n_agents]
    if "missions" not in scenario:
        goals = _goal_list(scenario)
        poses = {f"start{i}": pose for i, pose in enumerate(scenario["start"])}
        poses.update({f"goal{k}_{i}": pose for k, goal in enumerate(goals) for i, pose in enumerate(goal)})
        mission = [tuple([f"start{i}"] + [f"goal{k}_{i}" for k in range(len(goals))]) for i in range(n_agents)]
        return poses, [mission]
    missions = [list(mission) for mission in scenario["missions"]]
    for mission in missions:
        assert len(mission) == n_agents and len({len(car) for car in mission}) == 1, "one (start, goals...) per car, same number of legs"
    return scenario["poses"], missions


def _nominal_start_and_goals():
    """Nominal start poses and goals (one list of poses per leg) of the scenario: its first mission."""
    poses, missions = getMissions()
    mission = missions[0]
    start = [poses[car[0]] for car in mission]
    goals = [[poses[car[leg]] for car in mission] for leg in range(1, len(mission[0]))]
    return start, goals


def getCarInitParams(device):
    if n_agents not in SCENARIOS:
        raise ValueError(f"No scenario defined for n_agents={n_agents}; add one to SCENARIOS in experiment_params.py")
    start, goals = _nominal_start_and_goals()
    x0_bumpercar = _poses_to_state(start)
    x_final = _poses_to_state(goals[-1])                                # the last goal

    obstacle_centers = [torch.tensor([center], device=device) for center in OBSTACLES["centers"]]
    obstacle_covs = [torch.tensor([OBSTACLES["cov"]], device=device)] * len(obstacle_centers)
    
    car_init_radius = args.start_radius                              # [m] start position spread
    start_heading_range = args.start_heading_range * torch.pi / 180   # [rad] start heading spread, +-

    return x0_bumpercar, x_final, obstacle_centers, obstacle_covs, car_init_radius, start_heading_range

def getGoalSequence():
    """Nominal goals (first mission) in visiting order: one flat state [x, y, theta, 0, 0, 0, 0] per car for each goal
    (a single goal: a list of one)."""
    _, goals = _nominal_start_and_goals()
    return [_poses_to_state(goal) for goal in goals]


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
        goal_window_steps=args.goal_window_steps,
        alpha_offset_final=args.alpha_offset_final,
        obst_safe_dist=args.obst_safe_dist,
        alpha_bounds=args.alpha_bounds,
        alpha_corridor=args.alpha_corridor,
        corridor_radius=args.corridor_radius,
        alpha_smooth=args.alpha_smooth,
    )
