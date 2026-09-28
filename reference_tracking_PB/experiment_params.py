import torch

from experiments.bumpercar.arg_parser import argument_parser

STD_INIT_XY = 0.2
STD_INIT_THETA = 10/360*2*torch.pi
args = argument_parser()
n_agents = args.n_agents

# Nominal (x, y, theta) of each car's start and goal, per number of cars.
# The other 4 states (v_f, beta_f, beta_r, delta) start and end at 0.
SCENARIOS = {
    2: {
        "start": [(-2.0, -2.5, 0.0), (2.0, -2.5, -torch.pi)],
        "goal": [(2.0, 3.0, torch.pi / 2), (-2.0, 3.0, torch.pi / 2)],
    },
    # TODO: placeholder 3-car scenario (car 3 drives top -> bottom through the gap); define the real one.
    3: {
        "start": [(-2.0, -2.5, 0.0), (2.0, -2.5, -torch.pi), (0.0, 4.5, -torch.pi / 2)],
        "goal": [(2.0, 3.0, torch.pi / 2), (-2.0, 3.0, torch.pi / 2), (0.0, -3.5, -torch.pi / 2)],
    },
}


def _poses_to_state(poses):
    """[(x, y, theta), ...] -> flat state [x, y, theta, 0, 0, 0, 0] per car."""
    return torch.tensor([v for (x, y, th) in poses for v in (x, y, th, 0.0, 0.0, 0.0, 0.0)])


def getCarInitParams(device):
    if n_agents not in SCENARIOS:
        raise ValueError(f"No scenario defined for n_agents={n_agents}; add one to SCENARIOS in experiment_params.py")
    x0_bumpercar = _poses_to_state(SCENARIOS[n_agents]["start"])
    x_final = _poses_to_state(SCENARIOS[n_agents]["goal"])

    obstacle_centers = [
        torch.tensor([[-2.0, 0.0]], device=device),
        torch.tensor([[2.0, 0.0]], device=device),
    ]
    obstacle_covs = [torch.tensor([[0.44, 0.44]], device=device)] * len(obstacle_centers)
    
    car_init_radius = 0.3
    std_init_theta = STD_INIT_THETA
    
    return x0_bumpercar, x_final, obstacle_centers, obstacle_covs, car_init_radius, std_init_theta

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
    Q_final = Q * 20
    Qs = torch.kron(torch.eye(n_agents), torch.eye(1)).to(device)

    position_deadzone = 0.01
    steady_state_velocity_radius = 0.1
    min_dist = 2 if args.col_av else None   # BumpercarLoss requires None together with alpha_col=None

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
