import torch

try:
    from experiments.robots.arg_parser import argument_parser
except ImportError:
    argument_parser = None

STD_INIT_XY = 1.0
STD_INIT_THETA = 10 / 360 * 2 * torch.pi
n_agents = 2

if argument_parser is None:
    class _Args:
        alpha_col = 0.0
        alpha_obst = 0.0
        alpha_u = 0.0

    args = _Args()
else:
    args = argument_parser()


def getCarInitParams(device):
    # Aligned with reference_tracking_PB/experiment_params.py
    x0_bumpercar = torch.tensor([
            -2.0, -2.5, 0.0, 0.0, 0.0, 0.0, 0.0,         # car 1: pos (-2, -2.5), theta 0
            2.0, -2.5, -torch.pi, 0.0, 0.0, 0.0, 0.0,    # car 2: pos (2, -2.5), theta -pi
        ], device=device)

    x_final = torch.tensor([
            2.0, 3.0, torch.pi/2, 0.0, 0.0, 0.0, 0.0,    # car 1: goal (2, 3)
            -2.0, 3.0, torch.pi/2, 0.0, 0.0, 0.0, 0.0,   # car 2: goal (-2, 3)
        ], device=device)

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
    Q = 10 * torch.kron(torch.eye(n_agents), torch.eye(2)).to(device)
    Qs = torch.kron(torch.eye(n_agents), torch.eye(1)).to(device)

    position_deadzone = 0.01
    steady_state_velocity_radius = 0.15
    min_dist = 1

    return (
        Q,
        Qs,
        args.alpha_col,
        args.alpha_obst,
        args.alpha_u,
        position_deadzone,
        steady_state_velocity_radius,
        min_dist
    )
