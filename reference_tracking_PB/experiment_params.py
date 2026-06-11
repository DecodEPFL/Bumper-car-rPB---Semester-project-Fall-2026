import torch

from experiments.robots.arg_parser import argument_parser

STD_INIT_XY = 0.2
STD_INIT_THETA = 10/360*2*torch.pi
args = argument_parser()
n_agents = 2

def getCarInitParams(device):
    x0_bumpercar = torch.tensor([
            -2.0, -2.5, 0, 0.0, 0.0, 0.0, 0.0,   # car 1
            2.0, -2.5, -torch.pi, 0.0, 0.0, 0.0, 0.0,   # car 2
        ])
    
    x_final = torch.tensor([
            2.0, 3.0, torch.pi/2, 0.0, 0.0, 0.0, 0.0,   # car 1
            -2.0, 3.0, torch.pi/2, 0.0, 0.0, 0.0, 0.0,   # car 2
        ])
    
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
    ], device=device)) / 5
    
    Q = torch.kron(torch.eye(n_agents, device=device), Q_agent)
    Q_final = Q * 5
    Qs = torch.kron(torch.eye(n_agents), torch.eye(1)).to(device)

    position_deadzone = 0.01
    steady_state_velocity_radius = 0.1
    min_dist = 2

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
