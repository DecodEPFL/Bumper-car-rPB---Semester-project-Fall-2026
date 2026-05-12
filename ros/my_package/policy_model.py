import torch
import torch.nn as nn

## CURRENTLY NOT USED ANYWHERE IN THE NODE ##
class PolicyMLP(nn.Module):
    def __init__(self, x_scale, params, hidden=(128, 64)):
        super().__init__()
        self.register_buffer("x_scale", x_scale.detach().clone().float())  # (7,)
        self.register_buffer("u_max",   torch.tensor([params.max_speed, params.steering_range],   dtype=torch.float32))  # (2,)
        self.net = nn.Sequential(
            nn.Linear(8, hidden[0]),
            nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]),
            nn.ReLU(),
            nn.Linear(hidden[1], 2),
        )
        nn.init.uniform_(self.net[-1].weight, -1.0, 1.0)
        nn.init.uniform_(self.net[-1].bias, -1.0, 1.0)
        self.params = params

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        x_n = x / self.x_scale
        u   = self.net(x_n)

        v_ref = torch.tanh(u[..., [0]])
        delta_ref = torch.tanh(u[..., [1]])
        
        return torch.cat([v_ref, delta_ref], dim=-1)