import torch
import torch.nn as nn
import numpy as np

# from config import device
from utils.assistive_functions import to_tensor


class MLP(nn.Module):
    #dim_in = 7 states + 2 ref states (xy)
    #dim_out = v_ref, delta_ref for 2 cars
    def __init__(self, dim_in = 18,dim_out = 4):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(dim_in, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, dim_out),
)


    def forward(self, x):
        return  self.mlp(x)


class SharedMLPGain(nn.Module):
    """
    M2 per car with shared weights, without memory: the MLP above on each car's own features (the same weights for all
    cars), output gain (dX, dY) per car bounded by gain_bound * tanh -> |gain| <= gain_bound, so M1 * M2 stays in l_p.
    Same inputs and bound as SharedGRUGain; the only difference is the GRU's memory.
    """
    def __init__(self, dim_in, gain_bound=5.0):
        super().__init__()
        self.mlp = MLP(dim_in=dim_in, dim_out=2)
        self.dim_in, self.gain_bound = dim_in, gain_bound

    def forward(self, features, hidden=None):
        """
        features: [B * n_agents, dim_in] -> gain [B * n_agents, 2]; hidden is passed through unchanged (no memory),
        so the controller calls both M2 types the same way
        """
        gain = self.gain_bound * torch.tanh(self.mlp(features))
        return gain, hidden



class ZeroController(torch.nn.Module):
    def __init__(self, ref_dim):
        super().__init__()
        self.ref_dim = ref_dim

    def reset(self):
        pass

    def forward(self, x, v, xbar, theta_ref=None):
        return torch.zeros_like(xbar)