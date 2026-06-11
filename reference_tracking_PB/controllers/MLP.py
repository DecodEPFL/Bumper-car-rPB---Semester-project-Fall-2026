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



class ZeroController(torch.nn.Module):
    def __init__(self, ref_dim):
        super().__init__()
        self.ref_dim = ref_dim

    def reset(self):
        pass

    def forward(self, x, v, xbar):
        return torch.zeros_like(xbar)