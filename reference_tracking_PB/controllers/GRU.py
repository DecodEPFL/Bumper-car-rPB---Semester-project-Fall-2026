import torch
import torch.nn as nn


class SharedGRUGain(nn.Module):
    """
    M2 per car with shared weights: one GRU cell applied to every car's own features (the same weights for all cars,
    one memory per car), output gain (dX, dY) per car bounded by gain_bound * tanh -> |gain| <= gain_bound, so
    M1 * M2 stays in l_p whatever the inputs or the memory are.

    GRUCell, not nn.GRU: the closed loop is computed step by step (the input at t depends on the output at t-1),
    so there is no known input sequence to process at once. The memory is kept by the caller (the controller).
    """
    def __init__(self, dim_in, dim_hidden=32, gain_bound=5.0):
        super().__init__()
        self.gru = nn.GRUCell(dim_in, dim_hidden)
        self.out = nn.Linear(dim_hidden, 2)
        self.dim_in, self.dim_hidden, self.gain_bound = dim_in, dim_hidden, gain_bound

    def forward(self, features, hidden):
        """
        features: [B * n_agents, dim_in], hidden: [B * n_agents, dim_hidden] (memory of the previous step)
        returns the gain [B * n_agents, 2] and the new memory [B * n_agents, dim_hidden]
        """
        hidden = self.gru(features, hidden)
        gain = self.gain_bound * torch.tanh(self.out(hidden))
        return gain, hidden
