import torch
import numpy as np
import torch.nn as nn

class PolicyGRU(nn.Module):
    def __init__(self, params, hidden=(128, 64), num_layers=1, dropout=0.0):
        super().__init__()

        x_scale = torch.tensor([3., 3.,np.pi, 1., 1., 1., 1.])
        self.register_buffer("x_scale", x_scale.detach().clone().float())  # expected shape: (7,)
        self.register_buffer(
            "u_max",
            torch.tensor(
                [params.max_speed, params.steering_range],
                dtype=torch.float32,
            ),
        ) 

        gru_hidden = hidden[0]
        head_hidden = hidden[1]

        self.gru = nn.GRU(
            input_size=8,              
            hidden_size=gru_hidden,
            num_layers=num_layers,
            batch_first=True,        
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.Linear(gru_hidden, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, 2),
        )

        nn.init.uniform_(self.head[-1].weight, -1.0, 1.0)
        nn.init.uniform_(self.head[-1].bias, -1.0, 1.0)

        self.params = params

    def forward(self, x, h0=None, return_hidden=True):
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32)
        
        unbatched = False

        if x.dim() == 2:
            x = x.unsqueeze(0)   
            unbatched = True

        elif x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0) 
            unbatched = True

        if x.size(-1) != 8:
            raise ValueError(f"Expected input feature size 8, got {x.size(-1)}")
        
        x = x.clone()
        x[..., :7] = x[..., :7] / self.x_scale.clamp_min(1e-6)

        y, h_n = self.gru(x, h0)
        y_last = y[:, -1, :]             

        u = self.head(y_last)            
        u = torch.tanh(u) * self.u_max   

        if unbatched:
            u = u.squeeze(0)

        if return_hidden:
            return u, h_n
        return u
