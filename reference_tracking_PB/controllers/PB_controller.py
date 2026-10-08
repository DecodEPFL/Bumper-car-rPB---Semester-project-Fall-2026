import torch
import torch.nn as nn
import numpy as np
import math
import time

from argparse import Namespace
from config import device
from .contractive_ren import ContractiveREN
from .MLP import SharedMLPGain, ZeroController
from .GRU import SharedGRUGain
from utils.assistive_functions import to_tensor


class PerfBoostController(nn.Module):
    """
    Performance boosting controller, following the paper:
        "Learning to Boost the Performance of Stable Nonlinear Systems".
    Implements a state-feedback controller with stability guarantees.
    NOTE: When used in closed-loop, the controller input is the measured state
        of the plant and the controller output is the input to the plant.
        This controller has a memory for the last input ("self.last_input") and
        the last output ("self.last_output").
    """
    def __init__(
        self, noiseless_forward, input_init: torch.Tensor, output_init: torch.Tensor,
        # acyclic REN properties
        dim_internal: int, dim_nl: int,
        initialization_std: float = 0.5,
        posdef_tol: float = 0.001, contraction_rate_lb: float = 1.0,
        ren_internal_state_init=None,
        # misc
        dt: float = 0.04,
        ref_trace_tau: float = 5.0,
        m2_type: str = "gru", m2_gain_bound: float = 5.0,
    ):
        """
         Args:
            noiseless_forward: system dynamics without process noise. can be TV.
            input_init (torch.Tensor): initial input to the controller.
            output_init (torch.Tensor): initial output from the controller before anything is calculated.
            * the following are the same as AcyclicREN args:
            dim_internal (int): Internal state (x) dimension. This state evolves with contraction properties.
            dim_nl (int): Dimension of the input ("v") and ouput ("w") of the nonlinear static block of REN.
            initialization_std (float, optional): Weight initialization. Set to 0.1 by default.
            epsilon (float, optional): Positive and negligible scalar to force positive definite matrices.
            contraction_rate_lb (float, optional): Lower bound on the contraction rate. Defaults to 1.
            ren_internal_state_init (torch.Tensor, optional): initial state of the REN. Defaults to 0 when None.
            dt (float): time step [s] of the plant, for the reference trace.
            ref_trace_tau (float): time constant [s] of the reference trace per car: whenever the car's goal changes (and
                at t = 0) it restarts at what is left to travel, new goal minus the car's current pose (x, y and heading
                as sin, 1 - cos), and then decays from that value. 0 = only the pulse itself. At a goal change the car's
                REN state and GRU memory are reset too (per-car reset): every new reference starts like a training rollout.
            m2_type (str): M2 per car with shared weights, on the car's own w_in, goal and target heading plus its
                current state (goal relative to the car, heading error, nearest other car): "gru" (SharedGRUGain, with
                memory) or "mlp" (SharedMLPGain, without memory). Output bounded by m2_gain_bound.
            m2_gain_bound (float): bound of M2's output gain per component.

        Everything is per car: M1 (REN) and M2 are the same networks for every car, each car has its own input and its
        own internal state; the cars are rows of the batch (row = sample * n_agents + car), so the network sizes do not
        depend on the number of cars.
        """
        super().__init__()


        # set initial conditions
        # self.input_init = input_init.reshape(1, -1)
        # self.output_init = output_init.reshape(1, -1)
        # self.vg_init = torch.zeros(1,4).reshape(1,-1)
        # self.xbar_init = torch.zeros(1,4).reshape(1,-1)

        self.register_buffer("input_init", input_init.reshape(1, -1))
        self.register_buffer("output_init", output_init.reshape(1, -1))
        self.register_buffer("vg_init", torch.zeros(1, output_init.shape[-1]))
        self.register_buffer("xbar_init", torch.zeros(1, output_init.shape[-1]))
        self.register_buffer("theta_ref_init", torch.zeros(1, output_init.shape[-1] // 2))

        # set dimensions
        self.dim_in = self.input_init.shape[-1]
        self.dim_out = self.output_init.shape[-1]
        self.n_agents = self.dim_in // 7
        # per-step factor of the reference trace: exp(-dt / tau), so a change c is c * exp(-t / tau) after t seconds
        # (tau = 0: factor 0, the trace is only the change at its own step)
        self.dt, self.ref_trace_tau = dt, ref_trace_tau
        self.ref_trace_decay = math.exp(-dt / ref_trace_tau) if ref_trace_tau > 0 else 0.0

        # M1: one REN for every car. Input per car: w_in (8: w_hat with (sin, 1 - cos) for the heading) and the
        # reference trace (4: dx, dy, sin, 1 - cos); output per car: energy (2)
        self.c_ren = ContractiveREN(
            dim_in=12, dim_out=2, dim_internal=dim_internal,
            dim_nl=dim_nl, initialization_std=initialization_std,
            internal_state_init=ren_internal_state_init,
            posdef_tol=posdef_tol, contraction_rate_lb=contraction_rate_lb
        ).to(device)

        # M2: one network for every car. Features per car: w_in (8), goal (2), target heading (2),
        # goal relative to the car (2), heading error (2), nearest other car relative to the car (2)
        self.m2_type = m2_type
        if m2_type == "gru":
            self.M2 = SharedGRUGain(dim_in=18, gain_bound=m2_gain_bound)
        elif m2_type == "mlp":
            self.M2 = SharedMLPGain(dim_in=18, gain_bound=m2_gain_bound)
        else:
            raise ValueError(f"unknown m2_type {m2_type}")

        # define the system dynamics without process noise
        self.noiseless_forward = noiseless_forward

        self.reset()

    def reset(self):
        """
        set time to 0 and reset to initial state.
        """
        self.t = 0  # time
        self.last_input = self.input_init.detach().clone()
        self.last_output = self.output_init.detach().clone()
        self.last_vg = self.vg_init.detach().clone()
        self.last_xbar = self.xbar_init.detach().clone()
        self.last_theta_ref = self.theta_ref_init.detach().clone()
        self.ref_trace = None   # decaying trace of the reference change, per sample (set at the first call)
        self.m2_hidden = None   # GRU M2: memory per sample and car (set to 0 at the first call)

        self.c_ren.x = self.c_ren.init_x    # reset the REN state to the initial value

    def forward(self, input_t: torch.Tensor, vg: torch.Tensor, xbar: torch.Tensor, theta_ref: torch.Tensor):
        """
        Forward pass of the controller.

        Args:
            input_t (torch.Tensor): Input with the size of (batch_size, 1, self.dim_in).
            NOTE: when used in closed-loop, "input_t" is the measured states.
            xbar (torch.Tensor): goal (x, y) per car, (batch_size, 1, 2*n_agents).
            theta_ref (torch.Tensor): target heading per car, (batch_size, 1, n_agents).

        Return:
            y_out (torch.Tensor): Output with (batch_size, 1, self.dim_out).
        """


        B, n = input_t.shape[0], self.n_agents
        car = input_t.view(B, n, 7)                     # measured state per car: x, y, theta, v_f, beta_f, beta_r, delta
        position, theta = car[..., 0:2], car[..., 2]    # [B, n, 2], [B, n]
        goal = xbar.view(B, n, 2)                       # goal position per car
        target = theta_ref.view(B, n)                   # target heading per car

        # 1) internal model, all cars in one call (the cars are dynamically decoupled: each car's prediction uses only
        #    its own state, input and goal). It uses the reference the plant used in this step (the current one), so a
        #    known goal change is not mistaken for a disturbance. At t = 0 there is no previous step (w_hat_0 = 0 below).
        x_predicted, _ = self.noiseless_forward(
            t=self.t,
            x=self.last_input,    # last input to the controller is the last state of the plant
            u=self.last_output,   # last output of the controller is the last input to the plant
            v=self.last_vg,
            xbar=xbar if self.t > 0 else self.last_xbar,
        )                                                                     # [B, 1, 7*n]

        # 2) reconstructed disturbance per car: measured state minus predicted state
        w_hat = (input_t - x_predicted).view(B, n, 7)                        # [B, n, 7]
        if self.t == 0:
            # the start state is known, not a disturbance: w_hat_0 = 0. The start reaches M1 relative to the goal
            # through the reference-trace pulse at t = 0 -> a car with nothing to do gets no energy of its own
            w_hat = torch.zeros_like(w_hat)

        # 3) w_in per car: heading entry as (sin, 1 - cos) -> continuous across +-pi, and 0 when w_hat = 0 (l_p)
        w_theta = w_hat[..., 2:3]
        w_in = torch.cat((w_hat[..., 0:2], torch.sin(w_theta), 1 - torch.cos(w_theta), w_hat[..., 3:7]), dim=-1)   # [B, n, 8]

        # 4) reference trace per car: when this car's goal changes (and at t = 0, from the zero initial reference), the
        #    way still to go: own new goal minus own pose (x, y and heading as sin, 1 - cos); 0 at all other steps.
        #    At a goal change the trace restarts from the new pulse (see 4b), otherwise it decays:
        #    trace_t = decay * trace_{t-1}: bounded pulses, stable filter -> l_p
        last_goal = self.last_xbar.view(-1, n, 2)
        last_target = self.last_theta_ref.view(-1, n)
        goal_changed = ((goal - last_goal).abs().sum(-1) + (1 - torch.cos(target - last_target))) > 0   # [B, n]
        to_turn = target - theta                                                          # [B, n]
        pulse = torch.cat((goal - position, torch.sin(to_turn).unsqueeze(-1), (1 - torch.cos(to_turn)).unsqueeze(-1)),
                          dim=-1) * goal_changed.unsqueeze(-1)                            # [B, n, 4]
        if self.ref_trace is None:
            self.ref_trace = pulse
        else:
            self.ref_trace = torch.where(goal_changed.unsqueeze(-1), pulse, self.ref_trace_decay * self.ref_trace)

        # 4b) per-car reset at a goal change (t > 0): the car starts over as at t = 0 of a training rollout (one
        #     reference per rollout) -> its REN state and GRU memory back to their initial values, its trace = only the
        #     new pulse (above); the other cars keep theirs. Never active in training (the goal only changes at t = 0)
        if self.t > 0:
            restart = goal_changed.view(B * n)                                           # rows: sample * n + car
            self.c_ren.x = torch.where(restart.view(B * n, 1, 1), self.c_ren.init_x, self.c_ren.x)
            if self.m2_hidden is not None:
                self.m2_hidden = torch.where(restart.view(B * n, 1), torch.zeros_like(self.m2_hidden), self.m2_hidden)

        # 5) M1: the same REN for every car, own input, own state (rows: sample * n + car).
        #    theta* is not given to the REN: it does not vanish, so it would break l_p
        ren_input = torch.cat((w_in, self.ref_trace), dim=-1)                             # [B, n, 12]
        m1 = self.c_ren.forward(ren_input.view(B * n, 1, 12)).view(B, n, 2)              # energy per car

        # 6) M2: the same network for every car, own features, own memory (GRU). Features: own w_in, goal and target
        #    heading, plus the current state: goal relative to the car, heading error, nearest other car
        to_goal = goal - position                                                         # [B, n, 2]
        heading_error = target - theta                                                    # [B, n]
        if n > 1:
            relative = position.unsqueeze(1) - position.unsqueeze(2)                     # [B, n, n, 2]  relative[:, i, j] = p_j - p_i
            distance = torch.linalg.norm(relative, dim=-1) + 1e6 * torch.eye(n, device=position.device)   # not the car itself
            nearest = distance.argmin(dim=-1)                                            # [B, n]
            to_nearest = relative.gather(2, nearest[..., None, None].expand(-1, -1, 1, 2)).squeeze(2)   # [B, n, 2]
        else:
            to_nearest = torch.zeros_like(position)                                      # no other car
        features = torch.cat((
            w_in,
            goal, torch.sin(target).unsqueeze(-1), (1 - torch.cos(target)).unsqueeze(-1),
            to_goal, torch.sin(heading_error).unsqueeze(-1), (1 - torch.cos(heading_error)).unsqueeze(-1),
            to_nearest,
        ), dim=-1)                                                                        # [B, n, 18]
        if self.m2_hidden is None and self.m2_type == "gru":                             # first step: GRU memory 0
            self.m2_hidden = torch.zeros(B * n, self.M2.dim_hidden, device=features.device, dtype=features.dtype)
        gain, self.m2_hidden = self.M2.forward(features.view(B * n, 18), self.m2_hidden)
        gain = gain.view(B, n, 2)                                                         # bounded gain per car

        # 7) offset per car, back to the plant's layout [B, 1, 2*n]
        dxref = m1 * gain                                                                 # [B, n, 2]
        output = dxref.view(B, 1, 2 * n)
        #output = torch.clamp(output,min = -10,max = 10)
        # update internal states
        self.last_input, self.last_output, self.last_xbar = input_t, output, xbar
        self.last_theta_ref = theta_ref
        self.last_vg = vg
        self.t += 1
        return output

    # setters and getters
    def get_parameter_shapes(self):
        return self.c_ren.get_parameter_shapes()

    def get_named_parameters(self):
        return self.c_ren.get_named_parameters()

    def get_parameters_as_vector(self):
        # TODO: implement without numpy
        return np.concatenate([p.detach().clone().cpu().numpy().flatten() for p in self.c_ren.parameters()])
    
    def get_m2_parameters(self):
        # copies: state_dict() alone shares memory with the live weights, which the optimizer keeps changing
        return {name: value.detach().clone() for name, value in self.M2.state_dict().items()}

    def set_m2_parameters(self, params):
        self.M2.load_state_dict(params)

    def set_parameter(self, name, value):
        current_val = getattr(self.c_ren, name)
        value = torch.nn.Parameter(to_tensor(value.reshape(current_val.shape)))
        setattr(self.c_ren, name, value)
        self.c_ren._update_model_param()    # update dependent params

    def set_parameters(self, param_dict):
        for name, value in param_dict.items():
            self.set_parameter(name, value)

    def set_parameters_as_vector(self, value):
        idx = 0
        for name, shape in self.get_parameter_shapes().items():
            if len(shape) == 1:
                dim = shape
            elif len(shape) == 2:
                dim = shape[0]*shape[1]
            else:
                raise NotImplementedError
            idx_next = idx + dim
            # select indx
            if len(value.shape) == 1:
                value_tmp = value[idx:idx_next]
            elif len(value.shape) == 2:
                value_tmp = value[:, idx:idx_next]
            else:
                raise AssertionError
            # set
            with torch.no_grad():
                self.set_parameter(name, value_tmp.reshape(shape))
            idx = idx_next
        assert idx_next == value.shape[-1]

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
