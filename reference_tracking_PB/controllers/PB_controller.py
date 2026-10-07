import torch
import torch.nn as nn
import numpy as np
import math
import time

from argparse import Namespace
from config import device
from .contractive_ren import ContractiveREN
from .MLP import MLP, ZeroController
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
        output_amplification: float=20,
        dt: float = 0.04,
        ref_trace_tau: float = None,
        m2_type: str = "mlp", m2_gain_bound: float = 5.0,
    ):
        """
         Args:
            noiseless_forward: system dynamics without process noise. can be TV.
            input_init (torch.Tensor): initial input to the controller.
            output_init (torch.Tensor): initial output from the controller before anything is calculated.
            output_amplification (float): TODO
            * the following are the same as AcyclicREN args:
            dim_internal (int): Internal state (x) dimension. This state evolves with contraction properties.
            dim_nl (int): Dimension of the input ("v") and ouput ("w") of the nonlinear static block of REN.
            initialization_std (float, optional): Weight initialization. Set to 0.1 by default.
            epsilon (float, optional): Positive and negligible scalar to force positive definite matrices.
            contraction_rate_lb (float, optional): Lower bound on the contraction rate. Defaults to 1.
            ren_internal_state_init (torch.Tensor, optional): initial state of the REN. Defaults to 0 when None.
            dt (float): time step [s] of the plant, for the reference trace.
            ref_trace_tau (float): if set, the REN also gets a trace per car [s]: whenever the car's goal changes (and at
                t = 0) it jumps by what is left to travel, new goal minus the car's current pose (x, y and heading as
                sin, 1 - cos), and then decays from that value with time constant ref_trace_tau.
                0 = only the pulse itself (non-zero only at the step of the change). None = off.
            m2_type (str): "mlp": M2 is one MLP for all cars on [w_in, goals, target headings]. "gru": M2 per car with
                shared weights (SharedGRUGain), on the car's own w_in, goal and target heading plus its current state
                (goal relative to the car, heading error, nearest other car), output bounded by m2_gain_bound.
            m2_gain_bound (float): bound of the GRU M2's output gain per component (m2_type "gru").
        """
        super().__init__()

        self.output_amplification = output_amplification

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
        # size of the networks' input w_in: 8 per car (w_hat with (sin theta, 1 - cos theta) instead of theta)
        self.dim_w_in = 8 * self.n_agents
        # REN input: w_in, plus the reference trace (4 per car: dx, dy, sin, 1 - cos) if enabled
        self.dt, self.ref_trace_tau = dt, ref_trace_tau
        self.dim_ren_in = self.dim_w_in + (4 * self.n_agents if ref_trace_tau is not None else 0)
        # per-step factor of the reference trace: exp(-dt / tau), so a change c is c * exp(-t / tau) after t seconds
        # (tau = 0: factor 0, the trace is only the change at its own step)
        self.ref_trace_decay = math.exp(-dt / ref_trace_tau) if ref_trace_tau else 0.0

        # define the REN
        self.c_ren = ContractiveREN(
            dim_in=self.dim_ren_in, dim_out=self.dim_out, dim_internal=dim_internal,
            dim_nl=dim_nl, initialization_std=initialization_std,
            internal_state_init=ren_internal_state_init,
            posdef_tol=posdef_tol, contraction_rate_lb=contraction_rate_lb
        ).to(device)

        # M2 (older checkpoints store it as "MLP.*": the loaders map those keys to "M2.*")
        self.m2_type = m2_type
        if m2_type == "mlp":
            # one MLP for all cars, input: [w_in, goal (x, y) per car, target heading (sin, 1 - cos) per car]
            self.M2 = MLP(dim_in=self.dim_w_in + self.dim_out + 2 * self.n_agents, dim_out=self.dim_out)
        elif m2_type == "gru":
            # per car, shared weights; features per car: w_in (8), goal (2), target heading (2),
            # goal relative to the car (2), heading error (2), nearest other car relative to the car (2)
            self.M2 = SharedGRUGain(dim_in=18, gain_bound=m2_gain_bound)
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


        batch_size = input_t.shape[0]

        # 1) internal model: predicted state from the previous state and input, with the reference the plant used
        #    in this step (the current one) -> a known reference change is not mistaken for a disturbance.
        #    At t = 0 there is no previous step: w_hat_0 is set to 0 in 2).
        x_predicted, _ = self.noiseless_forward(
            t=self.t,
            x=self.last_input,    # last input to the controller is the last state of the plant
            u=self.last_output,   # last output of the controller is the last input to the plant
            v=self.last_vg,
            xbar=xbar if self.t > 0 else self.last_xbar,
        )                                                                     # [B, 1, 7*n_agents]

        # 2) reconstructed disturbance: measured state minus predicted state
        w_hat = input_t - x_predicted                                         # [B, 1, 7*n_agents]
        if self.t == 0:
            # the start state is known, not a disturbance: w_hat_0 = 0. The start reaches M1 relative to the goal
            # through the reference-trace pulse at t = 0 -> a car with nothing to do gets no energy of its own
            w_hat = torch.zeros_like(w_hat)

        # 3) controller input: per car, replace the heading entry theta by (sin theta, 1 - cos theta)
        #    -> continuous across +-pi, and still 0 when w_hat = 0 (keeps the l_p property)
        w_hat_per_car = w_hat.view(batch_size, 1, self.n_agents, 7)          # [x, y, theta, v_f, beta_f, beta_r, delta]
        position = w_hat_per_car[..., 0:2]
        theta = w_hat_per_car[..., 2:3]
        rest = w_hat_per_car[..., 3:7]                                        # v_f, beta_f, beta_r, delta
        w_in = torch.cat((position, torch.sin(theta), 1 - torch.cos(theta), rest), dim=-1)
        w_in = w_in.view(batch_size, 1, 8 * self.n_agents)

        # 4) M1 (REN) input: w_in, plus the reference trace, an exogenous l_p signal that lets M1 act after w_hat has vanished
        delta_xbar = xbar - self.last_xbar                                    # [B, 1, 2*n_agents]  reference change
        delta_theta = theta_ref - self.last_theta_ref                         # [B, 1, n_agents]
        ren_parts = [w_in]
        if self.ref_trace_tau is not None:
            # pulse when a car's goal changes (and at t = 0, from the zero initial reference): what is left to travel,
            # new goal minus the car's current pose; 0 at all other steps
            goal_changed = (delta_xbar.view(batch_size, 1, self.n_agents, 2).abs().sum(-1) + (1 - torch.cos(delta_theta))) > 0   # [B, 1, n_agents]
            car_pose = input_t.view(batch_size, 1, self.n_agents, 7)          # measured state per car
            pulse_xy = (xbar.view(batch_size, 1, self.n_agents, 2) - car_pose[..., 0:2]) * goal_changed.unsqueeze(-1)   # [B, 1, n_agents, 2]
            pulse_theta = (theta_ref - car_pose[..., 2]) * goal_changed      # [B, 1, n_agents]
            pulse = torch.cat((pulse_xy.reshape(batch_size, 1, 2 * self.n_agents),
                               torch.sin(pulse_theta), 1 - torch.cos(pulse_theta)), dim=2)   # [B, 1, 4*n_agents]
            # trace: jumps by the pulse, then decays from that value: trace_t = decay * trace_{t-1} + pulse_t
            # -> l_p (finitely many bounded pulses, stable filter)
            self.ref_trace = pulse if self.ref_trace is None else self.ref_trace_decay * self.ref_trace + pulse
            ren_parts.append(self.ref_trace)
        ren_input = torch.cat(ren_parts, dim=2)                               # [B, 1, dim_ren_in]

        # 5) M1 (REN) on ren_input. theta* is not given to the REN: it does not vanish, so it would break l_p
        output_REN = self.c_ren.forward(ren_input)

        # 6) M2
        if self.m2_type == "mlp":
            # one MLP for all cars on w_in, the goal (x, y) and the target heading as (sin theta*, 1 - cos theta*) per car
            mlp_input = torch.cat((w_in, xbar, torch.sin(theta_ref), 1 - torch.cos(theta_ref)), dim=2).view(batch_size, 1, -1)
            output_M2 = self.M2.forward(mlp_input)                            # [B, 1, 2*n_agents]
        else:
            # per car (shared weights): own w_in, goal and target heading (as the MLP), plus the current state:
            # goal relative to the car, heading error, nearest other car relative to the car
            n = self.n_agents
            car = input_t.view(batch_size, n, 7)                              # measured state per car
            position, theta = car[..., 0:2], car[..., 2]                      # [B, n, 2], [B, n]
            goal, target = xbar.view(batch_size, n, 2), theta_ref.view(batch_size, n)
            to_goal = goal - position                                         # [B, n, 2]
            heading_error = target - theta                                    # [B, n]
            if n > 1:
                relative = position.unsqueeze(1) - position.unsqueeze(2)      # [B, n, n, 2]  relative[:, i, j] = p_j - p_i
                distance = torch.linalg.norm(relative, dim=-1) + 1e6 * torch.eye(n, device=position.device)   # not the car itself
                nearest = distance.argmin(dim=-1)                             # [B, n]
                to_nearest = relative.gather(2, nearest[..., None, None].expand(-1, -1, 1, 2)).squeeze(2)   # [B, n, 2]
            else:
                to_nearest = torch.zeros_like(position)                       # no other car
            features = torch.cat((
                w_in.view(batch_size, n, 8),
                goal, torch.sin(target).unsqueeze(-1), (1 - torch.cos(target)).unsqueeze(-1),
                to_goal, torch.sin(heading_error).unsqueeze(-1), (1 - torch.cos(heading_error)).unsqueeze(-1),
                to_nearest,
            ), dim=-1)                                                        # [B, n, 18]
            if self.m2_hidden is None:                                        # first step of the rollout: memory 0
                self.m2_hidden = torch.zeros(batch_size * n, self.M2.dim_hidden, device=features.device, dtype=features.dtype)
            gain, self.m2_hidden = self.M2.forward(features.view(batch_size * n, -1), self.m2_hidden)
            output_M2 = gain.view(batch_size, 1, 2 * n)                       # (dX, dY) gain per car

        output = output_REN*output_M2*self.output_amplification   # shape = (self.batch_size, 1, self.dim_out)
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
