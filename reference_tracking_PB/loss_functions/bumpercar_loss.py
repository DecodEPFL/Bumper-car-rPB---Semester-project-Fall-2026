import torch
from . import LQLossFH
from config import device


class BumpercarLoss(LQLossFH):
    def __init__(
        self, xbar, Q, Qs, Q_final=None, alpha_u=1,
        alpha_col=None, alpha_obst=None,
        loss_bound=None, sat_bound=None,
        n_agents=2, min_dist=0.5,
        obstacle_centers=None, obstacle_covs=None,
        obst_safe_dist=None,
        position_deadzone=0.0,
        steady_state_velocity_radius=None,
        alpha_heading=0.0, alpha_heading_final=0.0,
        heading_radius=0.5, goal_window_steps=50,
        alpha_offset_final=0.0,
        goal_leg_ends=None, final_leg_start=0,
        alpha_bounds=0.0,
        alpha_smooth=0.0,
    ):
        super().__init__(Q=Q, R=alpha_u, loss_bound=loss_bound, sat_bound=sat_bound, xbar=None)
        self.Qs = Qs
        self.Q_final = self.Q if Q_final is None else torch.as_tensor(Q_final, device=device)
        self.xbar = xbar
        self.n_agents = n_agents
        self.position_deadzone = position_deadzone
        self.steady_state_velocity_radius = steady_state_velocity_radius
        self.alpha_heading, self.alpha_heading_final = alpha_heading, alpha_heading_final
        # arrival window: steps at the end of every goal leg over which final position and heading are averaged
        self.heading_radius, self.goal_window_steps = heading_radius, goal_window_steps
        self.alpha_offset_final = alpha_offset_final
        # smoothness: squared change of the rPB offset dxref per step (jumps of the tracked reference), weight alpha_smooth
        self.alpha_smooth = alpha_smooth
        # goal sequence: last step of each goal's leg (None: one goal, the end of the rollout) and the first step of the
        # last goal's leg (the speed term only acts there: earlier goals may be passed at speed)
        self.goal_leg_ends, self.final_leg_start = goal_leg_ends, final_leg_start
        self.alpha_col, self.alpha_obst, self.min_dist = alpha_col, alpha_obst, min_dist
        assert (self.alpha_col is None and self.min_dist is None) or not (self.alpha_col is None or self.min_dist is None)
        if self.alpha_col is not None:
            assert self.n_agents is not None
        # define obstacles
        if obstacle_centers is None:
            self.obstacle_centers = [
                torch.tensor([[-2.5, 0]], device=device),
                torch.tensor([[2.5, 0.0]], device=device),
                torch.tensor([[-1.5, 0.0]], device=device),
                torch.tensor([[1.5, 0.0]], device=device),
            ]
        else:
            self.obstacle_centers = obstacle_centers
        if obstacle_covs is None:
            self.obstacle_covs = [
                torch.tensor([[0.2, 0.2]], device=device)
            ] * len(self.obstacle_centers)
        else:
            self.obstacle_covs = obstacle_covs
        # obstacle cost is 0 beyond this distance from each center (None = no cutoff); needs isotropic covariances
        self.obst_safe_dist = obst_safe_dist
        if self.obst_safe_dist is not None:
            assert all(cov.flatten()[0] == cov.flatten()[1] for cov in self.obstacle_covs)

        # mask
        self.mask = torch.logical_not(torch.eye(self.n_agents, device=device))   # shape = (n_agents, n_agents)

        # arena: cost for a car closer than arena_margin to the edge (quadratic in the depth), weight alpha_bounds (0 = off)
        self.arena_bounds = ((-3.5, 3.5), (-4.0, 6.0))
        self.arena_margin = 1.0  # car radius [m]: cost as soon as the car body reaches the wall
        self.alpha_bounds = alpha_bounds

        assert self.Q.shape[0] == 7 * self.n_agents
        assert self.Q_final.shape == self.Q.shape

    def forward(self, xs, us, es):
        """
        Compute loss.

        Args:
            - xs: tensor of shape (S, T, state_dim)
            - us: tensor of shape (S, T, in_dim)
            - es: tensor of shape (S, T, error_dim)
        Return:
            - loss of shape (1, 1).
        """
        # batch
        x_batch = xs.reshape(*xs.shape, 1)
        u_batch = us.reshape(*us.shape, 1)
        e_batch = es.reshape(*es.shape, 1)
        
        """x_bar = self.xbar.reshape(*self.xbar.shape,1)
        # loss states = 1/T sum_{t=1}^T (x_t-xbar)^T Q (x_t-xbar)
        if self.xbar is not None:
            x_batch_centered = x_batch - x_bar
        else:
            x_batch_centered = x_batch"""
        # speed_idx = [3, 10]
        e_batch = self.apply_position_deadzone(e_batch)
        speed_idx = [7 * i + 3 for i in range(self.n_agents)]
        speed = x_batch[:, :, speed_idx]

        xTQx = torch.matmul(
            torch.matmul(e_batch.transpose(-1, -2), self.Q),
            e_batch
        )   # shape = (S, T, 1, 1)
        loss_x = torch.sum(xTQx, 1) / e_batch.shape[1]   # average over the time horizon. shape = (S, 1, 1)
        # final position: averaged over the arrival window (last goal_window_steps steps) of every goal's leg (the car has to be at
        # the goal during that whole window, not only at its last step), summed over the goals
        goal_leg_ends = self.goal_leg_ends if self.goal_leg_ends is not None else [e_batch.shape[1] - 1]
        loss_x_final = 0
        for end in goal_leg_ends:
            e_window = e_batch[:, end - self.goal_window_steps + 1:end + 1]           # [S, K, nx, 1]
            eTQe = torch.matmul(torch.matmul(e_window.transpose(-1, -2), self.Q_final), e_window)   # [S, K, 1, 1]
            loss_x_final = loss_x_final + eTQe.mean(1)                                # (S, 1, 1)

        if self.steady_state_velocity_radius is not None:
            speed = speed * self.get_steady_state_velocity_mask(es)
        # speed term only while the last goal is active (earlier goals may be passed at speed)
        last_leg = (torch.arange(speed.shape[1], device=speed.device) >= self.final_leg_start).view(1, -1, 1, 1)
        speed = speed * last_leg

        sTQs = torch.matmul(
            torch.matmul(speed.transpose(-1, -2), self.Qs),
            speed
        )   # shape = (S, T, 1, 1)
        loss_speed = torch.sum(sTQs, 1) / speed.shape[1]

        # heading loss: 1 - cos(theta* - theta) per car -> 0 when aligned, 2 when opposite, no +-pi jump
        heading_error = es[:, :, 2::7]                                            # [S, T, n_agents]  theta* - theta
        heading_cost = 1 - torch.cos(heading_error)                               # [S, T, n_agents]

        # running term: weighted by closeness to the goal (soft mask, no hard cut-off that could be dodged)
        position_error = es.reshape(es.shape[0], es.shape[1], self.n_agents, 7)[..., 0:2]
        distance_to_goal = torch.linalg.norm(position_error, dim=-1)              # [S, T, n_agents]
        near_goal = torch.exp(-distance_to_goal**2 / (2 * self.heading_radius**2))   # 1 at the goal, 0.61 at r, 0.14 at 2r
        loss_heading = self.alpha_heading * (heading_cost * near_goal).sum((1, 2)) / es.shape[1]      # [S]

        # terminal term: average over the arrival window of every goal's leg, summed over cars and goals
        loss_heading_final = 0
        for end in goal_leg_ends:
            final_cost = heading_cost[:, end - self.goal_window_steps + 1:end + 1, :]   # [S, K, n_agents]
            loss_heading_final = loss_heading_final + self.alpha_heading_final * final_cost.sum(-1).mean(-1)   # [S]

        loss_heading = loss_heading.reshape(-1, 1, 1)
        loss_heading_final = loss_heading_final.reshape(-1, 1, 1)

        # terminal offset: the rPB must have let go of the car by the end of the rollout,
        # so the pose the terminal terms see is the pose the car keeps afterwards
        final_offset = us[:, -self.goal_window_steps:, :]                      # [S, K, 2*n_agents]  dxref
        loss_offset_final = self.alpha_offset_final * (final_offset**2).sum(-1).mean(-1)   # [S]
        loss_offset_final = loss_offset_final.reshape(-1, 1, 1)

        loss_bounds = self.alpha_bounds * self.f_loss_bounds(x_batch)

        # smoothness: penalize how fast the rPB offset changes, not how large it is (large but smooth offsets, e.g. a
        # hook, stay cheap; jumps of the tracked reference are expensive). A goal switch changes xbar, not dxref -> free
        offset_rate = us[:, 1:, :] - us[:, :-1, :]                            # [S, T-1, 2*n_agents]  change of dxref per step
        loss_smooth = self.alpha_smooth * (offset_rate**2).sum(-1).sum(1) / us.shape[1]   # [S]
        loss_smooth = loss_smooth.reshape(-1, 1, 1)
        
        
        # loss control actions = 1/T sum_{t=1}^T u_t^T R u_t
        uTRu = self.R * torch.matmul(
            u_batch.transpose(-1, -2),
            u_batch
        )   # shape = (S, T, 1, 1)
        loss_u = torch.sum(uTRu, 1) / x_batch.shape[1]    # average over the time horizon. shape = (S, 1, 1)
        # collision avoidance loss
        if self.alpha_col is None:
            loss_ca = 0
        else:
            loss_ca = self.alpha_col * self.f_loss_ca(x_batch)       # shape = (S, 1, 1)
        # obstacle avoidance loss
        if self.alpha_obst is None:
            loss_obst = 0
        else:
            loss_obst = self.alpha_obst * self.f_loss_obst(x_batch) # shape = (S, 1, 1)
        # individual terms of this call, averaged over the samples (for logging only)
        def sample_mean(term):
            return term.detach().mean().item() if torch.is_tensor(term) else float(term)
        self.last_terms = {
            "track": sample_mean(loss_x),
            "final_pos": sample_mean(loss_x_final),
            "speed": sample_mean(loss_speed),
            "effort": sample_mean(loss_u),
            "col": sample_mean(loss_ca),
            "obst": sample_mean(loss_obst),
            "heading": sample_mean(loss_heading),
            "heading_final": sample_mean(loss_heading_final),
            "offset_final": sample_mean(loss_offset_final),
            "bounds": sample_mean(loss_bounds),
            "smooth": sample_mean(loss_smooth),
        }

        # sum up all losses
        loss_val = (loss_x + loss_x_final + loss_u + loss_ca + loss_obst + loss_speed + loss_bounds
                    + loss_heading + loss_heading_final + loss_offset_final + loss_smooth)   # shape = (S, 1, 1)
        # bound
        if self.sat_bound is not None:
            loss_val = torch.tanh(loss_val/self.sat_bound)  # shape = (S, 1, 1)
        if self.loss_bound is not None:
            loss_val = self.loss_bound * loss_val           # shape = (S, 1, 1)
        # average over the samples
        loss_val = torch.sum(loss_val, 0)/xs.shape[0]       # shape = (1, 1)
        return loss_val

    def apply_position_deadzone(self, e_batch):
        if self.position_deadzone <= 0:
            return e_batch

        e_agents = e_batch.reshape(e_batch.shape[0], e_batch.shape[1], self.n_agents, 7, 1)
        position_error = e_agents[:, :, :, 0:2, :]
        other_error = e_agents[:, :, :, 2:, :]
        distance = torch.linalg.norm(position_error, dim=3, keepdim=True)
        scale = torch.clamp(distance - self.position_deadzone, min=0.0) / (distance + 1e-6)
        position_error = position_error * scale
        return torch.cat((position_error, other_error), dim=3).reshape_as(e_batch)

    def get_steady_state_velocity_mask(self, es):
        e_agents = es.reshape(es.shape[0], es.shape[1], self.n_agents, 7)
        distance = torch.linalg.norm(e_agents[:, :, :, 0:2], dim=-1, keepdim=True)
        return (distance <= self.steady_state_velocity_radius)
    
    def f_loss_bounds(self, x_batched):
        qx = x_batched[:, :, 0::7, :]
        qy = x_batched[:, :, 1::7, :]

        x_min, x_max = self.arena_bounds[0]
        y_min, y_max = self.arena_bounds[1]
        margin = self.arena_margin

        left = torch.relu((x_min + margin) - qx)
        right = torch.relu(qx - (x_max - margin))
        bottom = torch.relu((y_min + margin) - qy)
        top = torch.relu(qy - (y_max - margin))

        loss = left**2 + right**2 + bottom**2 + top**2
        return loss.mean(dim=(1, 2, 3)).reshape(-1, 1, 1)
    
    def f_loss_obst(self, x_batched):
        """
        Obstacle avoidance loss.

        Args:
            - x_batched: tensor of shape (S, T, state_dim, 1)
                concatenated states of all agents on the third dimension.

        Return:
            - obstacle avoidance loss of shape (1, 1).
        """
        S, T = x_batched.shape[0], x_batched.shape[1]
        position = x_batched[:, :, :, 0].reshape(S, T, self.n_agents, 7)[..., 0:2]   # (x, y) per car. shape = (S, T, n_agents, 2)
        # sum up loss due to each obstacle, per car
        loss_obst = torch.zeros(S, T, device=x_batched.device)
        for center, cov in zip(self.obstacle_centers, self.obstacle_covs):
            gaussian = normpdf(position, mu=center, cov=cov)                  # shape = (S, T, n_agents)
            if self.obst_safe_dist is not None:
                # Gaussian minus its value at obst_safe_dist from the center: 0 (no gradient) beyond obst_safe_dist
                position_at_safe = center + torch.tensor([[self.obst_safe_dist, 0.0]], device=center.device)
                gaussian_at_safe = normpdf(position_at_safe, mu=center, cov=cov)   # shape = (1)
                gaussian = torch.relu(gaussian - gaussian_at_safe)
            loss_obst = loss_obst + gaussian.sum(-1)                          # shape = (S, T)
        # average over time steps
        loss_obst = loss_obst.sum(1) / T    # shape = (S)
        return loss_obst.reshape(-1, 1, 1)

    def f_loss_ca(self, x_batch):
        """
        Collision avoidance loss.

        Args:
            - x_batched: tensor of shape (S, T, state_dim, 1)
                concatenated states of all agents on the third dimension.


        Return:
            - collision avoidance loss of shape (1, 1).
        """
        min_sec_dist = self.min_dist + 0.2
        # compute pairwise distances
        distance_sq = self.get_pairwise_distance_sq(x_batch)              # shape = (S, T, n_agents, n_agents)
        # 1/d^2 minus its value at the safety distance: continuous, and 0 (no gradient) beyond min_sec_dist
        inverse_distance_sq = 1 / (distance_sq + 1e-3)                    # shape = (S, T, n_agents, n_agents)
        inverse_distance_sq_at_safe = 1 / (min_sec_dist ** 2 + 1e-3)
        loss_ca = (torch.relu(inverse_distance_sq - inverse_distance_sq_at_safe) * self.mask).sum((-1, -2))/2        # shape = (S, T)
        # average over time steps
        loss_ca = loss_ca.sum(1)/loss_ca.shape[1]
        # reshape to S,1,1
        loss_ca = loss_ca.reshape(-1,1,1)
        return loss_ca

    def count_collisions(self, x_batch, min_dist=None):
        """
        Count the number of collisions between agents.

        Args:
            - x_batched: tensor of shape (S, T, state_dim, 1)
                concatenated states of all agents on the third dimension.

        Return:
            - number of collisions between agents.
        """
        if len(x_batch.shape) == 3:
            x_batch = x_batch.reshape(*x_batch.shape, 1)
        distance_sq = self.get_pairwise_distance_sq(x_batch)  # shape = (S, T, n_agents, n_agents)
        min_dist = self.min_dist if min_dist is None else min_dist   # evaluation may use its own safety distance
        col_matrix = (0.0001 < distance_sq) * (distance_sq < min_dist ** 2)  # Boolean collision matrix of shape (S, T, n_agents, n_agents)
        n_coll = col_matrix.sum().item()    # all collisions at all times and across all rollouts
        return n_coll/2                     # each collision is counted twice

    def get_pairwise_distance_sq(self, x_batch):
        """
        Squared distance between pairwise agents.

        Args:
            - x_batched: tensor of shape (S, T, state_dim, 1)
                concatenated states of all agents on the third dimension.

        Return:
            - matrix of shape (S, T, n_agents, n_agents) of squared pairwise distances.
        """
        state_dim_per_agent = int(x_batch.shape[2]/self.n_agents)
        # collision avoidance:
        x_agents = x_batch[:, :, 0::state_dim_per_agent, :]  # start from 0, pick every state_dim_per_agent. shape = (S, T, n_agents, 1)
        y_agents = x_batch[:, :, 1::state_dim_per_agent, :]  # start from 1, pick every state_dim_per_agent. shape = (S, T, n_agents, 1)
        deltaqx = x_agents.repeat(1, 1, 1, self.n_agents) - x_agents.repeat(1, 1, 1, self.n_agents).transpose(-2, -1)   # shape = (S, T, n_agents, n_agents)
        deltaqy = y_agents.repeat(1, 1, 1, self.n_agents) - y_agents.repeat(1, 1, 1, self.n_agents).transpose(-2, -1)   # shape = (S, T, n_agents, n_agents)
        distance_sq = deltaqx ** 2 + deltaqy ** 2             # shape = (S, T, n_agents, n_agents)
        return distance_sq


def normpdf(q, mu, cov):
    """
    PDF of a 2D normal distribution with mean "mu" and diagonal covariance "cov", at each position.

    Args:
        - q: positions (x, y). shape = (..., 2)
        - mu: mean. shape = (1, 2)
        - cov: diagonal of the covariance matrix. shape = (1, 2)

    Return:
        - density at each position. shape = (...)
    """
    d = 2
    mu = mu.view(1, d)
    cov = cov.view(1, d)
    den = (2*torch.pi)**(0.5*d) * torch.sqrt(torch.prod(cov))
    nom = torch.exp((-0.5 * (q - mu)**2 / cov).sum(-1))
    return nom/den
