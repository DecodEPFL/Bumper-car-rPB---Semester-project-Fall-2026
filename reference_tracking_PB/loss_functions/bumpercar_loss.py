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
        position_deadzone=0.0,
        steady_state_velocity_radius=None,

    ):
        super().__init__(Q=Q, R=alpha_u, loss_bound=loss_bound, sat_bound=sat_bound, xbar=None)
        self.Qs = Qs
        self.Q_final = self.Q if Q_final is None else torch.as_tensor(Q_final, device=device)
        self.xbar = xbar
        self.n_agents = n_agents
        self.position_deadzone = position_deadzone
        self.steady_state_velocity_radius = steady_state_velocity_radius
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

        # mask
        self.mask = torch.logical_not(torch.eye(self.n_agents, device=device))   # shape = (n_agents, n_agents)

        self.arena_bounds = ((-3.5, 3.5), (-4.0, 6.0))
        self.arena_margin = 0.4  # car radius / safety buffer
        self.alpha_bounds = 1e4 * 0  # tune relative to alpha_obst

        assert self.Q.shape[0] == 7 * self.n_agents
        assert self.Q_final.shape == self.Q.shape

    def forward(self, xs, us,es):
        """
        Compute loss.

        Args:
            - xs: tensor of shape (S, T, state_dim)
            - us: tensor of shape (S, T, in_dim)

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
        loss_x_final = torch.matmul(
            torch.matmul(e_batch[:, -1].transpose(-1, -2), self.Q_final),
            e_batch[:, -1]
        )   # shape = (S, 1, 1)

        if self.steady_state_velocity_radius is not None:
            speed = speed * self.get_steady_state_velocity_mask(es)
        
        sTQs = torch.matmul(
            torch.matmul(speed.transpose(-1, -2), self.Qs),
            speed
        )   # shape = (S, T, 1, 1)
        loss_speed = torch.sum(sTQs, 1) / speed.shape[1] 

        loss_bounds = self.alpha_bounds * self.f_loss_bounds(x_batch)
        
        
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
        # sum up all losses
        loss_val = loss_x + loss_x_final + loss_u + loss_ca + loss_obst + loss_speed + loss_bounds          # shape = (S, 1, 1)
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
        qx = x_batched[:, :, 0::7, :]   # x of all agents. shape = (S, T, n_agents, 1)
        qy = x_batched[:, :, 1::7, :]   # y of all agents. shape = (S, T, n_agents, 1)
        # batched over all samples and all times of [x agent 1, y agent 1, ..., x agent n, y agent n]
        q = torch.cat((qx,qy), dim=-1).view(x_batched.shape[0], x_batched.shape[1], 1,-1).squeeze(dim=2)    # shape = (S, T, 2*n_agents)
        # sum up loss due to each obstacle #TODO
        for ind, (center, cov) in enumerate(zip(self.obstacle_centers, self.obstacle_covs)):
            if ind == 0:
                loss_obst = normpdf(q, mu=center, cov=cov)   # shape = (S, T)
            else:
                loss_obst += normpdf(q, mu=center, cov=cov)  # shape = (S, T)
        # average over time steps
        loss_obst = loss_obst.sum(1) / loss_obst.shape[1]    # shape = (S)
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
        # compute and sum up loss when two agents are too close
        loss_ca = (1/(distance_sq + 1e-3) * (distance_sq.detach() < (min_sec_dist ** 2)) * self.mask).sum((-1, -2))/2        # shape = (S, T)
        # average over time steps
        loss_ca = loss_ca.sum(1)/loss_ca.shape[1]
        # reshape to S,1,1
        loss_ca = loss_ca.reshape(-1,1,1)
        return loss_ca

    def count_collisions(self, x_batch):
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
        col_matrix = (0.0001 < distance_sq) * (distance_sq < self.min_dist ** 2)  # Boolean collision matrix of shape (S, T, n_agents, n_agents)
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


def normpdf(q, mu, cov):  #TODO
    """
    PDF of normal distribution with mean "mu" and covariance "cov".

    Args:
        - q: shape(S, T, state_dim_per_agent)
        - mu:
        - cov:

    Return:
            -
    """
    d = 2
    mu = mu.view(1, d)
    cov = cov.view(1, d)  # the diagonal of the covariance matrix
    qs = torch.split(q, 2, dim=-1)
    for ind, qi in enumerate(qs):
        # if qi[1]<1.5 and qi[1]>-1.5:
        den = (2*torch.pi)**(0.5*d) * torch.sqrt(torch.prod(cov))
        nom = torch.exp((-0.5 * (qi - mu)**2 / cov).sum(-1))
        # out = out + num/den
        if ind == 0:
            out = nom/den
        else:
            out += nom/den
    return out
