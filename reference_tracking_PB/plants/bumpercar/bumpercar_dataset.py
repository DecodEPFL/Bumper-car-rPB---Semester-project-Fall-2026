import torch
from plants import CostumDataset

class BumpercarDataset(CostumDataset):
    """
    One reference per rollout: every sample draws a family, then one of its variants (both uniformly); a variant gives
    every car a (start zone, goal zone). Start and goal poses follow the same rules (a car starts where it last arrived);
    cars start at rest. Data per sample [horizon, 2 * 7 * n_agents]: the start state at t = 0 (first half), the goal
    state at every step (second half).
    """
    def __init__(
        self,
        random_seed,
        horizon,
        zones,
        families,
        sampling,
        n_agents=2,
    ):
        exp_name = 'bumpercar'
        file_name = 'data_T'+str(horizon)+'_agents'+str(n_agents)+'_RS'+str(random_seed)+'.pkl'

        super().__init__(random_seed, horizon, exp_name, file_name)

        self.n_agents = n_agents
        # experiment_params.getFamilies: zones {name: (x_min, x_max, y_min, y_max)}, families {name: [variant, ...]}
        # with a variant = one (start zone, goal zone) per car, and the sampling rules
        self.zones, self.families = zones, families
        self.allowed = sampling["allowed"]                                      # (x_min, x_max, y_min, y_max) for car centres
        self.obstacle_centers = torch.tensor(sampling["obstacle_centers"], dtype=torch.float32)   # [n_obstacles, 2]
        self.obstacle_clearance = sampling["obstacle_clearance"]                # [m] from the obstacle centres
        self.approach_length = sampling["approach_length"]                      # [m] free run-up behind a start / goal
        self.car_spacing = sampling["car_spacing"]                              # [m] between the cars' starts / goals
        self.min_move = sampling["min_move"]                                    # [m] goal from the car's own start

    def in_allowed_area(self, position):
        """True if a car centre at position (x, y) is inside the allowed area and clear of the obstacles."""
        x_min, x_max, y_min, y_max = self.allowed
        inside = x_min <= position[0] <= x_max and y_min <= position[1] <= y_max
        clear = torch.linalg.norm(self.obstacle_centers - position, dim=-1).min() >= self.obstacle_clearance
        return bool(inside and clear)

    def sample_position(self, zone):
        """Uniform position in a zone (x_min, x_max, y_min, y_max)."""
        x_min, x_max, y_min, y_max = zone
        position = torch.stack((x_min + (x_max - x_min) * torch.rand(()), y_min + (y_max - y_min) * torch.rand(())))
        assert self.in_allowed_area(position), f"zone {zone} reaches outside the allowed area"
        return position

    def sample_feasible_heading(self, position):
        """Heading uniform over the full circle, redrawn until the approach point approach_length behind position
        (where the final straight run-up to that pose starts) is in the allowed area. For goals and starts: a car starts
        where it last arrived, so start poses follow the same rule as goal poses."""
        while True:
            heading = torch.pi * (2 * torch.rand(()) - 1)
            approach = position - self.approach_length * torch.stack((torch.cos(heading), torch.sin(heading)))
            if self.in_allowed_area(approach):
                return heading

    def spaced(self, a, b):
        """True if every point of a is at least car_spacing from the points of b of the other cars ([n, 2] each)."""
        distance = torch.cdist(a, b) + 1e6 * torch.eye(self.n_agents)          # not the car itself
        return bool(distance.min() >= self.car_spacing)

    def _generate_data(self, num_samples):
        nx = 7 * self.n_agents
        data = torch.zeros(num_samples, self.horizon, 2 * nx)
        names = list(self.families)

        for rollout_num in range(num_samples):
            # 1) family, then variant: one (start zone, goal zone) per car
            variants = self.families[names[torch.randint(len(names), ()).item()]]
            variant = variants[torch.randint(len(variants), ()).item()]

            # 2) positions: uniform in the zones; redrawn (all cars) until starts, goals, and every goal vs the other
            #    cars' starts are car_spacing apart and every goal is min_move from its own start
            while True:
                starts = torch.stack([self.sample_position(self.zones[start_zone]) for start_zone, _ in variant])   # [n, 2]
                goals = torch.stack([self.sample_position(self.zones[goal_zone]) for _, goal_zone in variant])      # [n, 2]
                own_move = torch.linalg.norm(goals - starts, dim=-1)                                              # [n]
                if (self.spaced(starts, starts) and self.spaced(goals, goals) and self.spaced(goals, starts)
                        and bool(own_move.min() >= self.min_move)):
                    break

            # 3) headings: start and goal uniform over the feasible directions (a start is a previous goal); at rest
            for i in range(self.n_agents):
                start_heading = self.sample_feasible_heading(starts[i])
                goal_heading = self.sample_feasible_heading(goals[i])
                data[rollout_num, 0, 7 * i:7 * i + 3] = torch.cat((starts[i], start_heading.view(1)))
                data[rollout_num, :, nx + 7 * i:nx + 7 * i + 3] = torch.cat((goals[i], goal_heading.view(1)))

        assert data.shape[0] == num_samples
        return data
