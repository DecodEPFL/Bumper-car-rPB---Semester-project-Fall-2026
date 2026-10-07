import torch
from plants import CostumDataset

class BumpercarDataset(CostumDataset):
    def __init__(
        self,
        random_seed,
        horizon,
        x_final_limit,
        y_final_limit,
        x0,
        x_final,
        start_heading_range,
        car_init_radius=1.0,
        final_car_radius=1.0,
        final_car_min_dist=2,
        n_agents=2,
        goal_heading_range=0.0,
        poses=None,
        missions=None,
        leg_steps=None,
    ):
        exp_name = 'bumpercar'
        file_name = 'data_T'+str(horizon)+'_stdini'+str(car_init_radius)+'_agents'+str(n_agents)+'_RS'+str(random_seed)+'.pkl'
        
        super().__init__(random_seed, horizon, exp_name, file_name)
        
        self.car_init_radius = car_init_radius
        self.n_agents = n_agents

        self.x0 = x0
        self.x_final = x_final
        self.x_final_limit = x_final_limit
        self.y_final_limit = y_final_limit
        self.final_car_min_dist = final_car_min_dist
        self.final_car_radius = final_car_radius
        self.start_heading_range = start_heading_range  # [rad] half-width of the start heading spread
        self.goal_heading_range = goal_heading_range    # [rad] half-width of the target heading spread; 0 = off
        # missions (experiment_params.getMissions): named poses {name: (x, y, theta)} and a list of missions, each with one
        # (start, goal of leg 1, goal of leg 2, ...) of pose names per car; one mission is drawn per sample
        self.poses, self.missions = poses, missions
        self.n_legs = len(missions[0][0]) - 1
        self.leg_starts, self.goal_leg_ends, self.final_leg_start = self.goal_schedule(horizon, self.n_legs, leg_steps)

    @staticmethod
    def goal_schedule(horizon, n_goals, leg_steps=None):
        """
        Time-based switching between the goals: every goal but the last has a leg of leg_steps steps
        (default: equal legs, horizon // n_goals); the last goal takes the rest of the horizon.
        Returns the first step of each leg, the last step of each leg and the first step of the last leg.
        """
        leg_steps = horizon // n_goals if leg_steps is None else leg_steps
        leg_starts = [k * leg_steps for k in range(n_goals)]
        goal_leg_ends = [start - 1 for start in leg_starts[1:]] + [horizon - 1]
        return leg_starts, goal_leg_ends, leg_starts[-1]

    def sample_uniform_disk(self, radius, device):
        angle = 2.0 * torch.pi * torch.rand((), device=device)
        r = radius * torch.sqrt(torch.rand((), device=device))
        return torch.stack((r * torch.cos(angle), r * torch.sin(angle)))
    
    def generate_vector_with_min_distance(self):
        while True:
            # Generate the first vector (x1 in [-2, 2], x2 in [0, 1])
            interval_x1 = self.x_final_limit
            interval_x2 = self.y_final_limit
            min_distance = self.final_car_min_dist
            
            vec1 = torch.tensor([
                torch.empty(1).uniform_(interval_x1[0], interval_x1[1]),
                torch.empty(1).uniform_(interval_x2[0], interval_x2[1])
            ]).flatten()

            # Generate the second vector (x1 in [-2, 2], x2 in [0, 1])
            vec2 = torch.tensor([
                torch.empty(1).uniform_(interval_x1[0], interval_x1[1]),
                torch.empty(1).uniform_(interval_x2[0], interval_x2[1])
            ]).flatten()
            # Check the Euclidean distance
            distance = torch.norm(vec1 - vec2)
            if distance >= min_distance:
                # Concatenate vec1 and vec2 to form a (4,) vector
                return torch.cat((vec1, vec2))
            
    def _generate_data(self, num_samples):
        # Initial-condition / disturbance block
        state_dim_x0 = 7 * self.n_agents

        # Full reference block, same dimension as state
        state_dim_ref = 7 * self.n_agents

        state_dim = state_dim_x0 + state_dim_ref
        data = torch.zeros(num_samples, self.horizon, state_dim)

        for rollout_num in range(num_samples):
            # mission of this sample (all missions equally likely): one (start, goal of leg 1, ...) of pose names per car
            mission = self.missions[torch.randint(len(self.missions), ()).item()]
            x0_sample = torch.zeros(state_dim_x0)
            goal_samples = [torch.zeros(state_dim_x0) for _ in range(self.n_legs)]

            for i, names in enumerate(mission):
                base = 7 * i
                # start: nominal pose + position offset (disk) + heading offset U(-range, range)
                start = torch.tensor(self.poses[names[0]], dtype=torch.float32)
                start[0:2] += self.sample_uniform_disk(self.car_init_radius, start.device)
                start[2] += self.start_heading_range * (2 * torch.rand(()) - 1)
                x0_sample[base:base + 3] = start
                # goals: the same pose name as in the previous position is the same pose (parked: the goal is the
                # sampled start; staying: the goal of the previous leg); any other pose gets its own random offset
                previous_name, previous_pose = names[0], start
                for leg, name in enumerate(names[1:]):
                    if name == previous_name:
                        goal = previous_pose
                    else:
                        goal = torch.tensor(self.poses[name], dtype=torch.float32)
                        goal[0:2] += self.sample_uniform_disk(self.final_car_radius, goal.device)
                        # target heading: pose heading + U(-range, range), wrapped to (-pi, pi]
                        if self.goal_heading_range > 0:
                            heading = goal[2] + self.goal_heading_range * (2 * torch.rand(()) - 1)
                            goal[2] = torch.atan2(torch.sin(heading), torch.cos(heading))
                    goal_samples[leg][base:base + 3] = goal
                    previous_name, previous_pose = name, goal

            data[rollout_num, 0, :state_dim_x0] = x0_sample

            # reference: the goals in visiting order, each during its leg (the last goal until the end of the horizon)
            leg_ends = self.leg_starts[1:] + [self.horizon]
            for goal_sample, start, end in zip(goal_samples, self.leg_starts, leg_ends):
                data[rollout_num, start:end, state_dim_x0:] = goal_sample.view(1, -1)

        assert data.shape[0] == num_samples
        return data


        
