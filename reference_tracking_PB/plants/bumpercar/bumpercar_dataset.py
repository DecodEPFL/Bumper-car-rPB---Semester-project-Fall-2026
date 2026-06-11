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
        std_init_theta,
        car_init_radius=1.0,
        final_car_radius=1.0,
        final_car_min_dist=2,
        n_agents=2,
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
        self.std_init_theta = std_init_theta

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
            x0_sample = self.x0.clone()
            noise = torch.zeros_like(self.x0)
            x_final_sample = self.x_final.clone()
            
            for i in range(self.n_agents):
                base = 7 * i
                noise[base:base + 2] = self.sample_uniform_disk(
                    self.car_init_radius,
                    self.x0.device,
                )
                noise[base + 2] = self.std_init_theta * torch.rand(())
                x_final_sample[base:base + 2] += self.sample_uniform_disk(
                    self.final_car_radius,
                    self.x_final.device,
                )

            x0_sample = x0_sample + noise
            data[rollout_num, 0, :state_dim_x0] = x0_sample
            data[rollout_num, :, state_dim_x0:] = x_final_sample.view(1, -1)

        assert data.shape[0] == num_samples
        return data


        
