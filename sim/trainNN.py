import numpy as np
import torch
import matplotlib.pyplot as plt
from algorithms import NN_Controller
from parameters import params

def gaussianCarDistribution(car_state_init, car_state_final, std_init, std_final, car_number):
    state_size = car_state_init.shape[0]
    cars_state_init = np.array([car_state_init])
    cars_state_final = np.array([car_state_final])
    for i in range(1, car_number):
        new_state = car_state_init + std_init * (np.random.rand(state_size)-0.5) * np.array([1,1,1,0,0,0,0])
        final_state = car_state_final + std_final * (np.random.rand(state_size)-0.5) * np.array([0.0,0.0,1,0,0,0,0])
        cars_state_init = np.vstack([cars_state_init, new_state])
        cars_state_final = np.vstack([cars_state_final, final_state])
    
    return cars_state_init, cars_state_final

def initGridOfCars(car_state_final):
    cars_state_init = np.array([])
    for i in range(-1, 2):
        for j in range(-1, 2):
            if not (j == 0 and i == 0):
                for theta in [-np.pi, -np.pi/2, -np.pi/4, 0, np.pi/2, np.pi]:
                    init_state = car_state_final + i * np.array([2,0,0,0,0,0,0]) + j * np.array([0,2,0,0,0,0,0]) + theta * np.array([0,0,1,0,0,0,0])
                    cars_state_init = np.vstack([cars_state_init, init_state])
    return cars_state_init


if __name__ == "__main__":

    # variables for car instances
    n_cars = 320
    std_init_pos = 10   # variance
    std_final_pos = 0   

    car_state_init = np.array([5.0, 5.0, 0, 5.0, 0.0, 0.0, 0.0])
    car_state_final = np.array([5.0, 5.0, 0, 0, 0.0, 0.0, 0.0])

    cars_state_init, final_points = gaussianCarDistribution(car_state_init, car_state_final, std_init_pos , std_final_pos, n_cars)
    cars_state = cars_state_init[0:n_cars]
    init_points = torch.tensor(cars_state_init, dtype=torch.float32)
    final_points = torch.tensor(final_points, dtype=torch.float32)

    def experiment_ncars(n_cars, type, name, pretrained = False):
        cars_state_init, final_points = gaussianCarDistribution(car_state_init, car_state_final, std_init_pos , std_final_pos, n_cars)
        init_points = torch.tensor(cars_state_init, dtype=torch.float32)
        final_points = torch.tensor(final_points, dtype=torch.float32)
        nn_controller = NN_Controller(type=type, params=params)
        
        if pretrained:
            nn_controller.load_weigths("sim/NN_controllers/GRU/policy_weights_GRU_50_cars.pth")
            pass
        
        if type == "MLP":
            print("Training MLP controller")
            nn_controller.train_mlp(init_points=init_points, final_points=final_points, car_state_init=cars_state_init, 
                                    params=params, experiment_name=name)
        
        elif type == "GRU":
            print("Training GRU model")
            nn_controller.train_mlp(init_points=init_points, final_points=final_points, car_state_init=cars_state_init, 
                                    params=params, experiment_name=name)
    
    def get_final_eval(type, name):
        validation_losses = np.load(f"experiment_data/{type}/val_means_{name}.npy")
        print(f"{name}: {validation_losses[-1]}")
        
    ## Train the network
    ########################################
    ## Optional: load pre-trained weights ##
    # mlp_controller.load_weigths("mlp_control_models/pos_localFrame_w_angle.pth")
    # mlp_controller.load_weigths("mlp_control_models/policy_weight_BEST_final_body_with_orientation.pth")
    # mlp_controller.load_weigths("policy_weights_.pth")
    ########################################
    
    # experiment_ncars(40, "40cars_largerNet", pretrained = False)
    # experiment_ncars(100, "100cars_largerNet", pretrained = False)
    
    
    # experiment_ncars(50, "GRU", "GRU_50_cars", pretrained = False)
    # experiment_ncars(20, "GRU", "GRU_100_cars", pretrained = False)
    experiment_ncars(100, "GRU", "GRU_100_cars", pretrained = False)
    # experiment_ncars(200, "GRU", "GRU_200_cars", pretrained = True)
    # experiment_ncars(500, "GRU", "GRU_500_cars", pretrained = True)
    # experiment_ncars(1000, "GRU", "GRU_1000_cars", pretrained = True)
    # experiment_ncars(2000, "GRU", "GRU_2000_cars", pretrained = False)
    
    # get_final_eval("GRU", "GRU_200_cars")
    # get_final_eval("GRU", "GRU_50_cars")
    # get_final_eval("200cars_largerNet")
    # get_final_eval("500cars_largerNet")
    
    # get_final_eval("40cars_largerNet")
    # get_final_eval("100cars_largerNet")
    # get_final_eval("200cars_largerNet")
    # get_final_eval("500cars_largerNet")
    # get_final_eval("1000cars_largerNet")
    # get_final_eval("2000cars_largerNet")
     
    # experiment_ncars(200, "200cars_preTrain", pretrained = True)
    # experiment_ncars(500, "500cars_preTrain", pretrained = True)
    # experiment_ncars(1000, "1000cars_preTrain", pretrained = True)
    # experiment_ncars(2000, "2000cars_preTrain", pretrained = True)
    
    
    # get_final_eval("200cars")
    # get_final_eval("500cars")
    # get_final_eval("1000cars")
    # get_final_eval("2000cars")

    # get_final_eval("200cars_preTrain")
    # get_final_eval("500cars_preTrain")
    # get_final_eval("MLP", "1000cars_preTrain")
    # get_final_eval("2000cars_preTrain")


"""
THE MODELS: 
For the "larger-net":
self.net = nn.Sequential(
            nn.Linear(8, hidden[0]),
            nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]),
            nn.ReLU(),
            nn.Linear(hidden[1], hidden[1]),
            nn.ReLU(),
            nn.Linear(hidden[1], 2),
        )


The "normals":
self.net = nn.Sequential(
            nn.Linear(8, hidden[0]),
            nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]),
            nn.ReLU(),
            nn.Linear(hidden[1], 2),

"""
