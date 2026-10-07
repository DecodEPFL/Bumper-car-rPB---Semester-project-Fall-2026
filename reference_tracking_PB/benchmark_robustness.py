import os
import sys
import torch
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from config import device
from plants import BumpercarSystem, car_params
from sim import load_controller, TRAINED_PBR_MODEL_PATH
from experiment_params import getCarInitParams

def run_stress_scenario(name, system, controller, x0_generator, n_rollouts=100, horizon=300, min_dist=2.0):
    controller.eval()
    system.eval()
    
    collisions = 0
    obstacle_hits = 0
    min_car_distances = []
    final_errors = []
    
    _, x_final_nom, obstacle_centers, _, _, _ = getCarInitParams(device)
    centers = torch.stack([c.to(device).flatten() for c in obstacle_centers])
    
    for _ in range(n_rollouts):
        x0, xbar = x0_generator()
        data = torch.zeros(1, horizon, 28, device=device)
        data[:, 0, :14] = x0
        data[:, :, 14:] = xbar.view(1, 1, -1)
        
        with torch.no_grad():
            x_log, _, _ = system.rollout(controller, data, train=False)
            
        pos1 = x_log[0, :, 0:2]
        pos2 = x_log[0, :, 7:9]
        d_cars = torch.linalg.norm(pos1 - pos2, dim=-1)
        min_d = d_cars.min().item()
        min_car_distances.append(min_d)
        
        if min_d < min_dist:
            collisions += 1
            
        # Distanza dagli ostacoli (r = 1.5 m)
        d_obs1 = torch.linalg.norm(pos1.unsqueeze(1) - centers.unsqueeze(0), dim=-1).min()
        d_obs2 = torch.linalg.norm(pos2.unsqueeze(1) - centers.unsqueeze(0), dim=-1).min()
        if min(d_obs1.item(), d_obs2.item()) < 1.5:
            obstacle_hits += 1
            
        err1 = torch.linalg.norm(pos1[-1] - xbar[0:2]).item()
        err2 = torch.linalg.norm(pos2[-1] - xbar[7:9]).item()
        final_errors.append((err1 + err2) / 2.0)
        
    print(f"\n--- {name} ({n_rollouts} rollouts) ---")
    print(f"Collision Rate: {collisions / n_rollouts * 100:.1f}% ({collisions}/{n_rollouts})")
    print(f"Min Car-Car Distance: {np.min(min_car_distances):.3f} m (Avg Min: {np.mean(min_car_distances):.3f} m)")
    print(f"Obstacle Hit Rate: {obstacle_hits / n_rollouts * 100:.1f}%")
    print(f"Final Target Error: {np.mean(final_errors):.3f} m")

def main():
    if not os.path.exists(TRAINED_PBR_MODEL_PATH):
        raise FileNotFoundError(f"Checkpoint non trovato: {TRAINED_PBR_MODEL_PATH}")
        
    system = BumpercarSystem(params=car_params, dt=0.04).to(device)
    controller = load_controller(system, TRAINED_PBR_MODEL_PATH)
    
    x0_nom, x_final_nom, _, _, _, _ = getCarInitParams(device)
    
    # Test 1: Nominale
    def gen_nominal():
        x0 = x0_nom.clone().to(device)
        return x0, x_final_nom.to(device)
    run_stress_scenario("Scenario 1: Nominale", system, controller, gen_nominal, n_rollouts=50)

    # Test 2: Heading OOD (orientamenti iniziali ruotati casualmente)
    def gen_heading_ood():
        x0 = x0_nom.clone().to(device)
        x0[2] += (torch.rand(1, device=device).item() - 0.5) * np.pi
        x0[9] += (torch.rand(1, device=device).item() - 0.5) * np.pi
        return x0, x_final_nom.to(device)
    run_stress_scenario("Scenario 2: Heading OOD", system, controller, gen_heading_ood, n_rollouts=50)

    # Test 3: Velocità Iniziale Verso il Centro
    def gen_velocity_ood():
        x0 = x0_nom.clone().to(device)
        x0[3] = 0.8  # 0.8 m/s per car 1
        x0[10] = 0.8 # 0.8 m/s per car 2
        return x0, x_final_nom.to(device)
    run_stress_scenario("Scenario 3: Velocità Iniziale OOD", system, controller, gen_velocity_ood, n_rollouts=50)

if __name__ == "__main__":
    main()