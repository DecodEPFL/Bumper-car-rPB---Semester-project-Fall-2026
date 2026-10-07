#!/usr/bin/env python3
"""
diagnose_test_collisions.py
Analisi diagnostica ad alta risoluzione delle collisioni residue sul test set.
Estrae istanti temporali t, coordinate (x, y), velocità e distanze euclidee.
"""

import os
import sys
import torch
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, BASE_DIR)

from config import device
from experiment_params import getCarInitParams, getLossParams
from plants import BumpercarSystem, car_params
from sim import (
    load_controller,
    make_eval_data,
    OBSTACLE_RADIUS,
    CAR_SAFE_DISTANCE,
    TRAINED_PBR_MODEL_PATH,
    EVAL_HORIZON,
    EVAL_NUM_ROLLOUTS,
    EVAL_NUM_TEST_ROLLOUTS,
    EVAL_RANDOM_SEED,
    DT,
)


def run_collision_telemetry():
    _, _, obstacle_centers, _, _, _ = getCarInitParams(device)
    _, _, _, _, _, _, _, _, min_dist = getLossParams(device)

    system = BumpercarSystem(
        params=car_params,
        x_init=None,
        u_init=None,
        dt=DT,
        r_obs_safe=OBSTACLE_RADIUS,
        d_car_safe=CAR_SAFE_DISTANCE,
        obstacle_centers=obstacle_centers,
    ).to(device)

    controller = load_controller(system, TRAINED_PBR_MODEL_PATH)

    _, test_data = make_eval_data(
        horizon=EVAL_HORIZON,
        num_rollouts=EVAL_NUM_ROLLOUTS,
        num_test_rollouts=EVAL_NUM_TEST_ROLLOUTS,
        random_seed=EVAL_RANDOM_SEED,
    )

    print(f"[INFO] Avvio rollout diagnostico su {EVAL_NUM_TEST_ROLLOUTS} scenari di Test...")
    with torch.no_grad():
        x_log, _, _ = system.rollout(controller, test_data, train=False)

    # x_log: [B, T, 14]
    B, T, _ = x_log.shape
    p1 = x_log[:, :, 0:2]
    v1 = x_log[:, :, 3]
    p2 = x_log[:, :, 7:9]
    v2 = x_log[:, :, 10]

    d_12 = torch.linalg.norm(p1 - p2, dim=-1)  # [B, T]
    collision_mask = d_12 < min_dist

    total_collision_steps = int(collision_mask.sum().item())
    rollouts_with_collisions = torch.any(collision_mask, dim=-1).nonzero(as_tuple=True)[0].cpu().tolist()

    print("\n================ TELEMETRIA COLLISIONI TEST SET ================")
    print(f"Step totali valutati:            {B * T:,}")
    print(f"Step in collisione (d < {min_dist:.2f} m): {total_collision_steps}")
    print(f"Rollout impattati ({len(rollouts_with_collisions)}/{B}):    {rollouts_with_collisions}")

    if total_collision_steps == 0:
        print("[SUCCESS] Nessuna collisione riscontrata!")
        return

    # Estrazione campioni di collisione
    coll_b, coll_t = collision_mask.nonzero(as_tuple=True)
    dists = d_12[coll_b, coll_t].cpu().numpy()
    p1_coords = p1[coll_b, coll_t].cpu().numpy()
    p2_coords = p2[coll_b, coll_t].cpu().numpy()
    v1_vals = v1[coll_b, coll_t].cpu().numpy()
    v2_vals = v2[coll_b, coll_t].cpu().numpy()
    t_vals = coll_t.cpu().numpy()

    # 1. Analisi per Severità (Distanza Minima)
    close_grazing = np.sum((dists >= 1.96) & (dists < 2.00))
    mild_penetration = np.sum((dists >= 1.90) & (dists < 1.96))
    severe_penetration = np.sum(dists < 1.90)

    print("\n--- 1. Distribuzione della Severità (Penetrazione) ---")
    print(f"  Sfioramento al limite (1.96 m <= d < 2.00 m): {close_grazing} ({100*close_grazing/total_collision_steps:.1f}%)")
    print(f"  Penetrazione lieve   (1.90 m <= d < 1.96 m): {mild_penetration} ({100*mild_penetration/total_collision_steps:.1f}%)")
    print(f"  Penetrazione severa  (d < 1.90 m):           {severe_penetration} ({100*severe_penetration/total_collision_steps:.1f}%)")
    print(f"  Distanza minima registrata assoluta:         {np.min(dists):.4f} m")

    # 2. Analisi Spaziale (Posizione y nel Corridoio vs Target)
    mean_y = 0.5 * (np.abs(p1_coords[:, 1]) + np.abs(p2_coords[:, 1]))
    center_choke = np.sum(mean_y < 1.0)
    intermediate = np.sum((mean_y >= 1.0) & (mean_y < 2.5))
    target_area = np.sum(mean_y >= 2.5)

    print("\n--- 2. Distribuzione Spaziale lungo l'Asse Longitudinale (y) ---")
    print(f"  Collo di bottiglia centrale (|y| < 1.0 m):    {center_choke} ({100*center_choke/total_collision_steps:.1f}%)")
    print(f"  Zona intermedia/imbuto      (1.0 m <= |y| < 2.5 m): {intermediate} ({100*intermediate/total_collision_steps:.1f}%)")
    print(f"  Fase finale vicino ai target (|y| >= 2.5 m):   {target_area} ({100*target_area/total_collision_steps:.1f}%)")

    # 3. Analisi Temporale (Step t)
    early_t = np.sum(t_vals < 100)
    mid_t = np.sum((t_vals >= 100) & (t_vals < 300))
    late_t = np.sum(t_vals >= 300)

    print("\n--- 3. Distribuzione Temporale (Step t) ---")
    print(f"  Fase iniziale (t < 100):                       {early_t}")
    print(f"  Fase centrale incrocio (100 <= t < 300):       {mid_t}")
    print(f"  Fase finale stazionamento (t >= 300):          {late_t}")

    print("\n--- Prime 5 Violazioni Registrate ---")
    for i in range(min(5, total_collision_steps)):
        print(
            f"  Rollout {coll_b[i].item():3d} | Step {t_vals[i]:3d} | "
            f"d={dists[i]:.3f} m | "
            f"Car1=({p1_coords[i, 0]:.2f}, {p1_coords[i, 1]:.2f}, v={v1_vals[i]:.2f}) | "
            f"Car2=({p2_coords[i, 0]:.2f}, {p2_coords[i, 1]:.2f}, v={v2_vals[i]:.2f})"
        )
    print("=================================================================\n")


if __name__ == "__main__":
    run_collision_telemetry()