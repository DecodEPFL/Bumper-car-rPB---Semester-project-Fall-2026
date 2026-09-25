# rPB bumpercar

This project covers learning-based control of bumper cars, from simulation to ROS 2 deployment. The main result is a **robust Performance Boosting (rPB)** controller. It guides two cars to their goals while they avoid obstacles and each other.

All three parts share the same **learned car dynamics model**, `model_kinematic_mlp.pth`. It is an MLP that predicts the car's velocity and slip states, and identical copies sit in each folder.

```
rPB_bumpercar/
├── bumper_car_simulator/   # sandbox: dynamics models, baseline controllers, vanilla NN policies, ONNX export
├── reference_tracking_PB/  # rPB controller: training + evaluation   ← start here
└── bumpercar-ros/          # ROS 2 packages: simulator, visualiser, MLP and rPB controller nodes
```


## 1. `reference_tracking_PB/`: rPB controller

This folder trains and tests the rPB controller for the two-car scenario. It has its own README with full details.

| | Entry point |
|---|---|
| **Train** | `python experiments/bumpercar/run.py` (Colab wrapper: `colab/train_bumpercar.ipynb`) |
| **Evaluate / visualise** | `python sim.py` (loss, collisions, distance metrics, interactive plot, report figures) |
| **Provided checkpoint** | `experiments/bumpercar/trained_pRB/rPB_best.pt` |

In short, the controller outputs a correction `dxref` to each car's reference position, and a P-controller tracks that reference. The controller is a contractive REN driven by the reconstructed disturbance, multiplied by an MLP. It is trained by backpropagating through closed-loop rollouts of the learned dynamics.

## 2. `bumper_car_simulator/`: sandbox and baselines

This folder is a NumPy/PyTorch simulator of one or more cars in a 10 × 10 m arena, stepped at `dt = 0.1 s`. It came before the rPB work and is useful for understanding the car and for comparisons. It contains **no rPB code**.

| File | Content |
|---|---|
| `parameters.py` | Physical car parameters |
| `lagrangian_model.py` | Physics-based bicycle/tire model |
| `mlp_model.py` | Learned dynamics (`best_models/model_kinematic_mlp.pth`). The weights are trained elsewhere; this repo has no sysID training script. |
| `algorithms.py` | Baseline controllers: distributed/centralized CBF, NMPC, position PID, and **vanilla NN policies** (`PolicyMLP`, `PolicyGRU`, `NN_Controller`) |
| `simulator.py` | Main simulation and animation. Choose the model and controller with the flags at the top. |
| `trainNN.py` | Trains the vanilla MLP/GRU go-to-pose policies (single car, fixed goal, no collision terms) with `NN_Controller.train_mlp` |
| `evaluateNN.py` | Evaluates saved policies on a grid of start poses (success rate, RMSE) |
| `onnx-exportt/` | Exports policies and the dynamics MLP to ONNX for ROS |

Trained policies are in `NN_controllers/`, `mlp_control_models/` and `gru_models/`, and loss curves are in `figs/`.

> Note: `train_mlp` saves to `experiment_data/{MLP,GRU}/`. This folder is not in the repo, so create it before training.

## 3. `bumpercar-ros/`: ROS 2 deployment

This folder holds three ament packages. Build them with `colcon build` inside a ROS 2 workspace.

| Package | Content |
|---|---|
| `bump_msgs` | Messages: `EKFState` (`x, y, θ, v_f, β_f, β_r, δ`), `EKFStateArray`, `ControlInput` (`throttle, steering`), `ControlInputArray`, `Recording` |
| `bumpercar_sim` | `arena_simulator` (two cars with MLP dynamics, scenario in `arena_experiment_params.py`), `simulator` (single car), `visualizer` |
| `bumpercar_control` | `rpb_controller` (rPB node) and `mlp_controller` (vanilla MLP policy), with their model weights |

**rPB pipeline:**

```
arena_simulator ──/cars_states (EKFStateArray)──► rpb_controller ──/car0_safe_input, /car1_safe_input (ControlInput)──► arena_simulator
                         └──────────────► visualizer
```

```bash
ros2 run bumpercar_sim arena_simulator
ros2 run bumpercar_sim visualizer
ros2 run bumpercar_control rpb_controller   # optional: --ros-args -p checkpoint_path:=/path/to/ckpt.pt
```

The single-car `simulator` and `mlp_controller` communicate over the `state`, `goal_state` and `control_input` topics, which use `Float64MultiArray`.

**Known issues, to fix before you use the nodes:**
- `rPB_controller_node.py` uses `args` without defining it, and it passes the checkpoint *path* to `load_state_dict`. Load the checkpoint the way `load_controller()` in `reference_tracking_PB/sim.py` does.
- In `bumpercar_control/setup.py`, the `mlp_controller` entry point refers to `controller_node`. The file is actually named `mlp_controller_node.py`.
- The goal in the rPB node (`xbar`) is hardcoded. The arena start poses in `arena_experiment_params.py` also differ from the training scenario in `reference_tracking_PB/experiment_params.py`. Check that they match what the controller was trained on.
- `bumpercar_control` contains adapted copies of `bumpercar_sys.py` and the rPB controller. Keep them in sync with `reference_tracking_PB/`.
