# rPB controller for the bumpercar (reference tracking)

PyTorch code to train and evaluate a **robust Performance Boosting (rPB)** controller for two bumper cars.

**Task.** Two cars start near `(-2, -2.5)` and `(2, -2.5)` and swap sides: each must reach a goal near `(2, 3)` or `(-2, 3)`, so their paths cross between two obstacles at `(-2, 0)` and `(2, 0)`. They must not collide with each other. See [Dataset, obstacles and collisions](#dataset-obstacles-and-collisions) for exactly how this is set up.

**Control architecture.**

```
                 ┌─────────────── BumpercarSystem (plant) ───────────────┐
 xbar (goal) ──► │  ref = xbar + dxref ──► P-controller ──► MLP dynamics  │ ──► x
                 └───────────────────────────────▲────────────────────────┘
                                                 │ dxref
                         PerfBoostController: REN(w) * MLP([w, xbar])
                                                 ▲
                                w = x − f(x_prev, u_prev)  (reconstructed disturbance)
```

- The **plant** is two cars. Each car has the state `[x, y, θ, v_f, β_f, β_r, δ]`. The dynamics are a learned MLP (`plants/bumpercar/model_kinematic_mlp.pth`), stepped at `dt = 0.04 s`. A fixed position P-controller inside the plant turns a reference position into throttle and steering.
- The **rPB controller** outputs a *reference offset* `dxref` for each car. It reconstructs the disturbance `w` from the plant model and feeds it to a contractive REN. This keeps the closed loop stable for any parameter values. A small MLP scales the REN output based on the goal.
- **Training** backpropagates a loss through whole closed-loop rollouts. The loss combines tracking error, terminal error, speed near the goal, control effort, state bounds, car–car collisions and obstacles.

## Quickstart

The commands below assume you are in this folder (`reference_tracking_PB/`). The scripts find their files relative to their own location, so they also work from any other directory.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Train

```bash
python experiments/bumpercar/run.py
```

A quick smoke test, which takes about a minute on CPU:

```bash
python experiments/bumpercar/run.py --epochs 1 --num-rollouts 4 --batch-size 2 --horizon 20
```

Other useful flags: `--no-col-av`, `--no-obst-av`, `--horizon`, `--dim-internal`, `--dim-nl`, `--alpha-col`, `--alpha-obst`, `--load-controller <ckpt>` (warm start), `--save-path <dir>`. See `experiments/bumpercar/arg_parser.py` for the full list.

Each run writes to `experiments/bumpercar/saved_results/perf_boost_<MM_DD_HH_MM_SS>/` (or `--save-path`):

| Output | Content |
|---|---|
| `log` | Arguments, per-epoch train and validation loss, final train/test loss and collision counts |
| `checkpoints/checkpoint_epoch_XXXXX.pt`, `checkpoint_latest.pt` | **Full controller** (`controller_state_dict`, `ren_state_dict`, `mlp_state_dict`), optimizer state and args |
| `trained_controller.pt` | **REN weights only**, without the MLP. Use a file from `checkpoints/` when you evaluate or deploy. |
| `CL_P_only.pdf` | Closed loop with the P-controller alone (`ZeroController`, no rPB), from the nominal start to the nominal goal. This is the baseline. |
| `CL_untrained.pdf` / `CL_trained.pdf` | Closed-loop trajectory of one test sample before and after training, with the obstacles |
| `U_over_time.pdf` | rPB output `dxref` (the offset to each car's reference) over time for that sample, after training |

The dataset is regenerated at the start of every run, and it is fully determined by `--random-seed` and `--horizon`. Git ignores `saved_results/`.

### Reproducing the provided checkpoint

The run that produced `experiments/bumpercar/trained_pRB/rPB_best.pt` took place on Colab. Its `log` was saved to Google Drive and is not in this repo. The settings below come from the metadata stored in the checkpoint (`args`, `Q`, `Q_final`, `epoch`, and the losses):

| Setting | Value |
|---|---|
| Horizon | 300 steps (12 s at `dt = 0.04`) |
| Training rollouts / batch size | 400 / 100 |
| Epochs / checkpoint saved at | 500 / epoch 460 (the best validation loss) |
| Learning rate | 2e-3 (Adam) |
| `alpha_col` / `alpha_obst` / `alpha_u` | 1500 / 20000 / 2.5e-4 |
| REN `dim_internal` / `dim_nl` / `cont_init_std` | 8 / 8 / 0.1 |
| `Q` (x, y of each car) | 0.1333 (= 2/15). All other states have weight 0. |
| `Q_final` | 20 × `Q` (= 2.667) |
| Random seed | 5 |
| Final train / validation loss | 33.67 / 32.25 |

The loss weights in `experiment_params.py` have changed since this run. To reproduce it, first set these in `getLossParams`:

```python
Q_agent = torch.diag(torch.tensor([2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=device)) / 15
Q_final = Q * 20
```

Then run the following (a GPU is recommended; see `colab/train_bumpercar.ipynb`):

```bash
python experiments/bumpercar/run.py --horizon 300 --num-rollouts 400 --batch-size 100 --epochs 500 \
    --lr 2e-3 --alpha-col 1500 --alpha-obst 20000 --random-seed 5 --log-epoch 5
```

The checkpoint does not record the scenario (start poses, goal region, obstacles), the speed and deadzone parameters of the loss, or the P-controller gains. The values currently in `experiment_params.py` and `bumpercar_sys.py` are assumed to match, but this is not verified, so expect similar results rather than identical ones.

### Evaluate and visualise

```bash
python sim.py
```

`sim.py` is configured through the constants at the top of the file, not through CLI flags:

- `TRAINED_PBR_MODEL_PATH` is the checkpoint to load. The default is the provided `experiments/bumpercar/trained_pRB/rPB_best.pt`.
- `EVAL_*` set the number of rollouts, the horizon and the seed for `evaluate_controller()`. It prints the train/test loss, collision counts and obstacle hits.
- `SIM_*` choose the rollout that the interactive slider plot (`show_simulation()`) displays.
- `evaluate_rollout_metrics()` reports steady-state error and minimum car–car and car–obstacle distances. The `save_*_figure` and `save_trajectory_gif` functions produce the report figures in `experiments/bumpercar/figs/`.

## Dataset, obstacles and collisions

### Dataset (`plants/bumpercar/bumpercar_dataset.py`)

Each sample is one scenario, meaning a start state and a goal for both cars. It is stored as a tensor of shape `[horizon, 28]`:

- Columns `0:14`, row 0: the initial state `[x, y, θ, v_f, β_f, β_r, δ]` of car 1 and car 2. The later rows are zero.
- Columns `14:28`, every row: the goal state of both cars, constant over time. The rollout uses only the goal's `x, y` (as `xbar`). The tracking loss compares the full state, but `Q` puts weight only on `x, y`.

Each sample is drawn around the nominal values in `experiment_params.py` → `getCarInitParams`:

| | Car 1 | Car 2 | Randomisation |
|---|---|---|---|
| Start position | `(-2, -2.5)` | `(2, -2.5)` | + a point drawn uniformly in a disk of radius `car_init_radius = 0.3` m |
| Start heading | `0` | `-π` | + a uniform draw in `[0, STD_INIT_THETA)` = `[0°, 10°)`. The offset is always positive, despite the name. |
| Start velocities / steering | `0` | `0` | none |
| Goal position | `(2, 3)` | `(-2, 3)` | + a point drawn uniformly in a disk of radius `final_car_radius = 1.0` m |

`get_data()` generates 2048 samples for training and then 2048 more for testing. It keeps the first `--num-rollouts` of the first set as training data and the first 500 of the second set as test data. The generator is seeded with `--random-seed`, so the same seed and horizon always give the same data. Nothing is cached to disk.

Unused code: `getCarFinalParams` (a goal region `x ∈ [-2.5, 2.5]`, `y ∈ [2, 5]` with a 2 m minimum distance between the goals) and `generate_vector_with_min_distance`, which samples from that region, are passed in but never called. To sample goals from a region instead of around the nominal goals, call that function in `_generate_data`.

### Obstacles

The obstacles are **fixed**, not sampled. They are two Gaussians defined in `getCarInitParams`: centres `(-2, 0)` and `(2, 0)`, variance `0.44` per axis (std ≈ 0.66 m). They are not physical objects in the simulation and appear only in the loss, as a soft penalty:

```
loss_obst = alpha_obst · mean over time of  Σ_obstacles Σ_cars  N(p_car; centre, diag(cov))
```

(`BumpercarLoss.f_loss_obst`). This penalty is what the grey shading shows in the plots. For evaluation, `sim.py` counts an obstacle hit when a car comes within `OBSTACLE_RADIUS = 1.5` m of a centre.

### Collisions between the cars

The minimum distance is `min_dist = 2.0` m, set in `getLossParams`. The `--min-dist` CLI flag is **ignored**.

- **Loss** (`f_loss_ca`): `alpha_col · 1 / (d² + 1e-3)`, active when the distance between the cars satisfies `d < min_dist + 0.2`. It is averaged over time.
- **Collision count** (`count_collisions`, reported in `log` and by `sim.py`): the number of time steps with `d < min_dist`, summed over all rollouts.

### Other loss terms

These are set in `getLossParams` and `bumpercar_loss.py`:

- Tracking: `eᵀQe` averaged over time, plus a terminal term `eᵀQ_final e`. Each car's position error is shortened by `position_deadzone = 0.01` m, so errors below 1 cm cost nothing.
- Speed: `Qs · v_f²`, counted only once a car is within `steady_state_velocity_radius = 0.1` m of its goal, so the car is encouraged to stop there.
- Control effort: `alpha_u · ‖dxref‖²`.
- The arena-bounds term exists but is switched off (`alpha_bounds = 0`).

## Where to change things

| What | Where |
|---|---|
| Start poses, nominal goals, obstacles, spread of the initial position and heading | `experiment_params.py` → `getCarInitParams` |
| Spread of the goals (`final_car_radius`, default 1.0 m) | `plants/bumpercar/bumpercar_dataset.py` (not passed by `run.py`) |
| Loss weights `Q`, `Q_final`, speed weight, deadzone, collision distance | `experiment_params.py` → `getLossParams` (plus `--alpha-*` flags) |
| Car dynamics, P-controller gains, rollout loop | `plants/bumpercar/bumpercar_sys.py` |
| Controller structure (REN × MLP) | `controllers/PB_controller.py`, `controllers/contractive_ren.py`, `controllers/MLP.py` |
| Loss terms | `loss_functions/bumpercar_loss.py` (built on `lq_loss.py`) |

## Repository map

```
reference_tracking_PB/
├── config.py                  # device selection (cuda:0 if available), BASE_DIR
├── experiment_params.py       # scenario + loss parameters (shared by run.py and sim.py)
├── sim.py                     # evaluation, metrics, interactive plot, report figures/GIF
├── requirements.txt
├── colab/train_bumpercar.ipynb  # runs run.py on Colab with GPU
├── controllers/
│   ├── PB_controller.py       # PerfBoostController: disturbance reconstruction + REN × MLP
│   ├── contractive_ren.py     # ContractiveREN (contraction-constrained recurrent network)
│   └── MLP.py                 # MLP gain network, ZeroController (P-controller-only baseline)
├── plants/
│   ├── costum_dataset.py      # dataset base class (get_data: train/test split)
│   └── bumpercar/
│       ├── bumpercar_sys.py   # BumpercarSystem, PositionPidController, MLPDynamicsModel
│       ├── bumpercar_dataset.py  # samples initial states and goal references
│       ├── parameters.py      # physical car parameters
│       ├── utils.py           # angle helpers
│       └── model_kinematic_mlp.pth  # learned car dynamics (same file as in bumper_car_simulator/)
├── loss_functions/
│   ├── lq_loss.py             # finite-horizon LQ base class
│   └── bumpercar_loss.py      # tracking + collision + obstacle + bounds loss
├── utils/
│   ├── plot_functions.py      # trajectory plots and GIF helpers used by run.py
│   └── assistive_functions.py # to_tensor, WrapLogger
└── experiments/bumpercar/
    ├── run.py                 # TRAINING entry point
    ├── arg_parser.py          # CLI flags for run.py
    ├── trained_pRB/rPB_best.pt  # provided trained controller (full checkpoint)
    └── figs/                  # report figures of the provided controller
```

## Things to know

- `experiment_params.py` calls `argument_parser()` when it is imported. As a result, `sim.py` also accepts the same CLI flags as `run.py`, and flags such as `--alpha-col` change its evaluation loss.
- In `run.py`, the "validation" set is the training set. With `--return-best` (the default), the parameters with the best validation loss are restored before the final evaluation and the plots. The epoch checkpoints, however, store the parameters of each epoch.
- Deployment uses adapted copies of the plant and controller code in `../bumpercar-ros/bumpercar_control/bumpercar_control/` (`bumpercar_sys.py`, `rPB_controller.py`). These copies are **not** kept in sync automatically. If you change the structure here, update them too.
