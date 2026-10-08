import sys, os, logging, torch, time
from datetime import datetime
from torch.utils.data import DataLoader




BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, BASE_DIR)

from experiment_params import getCarInitParams, getFamilies, getLossParams, getExtraLossParams, SCENARIOS, OBSTACLES
from controllers.MLP import ZeroController
from config import device
from arg_parser import argument_parser, print_args
from plants import BumpercarDataset, BumpercarSystem, car_params
from utils.plot_functions import *
from controllers import PerfBoostController
from loss_functions import BumpercarLoss
from utils.assistive_functions import WrapLogger


def main():
    """
    Train and evaluate a performance boosting controller on the two-car bumpercar reference-tracking task.

    This script is intentionally self-contained: it generates synthetic rollouts/references,
    trains by backpropagating through closed-loop rollouts, and writes all artifacts (logs, plots,
    checkpoints) under `experiments/bumpercar/saved_results/` (or `--save-path`).
    """
    # ----- SET UP LOGGER / OUTPUT FOLDERS -----
    args = argument_parser()
    now = datetime.now().strftime("%m_%d_%H_%M_%S")
    save_path = args.save_path or os.path.join(BASE_DIR, 'experiments', 'bumpercar', 'saved_results')

    save_folder = os.path.join(save_path, 'perf_boost_' + now)
    save_folder_gif = os.path.join(save_folder, 'gifs')
    checkpoint_folder = os.path.join(save_folder, 'checkpoints')
    os.makedirs(save_folder, exist_ok=True)
    os.makedirs(save_folder_gif, exist_ok=True)
    os.makedirs(checkpoint_folder, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(save_folder, 'log'),
        format='%(asctime)s %(message)s',
        filemode='w',
    )
    logger = logging.getLogger('perf_boost_')
    logger.setLevel(logging.DEBUG)
    logger = WrapLogger(logger)

    # ----- parse and set experiment arguments -----
    
    msg = print_args(args)
    logger.info(msg)
    torch.manual_seed(args.random_seed)



    # ------------ 1. Dataset ------------
    # [x, y, theta, vf, beta_f, beta_r, delta] for each car
    n_agents = args.n_agents
    nx = 7 * n_agents
    x0_bumpercar, x_final_bumpercar, obstacle_centers, obstacle_covs = getCarInitParams(device)
    zones, families, sampling = getFamilies()

    dataset = BumpercarDataset(
        random_seed=args.random_seed,
        horizon=args.horizon,
        zones=zones,
        families=families,                      # from SCENARIOS (--families): one family drawn per sample
        sampling=sampling,
        n_agents=n_agents,
    )

# divide to train and test
    train_data, test_data = dataset.get_data(num_train_samples=args.num_rollouts, num_test_samples=500)
    train_data, test_data = train_data.to(device), test_data.to(device)


# data for plots
    t_ext = args.horizon * 4

    # nominal scenario (crossing up, no sampling): nominal start, nominal goal over the whole horizon
    plot_data = torch.zeros(1, args.horizon, 2 * nx)
    plot_data[:, 0, :nx] = x0_bumpercar
    plot_data[:, :, nx:] = x_final_bumpercar.view(1, 1, -1)
    plot_data = plot_data.to(device)

# batch the data
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

# ------------ 2. Plant ------------
    plant_input_init = None     # all zero
    plant_state_init = None    # same as xbar
    
    sys = BumpercarSystem(
        params=car_params,
        x_init=plant_state_init,
        u_init=plant_input_init,
        n_agents=n_agents,
        dt=0.04,
        turn_speed=args.turn_speed,
    ).to(device)

# ------------ 3. Controller ------------
    ctl = PerfBoostController(
    noiseless_forward=sys.noiseless_forward,
    input_init=sys.x_init,
    output_init=sys.u_init,
    dim_internal=args.dim_internal,
    dim_nl=args.dim_nl,
    initialization_std=args.cont_init_std,
    ren_internal_state_init=None,
    dt=sys.dt,
    ref_trace_tau=args.ren_ref_trace_tau,
    m2_type=args.m2,
    m2_gain_bound=args.m2_gain_bound,
).to(device)

    warm_optimizer_state = None   # Adam state of the loaded run (only if the whole controller was loaded)
    if args.load_controller is not None:
        ckpt = torch.load(args.load_controller, map_location=device)
        # load every tensor whose name and shape match; the rest keeps its initialization (e.g. a REN of another
        # size, or the other M2 type)
        current_state = ctl.state_dict()
        controller_state = {
            k: v for k, v in ckpt["controller_state_dict"].items()
            if k in current_state and current_state[k].shape == v.shape
        }
        skipped = sorted(set(ckpt["controller_state_dict"].keys()) - set(controller_state.keys()))
        ctl.load_state_dict(controller_state, strict=False)
        ctl.reset()
        logger.info(
            f"[INFO] loaded compatible controller tensors from {args.load_controller}; "
            f"skipped {len(skipped)} incompatible entries: {skipped}"
        )
        if len(skipped) == 0 and "optimizer_state_dict" in ckpt:
            warm_optimizer_state = ckpt["optimizer_state_dict"]


# ------------ 4. Loss ------------
    Q, Q_final, Qs, alpha_col, alpha_obst, alpha_u, position_deadzone, steady_state_velocity_radius, min_dist = getLossParams(device)
    loss_fn = BumpercarLoss(
        Q=Q,
        Q_final=Q_final,
        Qs=Qs,
        alpha_u=alpha_u,
        xbar=train_data[0, :, nx:],
        loss_bound=None,
        sat_bound=None,
        alpha_col=alpha_col,
        alpha_obst=alpha_obst,
        obstacle_centers=obstacle_centers,
        obstacle_covs=obstacle_covs,
        min_dist=min_dist,
        n_agents=n_agents,
        position_deadzone=position_deadzone,
        steady_state_velocity_radius=steady_state_velocity_radius,
        **getExtraLossParams(),
    )
 
# ------------ 5. Optimizer ------------
    valid_data = train_data      # use the entire train data for validation
    assert not (valid_data is None and args.return_best)
    optimizer = torch.optim.Adam(ctl.parameters(), lr=args.lr)
    if warm_optimizer_state is not None:
        # warm start: continue Adam's state of the loaded run (moment estimates, step count), so the first steps do not
        # kick the controller out of its minimum. load_state_dict also restores the loaded run's learning rate -> set
        # --lr explicitly (initial_lr: start of the schedule below)
        optimizer.load_state_dict(warm_optimizer_state)
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] = args.lr
        logger.info(f"[INFO] continued the optimizer state of the loaded run with learning rate {args.lr:.2e}")

    # learning-rate schedule over the epochs: constant, or decaying to lr * lr_end_factor at the last epoch
    lr_end = args.lr * args.lr_end_factor
    if args.lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=lr_end)
    elif args.lr_schedule == "linear":
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=args.lr_end_factor, total_iters=args.epochs,
        )
    else:
        scheduler = None
    # gradient clipping: max gradient norm per step (inf = no clipping, the norm is still measured and logged)
    grad_clip = args.grad_clip if args.grad_clip is not None else float("inf")

# ------------ 6. Training ------------
# closed-loop trajectory before training, with the P-controller only (no rPB correction) on the same sample overlaid
    logger.info('Plotting closed-loop trajectory before training the controller (with the P-controller only)...')
    pid_only_ctl = ZeroController(ref_dim=2 * n_agents).to(device)
    with torch.no_grad():
        x_pid, _, _ = sys.rollout(pid_only_ctl, plot_data[:1])
        x_log, _, u_log = sys.rollout(ctl, plot_data[:1])
    plot_closed_loop(
        x_log[0], plot_data[0, :, nx:], save_folder, 'CL_untrained.pdf',
        'rPB - untrained controller vs P-controller only', loss_fn, t_ext, x_baseline=x_pid[0],
    )

    logger.info('\n------------ Begin training ------------')
    best_valid_loss = 1e6
    best_epoch = 0   # epoch of the best validation loss (early stopping)
    assert args.patience_epoch is None or args.return_best, "early stopping needs --return-best (validation loss)"
    # loss history at the logged epochs, for loss_curves.pdf
    history = {"epoch": [], "train": [], "valid": [], "valid_terms": {}}
    t = time.time()
    for epoch in range(1 + args.epochs):
        max_grad_norm = 0.0   # largest gradient norm (before clipping) in this epoch
        train_loss_sum, train_batches = 0.0, 0   # train loss of the epoch: mean over its batches
        # iterate over all data batches
        for train_data_batch in train_dataloader:
            optimizer.zero_grad()
            # simulate over horizon steps
            x_log, e_log, u_log = sys.rollout(
                controller=ctl, data=train_data_batch, train=True,
            )
            # loss of this rollout
            loss = loss_fn.forward(x_log, u_log, e_log)
            # take a step
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(ctl.parameters(), grad_clip)   # returns the norm before clipping
            max_grad_norm = max(max_grad_norm, grad_norm.item())
            optimizer.step()
            train_loss_sum += loss.detach().item()
            train_batches += 1
        train_loss = train_loss_sum / train_batches

        # print info
        if epoch % args.log_epoch == 0:
            msg = 'Epoch: %i --- train loss (mean over the epoch): %.2f' % (epoch, train_loss)
            msg += ' ---||--- lr: %.2e ---||--- max grad norm: %.3g' % (optimizer.param_groups[0]['lr'], max_grad_norm)
            loss_valid_value = None
            is_best = False

            if args.return_best:
                # rollout the current controller on the valid data
                with torch.no_grad():
                    x_log_valid, e_log_valid, u_log_valid = sys.rollout(
                        controller=ctl, data=valid_data, train=False,
                    )
                    # loss of the valid data
                    loss_valid = loss_fn.forward(x_log_valid, u_log_valid, e_log_valid)
                loss_valid_value = loss_valid.item()
                msg += ' ---||--- validation loss: %.2f' % loss_valid_value
                msg += ' [' + ', '.join('%s %.2f' % (name, value) for name, value in loss_fn.last_terms.items()) + ']'
                valid_terms = dict(loss_fn.last_terms)
                # compare with the best valid loss
                if loss_valid_value < best_valid_loss:
                    best_valid_loss = loss_valid_value
                    best_params_ren = ctl.get_parameters_as_vector()  # record state dict if best on valid
                    best_params_m2 = ctl.get_m2_parameters()
                    is_best = True
                    best_epoch = epoch
                    msg += ' (best so far)'
            checkpoint = {
                "epoch": epoch,
                "controller_state_dict": ctl.state_dict(),
                "ren_state_dict": ctl.c_ren.state_dict(),
                "m2_state_dict": ctl.M2.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "Q": Q,
                "Q_final": Q_final,
                "args": vars(args),
                "scenario": {**SCENARIOS[n_agents], "obstacles": OBSTACLES},   # zones, families, sampling, obstacles of this run
                "train_loss": train_loss,
                "validation_loss": loss_valid_value,
                "best_valid_loss": best_valid_loss if args.return_best else None,
            }
            checkpoint_epoch_file = os.path.join(checkpoint_folder, f"checkpoint_epoch_{epoch:05d}.pt")
            checkpoint_latest_file = os.path.join(checkpoint_folder, "checkpoint_latest.pt")
            torch.save(checkpoint, checkpoint_epoch_file)
            torch.save(checkpoint, checkpoint_latest_file)
            if is_best:   # full controller with the lowest validation loss so far
                torch.save(checkpoint, os.path.join(checkpoint_folder, "best.pt"))
            duration = time.time() - t
            msg += ' ---||--- time: %.0f s' % (duration)
            msg += f' ---||--- checkpoint: {checkpoint_epoch_file}'
            logger.info(msg)

            # loss curves so far (redrawn at every logged epoch, so a stopped run has them too)
            history["epoch"].append(epoch)
            history["train"].append(train_loss)
            if loss_valid_value is not None:
                history["valid"].append(loss_valid_value)
                for name, value in valid_terms.items():
                    history["valid_terms"].setdefault(name, []).append(value)
            plot_loss_curves(history, save_folder, 'loss_curves.pdf')
            t = time.time()

            # early stopping: no validation improvement for patience_epoch epochs
            if args.patience_epoch is not None and epoch - best_epoch >= args.patience_epoch:
                logger.info('[INFO] early stopping at epoch %i: no validation improvement since epoch %i.' % (epoch, best_epoch))
                break

        if scheduler is not None:
            scheduler.step()   # once per epoch

    # set to best seen during training
    if args.return_best:
        ctl.set_parameters_as_vector(best_params_ren)
        ctl.set_m2_parameters(best_params_m2)

# ------ 7. Save and evaluate the trained model ------
# save
    res_dict = ctl.c_ren.state_dict()
    # minimal metadata to make the checkpoint self-describing
    res_dict['Q'] = Q
    res_dict['args'] = vars(args)
    filename = os.path.join(save_folder, 'trained_controller' + '.pt')
    torch.save(res_dict, filename)
    logger.info('[INFO] saved trained model.')

# evaluate on the train data
    logger.info('\n[INFO] evaluating the trained controller on %i training rollouts.' % train_data.shape[0])
    with torch.no_grad():
        x_log, e_log, u_log = sys.rollout(
            controller=ctl, data=train_data, train=False,
        )   # use the entire train data, not a batch
        # evaluate losses
        loss = loss_fn.forward(x_log, u_log, e_log)
        msg = 'Loss: %.4f' % (loss.item())
# count collisions
    if args.col_av:
        num_col = loss_fn.count_collisions(x_log)
        msg += ' -- Number of collisions = %i' % num_col
    logger.info(msg)

# evaluate on the test data
    logger.info('\n[INFO] evaluating the trained controller on %i test rollouts.' % test_data.shape[0])
    with torch.no_grad():
        # simulate over horizon steps
        x_log, e_log, u_log = sys.rollout(
            controller=ctl, data=test_data, train=False,
        )
        # loss
        test_loss = loss_fn.forward(x_log, u_log, e_log).item()
        msg = "Loss: %.4f" % (test_loss)
# count collisions
    if args.col_av:
        num_col = loss_fn.count_collisions(x_log)
        msg += ' -- Number of collisions = %i' % num_col
    logger.info(msg)


# plot closed-loop trajectory and controller output using the trained controller
    logger.info('Plotting closed-loop trajectory using the trained controller...')
    with torch.no_grad():
        x_log, _, u_log = sys.rollout(ctl, plot_data[:1])
    plot_closed_loop(
        x_log[0], plot_data[0, :, nx:], save_folder, 'CL_trained.pdf',
        'rPB - trained controller', loss_fn, t_ext,
    )
    plot_dxref_over_time(u_log[0], sys.dt, n_agents, save_folder, 'U_over_time.pdf')


def plot_closed_loop(x, ref, save_folder, filename, text, loss_fn, T, x_baseline=None):
    """Plot one closed-loop rollout x [T, 7*n_agents] with its route (reference over time ref [T, 7*n_agents]:
    every goal as a numbered star), the obstacles and optionally a baseline rollout on the same sample."""
    earlier_goals, goal = route_points(ref)
    plot_trajectories(
        x, xbar=goal, n_agents=loss_fn.n_agents, save_folder=save_folder,
        filename=filename, text=text, T=T,
        obstacle_centers=loss_fn.obstacle_centers, obstacle_covs=loss_fn.obstacle_covs,
        earlier_goals=earlier_goals, x_baseline=x_baseline,
    )


def plot_loss_curves(history, save_folder, filename):
    """Loss over the logged epochs: totals (train = mean over the epoch's batches, validation) and every validation loss term."""
    epochs = history["epoch"]
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    axs[0].plot(epochs, history["train"], label="train (mean over the epoch's batches)", alpha=0.6)
    if history["valid"]:
        axs[0].plot(epochs, history["valid"], label="validation (train data)")
    axs[0].set_title("total loss")
    for name, values in history["valid_terms"].items():
        if max(values) <= 1e-3:
            continue   # terms that are off or ~0 (log scale)
        axs[1].plot(epochs, values, label=name)
    axs[1].set_title("validation loss terms")
    axs[1].legend(fontsize=8, ncol=2)
    for ax in axs:
        ax.set_yscale("log")
        ax.set_xlabel("epoch")
        ax.grid(True, which="both", alpha=0.3)
    axs[0].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(save_folder, filename))
    plt.close(fig)


def plot_dxref_over_time(u, dt, n_agents, save_folder, filename):
    """Plot the rPB output dxref = (dX, dY) of each car over time for one rollout u [T, 2*n_agents]."""
    u = u.detach().cpu()
    t = torch.arange(u.shape[0]) * dt
    fig, axs = plt.subplots(n_agents, 1, figsize=(10, 3.5 * n_agents), sharex=True)
    for i in range(n_agents):
        axs[i].plot(t, u[:, 2 * i], label="dX")
        axs[i].plot(t, u[:, 2 * i + 1], label="dY")
        axs[i].set_title(f"Car {i + 1}")
        axs[i].set_ylabel("Reference offset [m]")
        axs[i].legend()
        axs[i].grid()
    axs[-1].set_xlabel("Time [s]")
    fig.suptitle("rPB offset to the reference over time (trained controller)")
    fig.tight_layout()
    fig.savefig(os.path.join(save_folder, filename))
    plt.close(fig)


if __name__ == "__main__":
    main()
