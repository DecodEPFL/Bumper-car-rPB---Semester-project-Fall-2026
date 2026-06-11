import sys, os, logging, torch, time
from datetime import datetime
from torch.utils.data import DataLoader




BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, BASE_DIR)

from experiment_params import getCarFinalParams, getCarInitParams, getLossParams
from controllers.MLP import ZeroController
from config import device
from arg_parser import argument_parser, print_args
from plants import RobotsSystem, RobotsDataset, BumpercarDataset, BumpercarSystem, car_params
from utils.plot_functions import *
from controllers import PerfBoostController
from loss_functions import RobotsLoss, BumpercarLoss
from utils.assistive_functions import WrapLogger


def main():
    """
    Train and evaluate a performance boosting controller on the robots reference-tracking task.

    This script is intentionally self-contained: it generates synthetic rollouts/references,
    trains by backpropagating through closed-loop rollouts, and writes all artifacts (logs, plots,
    checkpoints) under `experiments/robots/saved_results/`.
    """
    # ----- SET UP LOGGER / OUTPUT FOLDERS -----
    args = argument_parser()
    now = datetime.now().strftime("%m_%d_%H_%M_%S")
    save_path = args.save_path or os.path.join(BASE_DIR, 'experiments', 'robots', 'saved_results')

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
    x0_bumpercar, x_final_bumpercar, obstacle_centers, obstacle_covs, car_init_radius, std_init_theta = getCarInitParams(device)
    x_final_limit, y_final_limit, final_car_min_dist = getCarFinalParams()
    
    dataset = BumpercarDataset(
        random_seed=args.random_seed,
        horizon=args.horizon,
        x0=x0_bumpercar,
        x_final=x_final_bumpercar,
        car_init_radius=car_init_radius,
        x_final_limit=x_final_limit,
        y_final_limit=y_final_limit,
        final_car_min_dist=final_car_min_dist,
        std_init_theta = std_init_theta,
        n_agents=2,
    )
    
# divide to train and test
    train_data, test_data = dataset.get_data(num_train_samples=args.num_rollouts, num_test_samples=500)
    train_data, test_data = train_data.to(device), test_data.to(device)


# data for plots
    t_ext = args.horizon * 4
    n_agents = 2
    nx = 7 * n_agents

    plot_data = test_data[250:350, :, :]
    plot_data[:, 0, :nx] = dataset.x0.detach()
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
        dt=0.04
    ).to(device)

# ------------ 3. Controller ------------
    ctl = PerfBoostController(
    noiseless_forward=sys.noiseless_forward,
    input_init=sys.x_init,
    output_init=sys.u_init,
    dim_internal=args.dim_internal,
    dim_nl=args.dim_nl,
    initialization_std=args.cont_init_std,
    output_amplification=1,
    ren_internal_state_init=None,
).to(device)

    if args.load_controller is not None:
        ckpt = torch.load(args.load_controller, map_location=device)
        if "controller_state_dict" in ckpt:
            current_state = ctl.state_dict()
            controller_state = {
                k: v for k, v in ckpt["controller_state_dict"].items()
                if k in current_state and current_state[k].shape == v.shape
            }
            skipped = sorted(set(ckpt["controller_state_dict"].keys()) - set(controller_state.keys()))
            ctl.load_state_dict(controller_state, strict=False)
        elif "ren_state_dict" in ckpt:
            current_state = ctl.c_ren.state_dict()
            ren_state = {
                k: v for k, v in ckpt["ren_state_dict"].items()
                if k in current_state and current_state[k].shape == v.shape
            }
            skipped = sorted(set(ckpt["ren_state_dict"].keys()) - set(ren_state.keys()))
            ctl.c_ren.load_state_dict(ren_state, strict=False)
            if "mlp_state_dict" in ckpt:
                ctl.MLP.load_state_dict(ckpt["mlp_state_dict"], strict=False)
        else:
            current_state = ctl.c_ren.state_dict()
            ren_state = {
                k: v for k, v in ckpt.items()
                if k in current_state and current_state[k].shape == v.shape
            }
            skipped = sorted(set(ckpt.keys()) - set(ren_state.keys()))
            ctl.c_ren.load_state_dict(ren_state, strict=False)
        ctl.reset()
        logger.info(
            f"[INFO] loaded compatible controller tensors from {args.load_controller}; "
            f"skipped {len(skipped)} incompatible/non-REN entries."
        )


# ------------ 4. Loss ------------
    Q, Q_final, Qs, alpha_col, alpha_obst, alpha_u, position_deadzone, steady_state_velocity_radius, min_dist = getLossParams(device)
    loss_fn = BumpercarLoss(
        Q=Q,
        Q_final=Q_final,
        Qs=Qs,
        alpha_u=alpha_u,
        xbar=train_data[0, :, 14:],
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
    )
 
# ------------ 5. Optimizer ------------
    valid_data = train_data      # use the entire train data for validation
    assert not (valid_data is None and args.return_best)
    optimizer = torch.optim.Adam(ctl.parameters(), lr=args.lr)
 
# ------------ 6. Training ------------
# plot PID without untrained rPB
# ------------ PID-only verification ------------
    logger.info("Plotting closed-loop trajectories with PID only...")

# plot closed-loop trajectories before training the controller
    logger.info('Plotting closed-loop trajectories before training the controller...')
    x_log, _, u_log = sys.rollout(ctl, plot_data)


    nx = 7 * n_agents
    
    data_verif = torch.zeros(3, args.horizon + 200, 2 * nx)
    data_verif[:, 0:1, :nx] = dataset.x0.view(1, 1, -1)
    data_verif[0:1, :, nx:] = x0_bumpercar.view(1, 1, -1)
    data_verif[1:2, :, nx:] = x0_bumpercar.view(1, 1, -1)
    data_verif[2:3, :, nx:] = x0_bumpercar.view(1, 1, -1)
    data_verif = data_verif.to(device)
    pid_only_ctl = ZeroController(ref_dim=2 * n_agents).to(device)
    x_verif, _, u_verif = sys.rollout(pid_only_ctl, data_verif)
    
    plot_trajectories(
        x_verif[0, :, :],
        xbar=x0_bumpercar,
        n_agents=sys.n_agents,
        save_folder=save_folder,
        filename='Only PID.pdf',
        text="Not ONLY PID",
        T=t_ext,
        obstacle_centers=loss_fn.obstacle_centers,
        obstacle_covs=loss_fn.obstacle_covs,
    )
    
 
    logger.info('\n------------ Begin training ------------')
    best_valid_loss = 1e6
    t = time.time()
    for epoch in range(1 + args.epochs):
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
            optimizer.step()

        # print info
        if epoch % args.log_epoch == 0:
            msg = 'Epoch: %i --- train loss: %.2f' % (epoch, loss.detach().item())
            loss_valid_value = None

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
                # compare with the best valid loss
                if loss_valid_value < best_valid_loss:
                    best_valid_loss = loss_valid_value
                    best_params_ren = ctl.get_parameters_as_vector()  # record state dict if best on valid
                    best_params_mlp = ctl.get_mlp_parameters()
                    msg += ' (best so far)'
            checkpoint = {
                "epoch": epoch,
                "controller_state_dict": ctl.state_dict(),
                "ren_state_dict": ctl.c_ren.state_dict(),
                "mlp_state_dict": ctl.MLP.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "Q": Q,
                "Q_final": Q_final,
                "args": vars(args),
                "train_loss": loss.detach().item(),
                "validation_loss": loss_valid_value,
                "best_valid_loss": best_valid_loss if args.return_best else None,
            }
            checkpoint_epoch_file = os.path.join(checkpoint_folder, f"checkpoint_epoch_{epoch:05d}.pt")
            checkpoint_latest_file = os.path.join(checkpoint_folder, "checkpoint_latest.pt")
            torch.save(checkpoint, checkpoint_epoch_file)
            torch.save(checkpoint, checkpoint_latest_file)
            duration = time.time() - t
            msg += ' ---||--- time: %.0f s' % (duration)
            msg += f' ---||--- checkpoint: {checkpoint_epoch_file}'
            logger.info(msg)
            t = time.time()

    # set to best seen during training
    if args.return_best:
        ctl.set_parameters_as_vector(best_params_ren)
        ctl.set_mlp_parameters(best_params_mlp)

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


#     # plot closed-loop trajectories using the trained controller
#     logger.info('Plotting closed-loop trajectories using the trained controller...')
#     x_log, _, u_log = sys.rollout(ctl, plot_data)
#     plot_trajectories(
#         x_log[0, :, :],  # remove extra dim due to batching
#         xbar=plot_data[0, min(5, plot_data.shape[1] - 1), nx:],
#         n_agents=sys.n_agents,
#         save_folder=save_folder,
#         filename='CL_trained.pdf',
#         text="CL - trained controller",
#         T=t_ext,
#         obstacle_centers=loss_fn.obstacle_centers,
#         obstacle_covs=loss_fn.obstacle_covs,
#     )

#     x_verif, _, u_verif = sys.rollout(ctl, data_verif)
#     v_verif = sys.v_log
#     plot_trajectories(
#         x_verif[0, :, :],  # remove extra dim due to batching
#         xbar=xbar_train,
#         n_agents=sys.n_agents,
#         save_folder=save_folder,
#         filename='CL_diag_trained.pdf',
#         text="rPB - trained controller",
#         T=t_ext,
#         obstacle_centers=loss_fn.obstacle_centers,
#         obstacle_covs=loss_fn.obstacle_covs,
#     )

#     plot_trajectories(
#         x_verif[1, :, :],  # remove extra dim due to batching
#         xbar=xbar_verif2,
#         n_agents=sys.n_agents,
#         save_folder=save_folder,
#         filename='CL_direct_trained.pdf',
#         text="rPB - trained controller",
#         T=t_ext,
#         obstacle_centers=loss_fn.obstacle_centers,
#         obstacle_covs=loss_fn.obstacle_covs,
#     )

#     plot_trajectories(
#         x_verif[2, :, :],  # remove extra dim due to batching
#         xbar=xbar_verif3,
#         n_agents=sys.n_agents,
#         save_folder=save_folder,
#         filename='CL_center_trained.pdf',
#         text="CL - trained controller",
#         T=t_ext,
#         obstacle_centers=loss_fn.obstacle_centers,
#         obstacle_covs=loss_fn.obstacle_covs,
#     )



# #### Plot the evolution of the reference over time for the diagonal scenario ####
#     x_ref_evol = torch.zeros(1, args.horizon + 200, 14)
#     x_ref_evol[:, :, 0:2] = u_verif[0:1, :, 0:2]
#     x_ref_evol[:, :, 4:6] = u_verif[0:1, :, 2:4]
#     x_ref_evol = x_ref_evol + xbar_train



#     plot_trajectories(
#         x_ref_evol[0, :, :],  # remove extra dim due to batching
#         xbar=xbar_train,
#         n_agents=sys.n_agents,
#         save_folder=save_folder,
#         filename='CL_xbar_evolution.pdf',
#         text="CL - evolution of the reference",
#         T=t_ext,
#         dots=True,
#         obstacle_centers=loss_fn.obstacle_centers,
#         obstacle_covs=loss_fn.obstacle_covs,
#     )


# # Create a figure with a 2x2 grid of subplots
#     fig, axs = plt.subplots(2, 1, figsize=(10, 7))
#     axs[0].plot(np.array(range(u_verif.shape[1])), u_verif[2, :, 0], label="dX")
#     axs[0].plot(np.array(range(u_verif.shape[1])), u_verif[2, :, 1], label="dY")
#     axs[0].set_title("Robot 1")
#     axs[0].set_xlabel("Time (s)")
#     axs[0].set_ylabel("Delta ref")
#     axs[0].legend()
#     axs[0].grid()

#     axs[1].plot(np.array(range(u_verif.shape[1])), u_verif[2, :, 2], label="dX")
#     axs[1].plot(np.array(range(u_verif.shape[1])), u_verif[2, :, 3], label="dY")
#     axs[1].set_title("Robot 2")
#     axs[1].set_xlabel("Time (s)")
#     axs[1].set_ylabel("Delta ref")
#     axs[1].legend()
#     axs[1].grid()

#     # Adjust layout to prevent overlap
#     plt.tight_layout()
#     plt.subplots_adjust(top=0.9)  # Adjust the top space to make room for the suptitle

#     plt.suptitle(
#         'Performance boosting offset to the reference over time \n for the diagonal scenario',
#         fontsize=13,
#     )
#     plt.savefig(os.path.join(save_folder, "U_over_time.pdf"))
#     plt.close()


# # Create a figure with a 2x2 grid of subplots
#     fig, axs = plt.subplots(2, 1, figsize=(10, 7))
#     axs[0].plot(np.array(range(u_verif.shape[1])), v_verif[0, :, 0], label="v_X")
#     axs[0].plot(np.array(range(u_verif.shape[1])), v_verif[0, :, 1], label="v_Y")
#     axs[0].set_title("Robot 1")
#     axs[0].set_xlabel("Time (s)")
#     axs[0].set_ylabel("v")
#     axs[0].legend()
#     axs[0].grid()

#     axs[1].plot(np.array(range(u_verif.shape[1])), v_verif[0, :, 2], label="v_X")
#     axs[1].plot(np.array(range(u_verif.shape[1])), v_verif[0, :, 3], label="v_Y")
#     axs[1].set_title("Robot 2")
#     axs[1].set_xlabel("Time (s)")
#     axs[1].set_ylabel("v")
#     axs[1].legend()
#     axs[1].grid()

#     # Adjust layout to prevent overlap
#     plt.tight_layout()
#     plt.subplots_adjust(top=0.9)  # Adjust the top space to make room for the suptitle

#     plt.suptitle('Integral variable over time \n for the diagonal scenario', fontsize=13)
#     plt.savefig(os.path.join(save_folder, "V_over_time.pdf"))
#     plt.close()


if __name__ == "__main__":
    main()
