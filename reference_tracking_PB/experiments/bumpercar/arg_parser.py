import argparse


# argument parser
def argument_parser():
    parser = argparse.ArgumentParser(description="Bumpercar rPB training.")

    # experiment
    parser.add_argument('--random-seed', type=int, default=7, help='Random seed. Default is 7 (escapes the initial colliding regime reliably with the default controller).')
    parser.add_argument('--col-av', action=argparse.BooleanOptionalAction, default=True, help='Avoid collisions. Default is True.')
    parser.add_argument('--obst-av', action=argparse.BooleanOptionalAction, default=True, help='Avoid obstacles. Default is True.')
    parser.add_argument('--save-path', type=str, default=None, help='Optional output directory for saved results and checkpoints.')
    # dataset
    parser.add_argument('--horizon', type=int, default=400, help='Time horizon in steps (dt = 0.04 s). Default is 400 (16 s).')
    parser.add_argument('--n-agents', type=int, default=2, help='Number of agents. Default is 2.')
    parser.add_argument('--num-rollouts', type=int, default=400, help='Number of rollouts in the training data. Default is 400.')
    parser.add_argument('--goal-heading-range', type=float, default=0.0, help='Target heading per car sampled uniformly within +-this many degrees around the scenario heading. Default is 0 (off: exactly the SCENARIOS heading).')
    parser.add_argument('--start-radius', type=float, default=0.3, help='Start position per car sampled uniformly in a disk of this radius [m] around the scenario start. Default is 0.3.')
    parser.add_argument('--start-heading-range', type=float, default=10.0, help='Start heading per car sampled uniformly within +-this many degrees around the scenario heading. Default is 10.')

    # controller
    parser.add_argument('--cont-init-std', type=float, default=0.1, help='Initialization std for controller params. Default is 0.1.')
    parser.add_argument('--dim-internal', type=int, default=8, help='Dimension of the internal state of the controller. Adjusts the size of the linear part of REN. Default is 8.')
    parser.add_argument('--dim-nl', type=int, default=8, help='size of the non-linear part of REN. Default is 8.')
    parser.add_argument('--ren-clock-tau', type=float, default=None, help='Time constant [s] of an extra REN input exp(-t / tau), t = time since the last reference change. Lets the REN act after w_hat has vanished. Default is None (off).')
    parser.add_argument('--ren-ref-trace-tau', type=float, default=5.0, help='Time constant [s] of the reference-change trace fed to the REN: it jumps by the change (delta goal x, y and delta heading as sin, 1 - cos) and decays from that value. 0 = only the raw change; a negative value turns the input off. Default is 3.')

    # loss
    parser.add_argument('--alpha-u', type=float, default=0.1/400, help='Weight of the loss due to control input "u". Default is 0.1/400.')  #TODO: 400 is output_amplification^2
    parser.add_argument('--alpha-col', type=float, default=1500, help='Weight of the collision avoidance loss. Default is 1500 if "col-av" is True, else None.')
    parser.add_argument('--alpha-obst', type=float, default=20000, help='Weight of the obstacle avoidance loss. Default is 20000 if "obst-av" is True, else None.')
    parser.add_argument('--obst-safe-dist', type=float, default=1.7, help='Obstacle cost is 0 beyond this distance [m] from each obstacle center (Gaussian minus its value there, clipped at 0). Default is 1.7 (sim.py hit radius 1.5 + 0.2 margin).')
    parser.add_argument('--min-dist', type=float, default=2.0, help='Minimum distance [m] between car centers; the collision loss is active below min_dist + 0.2. Default is 2.0 if "col-av" is True, else None.')
    parser.add_argument('--q-final-factor', type=float, default=1500, help='Terminal position weight: Q_final = this * Q (per car (2/15) * factor per m^2), averaged over the last --heading-final-steps steps of every goal leg. Default is 1500 (a car stuck 15 cm short costs ~4.5 per car).')
    parser.add_argument('--alpha-heading', type=float, default=50.0, help='Weight of the running heading loss 1 - cos(theta* - theta), weighted by exp(-d^2 / (2 * heading_radius^2)) with d the distance to the goal. Default is 50 (0 = off).')
    parser.add_argument('--alpha-heading-final', type=float, default=50.0, help='Weight of the terminal heading loss over the last --heading-final-steps steps. Default is 50 (0 = off).')
    parser.add_argument('--heading-radius', type=float, default=2.5, help='Length scale [m] of the soft mask of the running heading loss (weight 0.61 at this distance from the goal). Default is 2.5.')
    parser.add_argument('--alpha-offset-final', type=float, default=20.0, help='Weight of the squared rPB offset dxref over the last --heading-final-steps steps (the car must have settled by the end). Default is 20 (0 = off).')
    parser.add_argument('--heading-final-steps', type=int, default=50, help='Number of final time steps of every goal leg averaged in the terminal position and heading losses. Default is 50 (2 s at dt 0.04).')

    # optimizer
    parser.add_argument('--batch-size', type=int, default=100, help='Number of forward trajectories of the closed-loop system at each step (-1 = all training rollouts). Default is 100.')
    parser.add_argument('--epochs', type=int, default=200, help='Total number of epochs for training. Default is 200.')
    parser.add_argument('--lr', type=float, default=2e-3, help='Initial learning rate. Default is 2e-3.')
    parser.add_argument('--log-epoch', type=int, default=5, help='Frequency of logging, validation and checkpoints in epochs. Default is 5.')
    parser.add_argument('--lr-schedule', type=str, default='cosine', choices=['constant', 'linear', 'cosine'], help='Learning-rate schedule over the epochs. Default is cosine.')
    parser.add_argument('--lr-end-factor', type=float, default=0.05, help='Learning rate at the last epoch as a fraction of --lr (linear/cosine schedules). Default is 0.05.')
    parser.add_argument('--grad-clip', type=float, default=3000, help='Clip the gradient norm of each step to this value (--grad-clip inf = no clipping; the norm is always logged). Default is 3000.')
    parser.add_argument('--return-best', action=argparse.BooleanOptionalAction, default=True, help='Return the best model on the validation data among all logged iterations. The train data can be used instead of validation data. The Default is True.')
    parser.add_argument('--load-controller', type=str, default=None, help='Optional path to a compatible trained controller checkpoint.')
    parser.add_argument('--checkpoint', type=str, default=None, help='sim.py only: checkpoint to evaluate. Default is the provided trained_pRB/rPB_best.pt.')

    # TODO: add the following
    # parser.add_argument('--patience-epoch', type=int, default=None, help='Patience epochs for no progress. Default is None which sets it to 0.2 * total_epochs.')
    # parser.add_argument('--lr-start-factor', type=float, default=1.0, help='Start factor of the linear learning rate scheduler. Default is 1.0.')
    # parser.add_argument('--lr-end-factor', type=float, default=0.01, help='End factor of the linear learning rate scheduler. Default is 0.01.')
    # # save/load args
    # parser.add_argument('--experiment-dir', type=str, default='boards', help='Name tag for the experiments. By default it will be the "boards" folder.')
    # parser.add_argument('--load-model', type=str, default=None, help='If it is not set to None, a pretrained model will be loaded instead of training.')
    # parser.add_argument('--device', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help='Device to run the computations on, "cpu" or "cuda:0". Default is "cuda:0" if available, otherwise "cpu".')

    args = parser.parse_args()

    # set default values that depend on other args
    if args.batch_size == -1:
        args.batch_size = args.num_rollouts  # use all train data

    if args.ren_ref_trace_tau is not None and args.ren_ref_trace_tau < 0:
        args.ren_ref_trace_tau = None   # reference-change input off

    # assertions and warning
    if not args.col_av:
        args.alpha_col = None
        args.min_dist = None
    if not args.obst_av:
        args.alpha_obst = None

    # if args.total_epochs < 10000:
    #     print(f'Minimum of 10000 epochs are required for proper training')

    return args


def print_args(args):
    msg = '\n[INFO] Dataset: n_agents: %i' % args.n_agents + ' -- num_rollouts: %i' % args.num_rollouts
    msg += ' -- start radius: %g m -- start heading range: +-%g deg' % (args.start_radius, args.start_heading_range)
    msg += ' -- time horizon: %i' % args.horizon
    msg += ' -- goal heading range: +-%g deg' % args.goal_heading_range

    msg += '\n[INFO] Controller: dimension of the internal state: %i' % args.dim_internal
    msg += ' -- dim_nl: %i' % args.dim_nl + ' -- cont_init_std: %.2f'% args.cont_init_std
    msg += ' -- REN clock tau: %s s -- REN reference trace tau: %s s' % (args.ren_clock_tau, args.ren_ref_trace_tau)

    msg += '\n[INFO] Loss:  alpha_u: %.6f' % args.alpha_u + ' -- q_final_factor: %g' % args.q_final_factor
    msg += ' -- alpha_col: %.f -- min_dist: %g' % (args.alpha_col, args.min_dist) if args.col_av else ' -- no collision avoidance'
    msg += ' -- alpha_obst: %.1f -- obst_safe_dist: %s' % (args.alpha_obst, args.obst_safe_dist) if args.obst_av else ' -- no obstacle avoidance'
    msg += ' -- alpha_heading: %g -- alpha_heading_final: %g -- heading_radius: %g -- heading_final_steps: %i -- alpha_offset_final: %g' % (
        args.alpha_heading, args.alpha_heading_final, args.heading_radius, args.heading_final_steps, args.alpha_offset_final)

    msg += '\n[INFO] Optimizer: lr: %.2e' % args.lr
    msg += ' -- lr schedule: %s (end factor %g) -- grad clip: %s' % (args.lr_schedule, args.lr_end_factor, args.grad_clip)
    msg += ' -- batch_size: %i' % args.batch_size + ', -- return best model for validation data among logged epochs:' + str(args.return_best)

    return msg
