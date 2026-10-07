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
    parser.add_argument('--turn-speed', type=float, default=0.6, help='Base P-controller: speed command [m/s] while a car moves away from its target (e.g. a new goal behind it), as min(kp * distance, turn_speed). Default is None (constant 0.1 m/s, the original rule).')
    parser.add_argument('--num-rollouts', type=int, default=800, help='Number of rollouts in the training data. Default is 400.')
    parser.add_argument('--goal-heading-range', type=float, default=20.0, help='Target heading per car sampled uniformly within +-this many degrees around the scenario heading. Default is 0 (off: exactly the SCENARIOS heading).')
    parser.add_argument('--start-radius', type=float, default=0.5, help='Start position per car sampled uniformly in a disk of this radius [m] around the scenario start. Default is 0.3.')
    parser.add_argument('--start-heading-range', type=float, default=30.0, help='Start heading per car sampled uniformly within +-this many degrees around the scenario heading. Default is 10.')

    # controller
    parser.add_argument('--cont-init-std', type=float, default=0.1, help='Initialization std for controller params. Default is 0.1.')
    parser.add_argument('--dim-internal', type=int, default=8, help='Dimension of the internal state of the REN (per car). Adjusts the size of the linear part of REN. Default is 8.')
    parser.add_argument('--dim-nl', type=int, default=8, help='size of the non-linear part of REN (per car). Default is 8.')
    parser.add_argument('--m2', type=str, default='gru', choices=['mlp', 'gru'], help='M2 of the rPB, per car with shared weights, on the car\'s own w_hat, goal, target heading and current state (goal relative to the car, heading error, nearest other car), output bounded by --m2-gain-bound: "gru" (with memory) or "mlp" (without memory). Default is gru.')
    parser.add_argument('--m2-gain-bound', type=float, default=5.0, help='M2: bound c of the output gain, c * tanh(.) per component. Default is 5 (trained M2 outputs reach about 3.2).')
    parser.add_argument('--ren-ref-trace-tau', type=float, default=5.0, help='Time constant [s] of the reference trace per car fed to the REN: at a goal change it jumps by the way still to go (goal minus pose: x, y and heading as sin, 1 - cos) and decays from that value. 0 = only the pulse itself. Default is 5.')

    # loss
    parser.add_argument('--alpha-u', type=float, default=0.1/400, help='Weight of the effort loss: squared rPB offset |dxref|^2, averaged over time. Default is 0.1/400.')
    parser.add_argument('--alpha-col', type=float, default=1500, help='Weight of the collision avoidance loss. Default is 1500 if "col-av" is True, else None.')
    parser.add_argument('--alpha-obst', type=float, default=20000, help='Weight of the obstacle avoidance loss. Default is 20000 if "obst-av" is True, else None.')
    parser.add_argument('--obst-safe-dist', type=float, default=1.7, help='Obstacle cost is 0 beyond this distance [m] from each obstacle center (Gaussian minus its value there, clipped at 0). Default is 1.7 (sim.py hit radius 1.5 + 0.2 margin).')
    parser.add_argument('--alpha-bounds', type=float, default=1e4, help='Weight of the arena loss: quadratic in how far a car center is inside the 1 m margin (car radius) of the arena edge (x in [-3.5, 3.5], y in [-4, 6]). Default is 1e4 (0 = off).')
    parser.add_argument('--min-dist', type=float, default=2.0, help='Minimum distance [m] between car centers; the collision loss is active below min_dist + 0.2. Default is 2.0 if "col-av" is True, else None.')
    parser.add_argument('--q-final-factor', type=float, default=500, help='Terminal position weight: Q_final = this * Q (per car (2/15) * factor per m^2), averaged over the --goal-window-steps arrival window of every goal leg. Default is 570 (final position and final heading of similar weight: 10 cm ~ 10 deg).')
    parser.add_argument('--alpha-heading', type=float, default=50.0, help='Weight of the running heading loss 1 - cos(theta* - theta), weighted by exp(-d^2 / (2 * heading_radius^2)) with d the distance to the goal. Default is 50 (0 = off).')
    parser.add_argument('--alpha-heading-final', type=float, default=50.0, help='Weight of the terminal heading loss over the --goal-window-steps arrival window of every goal leg. Default is 50 (0 = off).')
    parser.add_argument('--heading-radius', type=float, default=0.75, help='Length scale [m] of the soft mask of the running heading loss (weight 0.61 at this distance from the goal). Default is 0.75.')
    parser.add_argument('--alpha-smooth', type=float, default=50.0, help='Weight of the smoothness loss: squared change of the rPB offset dxref per step, averaged over time (penalizes jumps of the tracked reference, not its size). Default is 50 (0 = off).')
    parser.add_argument('--alpha-offset-final', type=float, default=20.0, help='Weight of the squared rPB offset dxref over the last --goal-window-steps steps of the rollout (the car must have settled by the end). Default is 20 (0 = off).')
    parser.add_argument('--goal-window-steps', type=int, default=50, help='Arrival window: number of time steps at the end of every goal leg over which the final position and heading losses are averaged (also the window of the offset loss at the end). Default is 50 (2 s at dt 0.04).')

    # optimizer
    parser.add_argument('--batch-size', type=int, default=100, help='Number of forward trajectories of the closed-loop system at each step (-1 = all training rollouts). Default is 100.')
    parser.add_argument('--epochs', type=int, default=300, help='Total number of epochs for training. Default is 200.')
    parser.add_argument('--lr', type=float, default=2e-3, help='Initial learning rate. Default is 2e-3.')
    parser.add_argument('--log-epoch', type=int, default=5, help='Frequency of logging, validation and checkpoints in epochs. Default is 5.')
    parser.add_argument('--lr-schedule', type=str, default='cosine', choices=['constant', 'linear', 'cosine'], help='Learning-rate schedule over the epochs. Default is cosine.')
    parser.add_argument('--lr-end-factor', type=float, default=0.05, help='Learning rate at the last epoch as a fraction of --lr (linear/cosine schedules). Default is 0.05.')
    parser.add_argument('--grad-clip', type=float, default=3000, help='Clip the gradient norm of each step to this value (--grad-clip inf = no clipping; the norm is always logged). Default is 3000.')
    parser.add_argument('--patience-epoch', type=int, default=50, help='Early stopping: stop when the validation loss has not improved for this many epochs (checked at the logged epochs; needs --return-best). Default is None (off).')
    parser.add_argument('--return-best', action=argparse.BooleanOptionalAction, default=True, help='Return the best model on the validation data among all logged iterations. The train data can be used instead of validation data. The Default is True.')
    parser.add_argument('--load-controller', type=str, default=None, help='Optional path to a compatible trained controller checkpoint.')
    parser.add_argument('--current-scenario', action=argparse.BooleanOptionalAction, default=False, help='sim.py only: evaluate on the current SCENARIOS/OBSTACLES of experiment_params.py instead of the scenario stored in the checkpoint. Default is False.')
    parser.add_argument('--checkpoint', type=str, default=None, help='sim.py only: checkpoint to evaluate. Default is the provided trained_pRB/rPB_best.pt.')

    # TODO: add the following
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
    msg += ' -- REN reference trace tau: %s s' % args.ren_ref_trace_tau
    msg += ' -- base turn speed: %s' % args.turn_speed
    msg += ' -- M1: REN per car -- M2: %s per car (gain bound %g)' % (args.m2, args.m2_gain_bound)

    msg += '\n[INFO] Loss:  alpha_u: %.6f' % args.alpha_u + ' -- q_final_factor: %g' % args.q_final_factor
    msg += ' -- alpha_col: %.f -- min_dist: %g' % (args.alpha_col, args.min_dist) if args.col_av else ' -- no collision avoidance'
    msg += ' -- alpha_obst: %.1f -- obst_safe_dist: %s' % (args.alpha_obst, args.obst_safe_dist) if args.obst_av else ' -- no obstacle avoidance'
    msg += ' -- alpha_bounds: %g' % args.alpha_bounds
    msg += ' -- alpha_heading: %g -- alpha_heading_final: %g -- heading_radius: %g -- goal_window_steps: %i -- alpha_offset_final: %g' % (
        args.alpha_heading, args.alpha_heading_final, args.heading_radius, args.goal_window_steps, args.alpha_offset_final)
    msg += ' -- alpha_smooth: %g' % args.alpha_smooth

    msg += '\n[INFO] Optimizer: lr: %.2e' % args.lr
    msg += ' -- lr schedule: %s (end factor %g) -- grad clip: %s' % (args.lr_schedule, args.lr_end_factor, args.grad_clip)
    msg += ' -- early stopping patience: %s epochs' % args.patience_epoch
    msg += ' -- batch_size: %i' % args.batch_size + ', -- return best model for validation data among logged epochs:' + str(args.return_best)

    return msg
