import torch, os
from scipy.stats import multivariate_normal # TODO: use something compatible with tensors
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import imageio

AGENT_COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']


def agent_color(i):
    return AGENT_COLORS[i % len(AGENT_COLORS)]


def draw_heading_arrow(ax, pose, color, length=0.5, hollow=False):
    """Short arrow from (x, y) in the direction theta of pose [x, y, theta]; hollow head for the car's final heading."""
    x, y, theta = (float(v) for v in pose.detach().cpu())
    ax.annotate(
        '', xy=(x + length * np.cos(theta), y + length * np.sin(theta)), xytext=(x, y),
        arrowprops=dict(arrowstyle='-|>', color=color, lw=1.2, fill=not hollow, mutation_scale=10),
        zorder=6,
    )


def route_points(ref):
    """
    Route of one rollout from its reference over time ref [T, 7*n_agents]: the distinct references
    in the order they become active. Returns (earlier_goals, goal): a list of [7*n_agents] states for
    the goals before the last one (empty for a single goal) and the [7*n_agents] state of the last goal.
    """
    changes = (ref[1:] != ref[:-1]).any(dim=-1).nonzero().flatten() + 1     # steps where the reference switches
    starts = [0] + changes.tolist()
    points = [ref[k] for k in starts]
    return points[:-1], points[-1]


def plot_trajectories(
    x, xbar, n_agents, save_folder, text="", save=True, filename='', T=100,
    dots=False, circles=False, axis=True, min_dist=1, f=5,
    obstacle_centers=None, obstacle_covs=None, earlier_goals=(), x_baseline=None
):
    """
    xbar: last goal state [7*n_agents]; earlier_goals: [7*n_agents] states of the goals visited before it (goal sequence);
    every goal is drawn as a star with its target-heading arrow, numbered in visiting order when there are several;
    x_baseline: optional second rollout on the same sample [T, 7*n_agents], drawn dashed grey (e.g. P-controller only).
    """
    filename = 'trajectories.pdf' if filename == '' else filename

    # fig = plt.figure(f)
    fig, ax = plt.subplots(figsize=(f,f))

    # axis limits: fit the trajectories, the goals and the obstacles (+-2 std), plus a margin
    points = list(earlier_goals) + [xbar]
    rollouts = [x] if x_baseline is None else [x, x_baseline]
    xs = [r[:, 7*i].detach().cpu().numpy() for r in rollouts for i in range(n_agents)] + [p[7*i].detach().cpu().numpy().reshape(1) for p in points for i in range(n_agents)]
    ys = [r[:, 7*i+1].detach().cpu().numpy() for r in rollouts for i in range(n_agents)] + [p[7*i+1].detach().cpu().numpy().reshape(1) for p in points for i in range(n_agents)]
    if obstacle_centers is not None:
        for k, center in enumerate(obstacle_centers):
            c = center.detach().cpu().numpy().flatten()
            r = 2 * np.sqrt(obstacle_covs[k].detach().cpu().numpy().flatten()) if obstacle_covs is not None else np.zeros(2)
            xs.append(np.array([c[0] - r[0], c[0] + r[0]]))
            ys.append(np.array([c[1] - r[1], c[1] + r[1]]))
    xs, ys = np.concatenate(xs), np.concatenate(ys)
    margin = 0.5
    x_lim = (xs.min() - margin, xs.max() + margin)
    y_lim = (ys.min() - margin, ys.max() + margin)

    # plot obstacles
    if not obstacle_covs is None:
        assert not obstacle_centers is None
        xx, yy = np.meshgrid(np.linspace(*x_lim, 150), np.linspace(*y_lim, 150))
        grid = np.stack([xx.ravel(), yy.ravel()], axis=-1)
        zz = np.zeros(xx.size)
        for center, cov in zip(obstacle_centers, obstacle_covs):
            distr = multivariate_normal(
                cov=torch.diag(cov.flatten()).detach().clone().cpu().numpy(),
                mean=center.detach().clone().cpu().numpy().flatten()
            )
            zz += distr.pdf(grid)
        zz = zz.reshape(xx.shape)
        z_min, z_max = np.abs(zz).min(), np.abs(zz).max()

        ax.pcolormesh(xx, yy, zz, cmap='Greys', vmin=z_min, vmax=z_max, shading='gouraud')

    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    ax.set_aspect('equal', adjustable='box')

    ax.set_title(text)
    if x_baseline is not None:
        for i in range(n_agents):
            ax.plot(
                x_baseline[:, 7*i].detach().cpu(), x_baseline[:, 7*i+1].detach().cpu(),
                color='0.45', linewidth=1, linestyle='--', zorder=1
            )
    for i in range(n_agents):
        ax.plot(
            x[:T+1,7*i].detach().cpu(), x[:T+1,7*i+1].detach().cpu(),
            color=agent_color(i), linewidth=1
        )
        ax.plot(
            x[T:,7*i].detach().cpu(), x[T:,7*i+1].detach().cpu(),
            color='k', linewidth=0.1, linestyle='dotted', dashes=(3, 15)
        )
    for i in range(n_agents):
        ax.plot(
            x[0,7*i].detach().cpu(), x[0,7*i+1].detach().cpu(),
            color=agent_color(i), marker='8'
        )
        for k, goal in enumerate(points):
            ax.plot(
                goal[7*i].detach().cpu(), goal[7*i+1].detach().cpu(),
                color=agent_color(i), marker='*', markersize=10
            )
            if len(points) > 1:   # goal sequence: number the goals in visiting order
                ax.annotate(str(k + 1), (float(goal[7*i]), float(goal[7*i+1])), textcoords='offset points',
                            xytext=(6, -9), color=agent_color(i), fontsize=7, fontweight='bold')
        # heading arrows: start heading, target heading at every goal (solid),
        # and the car's heading at the end of the rollout (hollow head) to compare with the goal's target heading
        for pose in [x[0, 7*i:7*i+3]] + [p[7*i:7*i+3] for p in points]:
            draw_heading_arrow(ax, pose, agent_color(i))
        draw_heading_arrow(ax, x[-1, 7*i:7*i+3], agent_color(i), hollow=True)

    if dots:
        for i in range(n_agents):
            ax.plot(  
                x[:T+1,7*i].detach().cpu(), x[:T+1,7*i+1].detach().cpu(),
                color=agent_color(i), linewidth=1, marker = "x"
            )

    if circles:
        for i in range(n_agents):
            r = min_dist/2
            circle = ax.Circle(
                (x[T, 7*i].detach().cpu(), x[T, 7*i+1].detach().cpu()),
                r, color=agent_color(i), alpha=0.5, zorder=10
            )
            ax.add_patch(circle)
    legend = [
        plt.Line2D([0], [0], color='0.25', linewidth=1, label='rPB'),
        plt.Line2D([0], [0], color='0.25', marker='8', linestyle='None', label='start (this sample)'),
        plt.Line2D([0], [0], color='0.25', marker='*', markersize=10, linestyle='None', label='goal (this sample)'),
    ]
    if len(earlier_goals) > 0:
        legend.append(plt.Line2D([0], [0], linestyle='None', label='goals numbered in visiting order'))
    legend.append(plt.Line2D([0], [0], color='0.25', marker=r'$\rightarrow$', markersize=10, linestyle='None', label='start / target heading'))
    legend.append(plt.Line2D([0], [0], color='0.25', marker=r'$\Rightarrow$', markersize=10, linestyle='None', label='final heading (rPB)'))
    if x_baseline is not None:
        legend.append(plt.Line2D([0], [0], color='0.45', linewidth=1, linestyle='--', label='P-controller only'))
    ax.legend(handles=legend, loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize=7, framealpha=0.9)   # outside the axes
    ax.axes.xaxis.set_visible(axis)
    ax.axes.yaxis.set_visible(axis)
    if save:
        fig.savefig(
            os.path.join(save_folder, filename),
            format='pdf', bbox_inches='tight'
        )
        plt.close()
    else:
        plt.show()


def plot_traj_vs_time(t_end, n_agents, save_folder, x, u=None, text="", save=True, filename=''):
    filename = filename if filename=='' else filename+'_'
    now = datetime.now()
    formatted_date = now.strftime('%m-%d-%H:%M')
    t = torch.linspace(0,t_end-1, t_end)
    if u is not None:
        p = 3
    else:
        p = 2
    plt.figure(figsize=(4*p, 4))
    plt.subplot(1, p, 1)
    for i in range(n_agents):
        plt.plot(t, x[:,7*i].detach().cpu())
        plt.plot(t, x[:,7*i+1].detach().cpu())
    plt.xlabel(r'$t$')
    plt.title(r'$x(t)$')
    plt.subplot(1, p, 2)
    for i in range(n_agents):
        plt.plot(t, x[:,7*i+2].detach().cpu())
        plt.plot(t, x[:,7*i+3].detach().cpu())
    plt.xlabel(r'$t$')
    plt.title(r'$v(t)$')
    plt.suptitle(text)
    if p == 3:
        plt.subplot(1, 3, 3)
        for i in range(n_agents):
            plt.plot(t, u[:, 2*i].detach().cpu())
            plt.plot(t, u[:, 2*i+1].detach().cpu())
        plt.xlabel(r'$t$')
        plt.title(r'$u(t)$')
    if save:
        plt.savefig(
            os.path.join(
                save_folder,
                filename+text+'_x_u.pdf'
            ),
            format='pdf'
        )
        plt.close()
    else:
        plt.show()


def save_trajectory_frames(x, xbar, n_agents, save_folder, T=100, interval=1,f=5, obstacle_centers=None, obstacle_covs=None):
    os.makedirs(save_folder, exist_ok=True)

    # fig = plt.figure(f)
    fig, ax = plt.subplots(figsize=(f,f))
    # plot obstacles
    if not obstacle_covs is None:
        assert not obstacle_centers is None
        yy, xx = np.meshgrid(np.linspace(-3, 7, 100), np.linspace(-3, 7, 100))
        zz = xx * 0
        for center, cov in zip(obstacle_centers, obstacle_covs):
            distr = multivariate_normal(
                cov=torch.diag(cov.flatten()).detach().clone().cpu().numpy(),
                mean=center.detach().clone().cpu().numpy().flatten()
            )
            for i in range(xx.shape[0]):
                for j in range(xx.shape[1]):
                    zz[i, j] += distr.pdf([xx[i, j], yy[i, j]])
        z_min, z_max = np.abs(zz).min(), np.abs(zz).max()

        ax.pcolormesh(xx, yy, zz, cmap='Greys', vmin=z_min, vmax=z_max, shading='gouraud')



    for t in range(0, T+1, interval):
        print(f'Saving frame {t}/{T}...')
        fig, ax = plt.subplots(figsize=(f,f))
        # plot obstacles
        if not obstacle_covs is None:
            assert not obstacle_centers is None
            yy, xx = np.meshgrid(np.linspace(-3, 7, 100), np.linspace(-3, 7, 100))
            zz = xx * 0
            for center, cov in zip(obstacle_centers, obstacle_covs):
                distr = multivariate_normal(
                    cov=torch.diag(cov.flatten()).detach().clone().cpu().numpy(),
                    mean=center.detach().clone().cpu().numpy().flatten()
                )
                for i in range(xx.shape[0]):
                    for j in range(xx.shape[1]):
                        zz[i, j] += distr.pdf([xx[i, j], yy[i, j]])
            z_min, z_max = np.abs(zz).min(), np.abs(zz).max()

            ax.pcolormesh(xx, yy, zz, cmap='Greys', vmin=z_min, vmax=z_max, shading='gouraud')
        for i in range(n_agents):
            ax.plot(
                x[:t+1, 7*i].detach().cpu(), x[:t+1, 7*i+1].detach().cpu(),
                color=agent_color(i), linewidth=1
            )
            ax.plot(
                x[t, 7*i].detach().cpu(), x[t, 7*i+1].detach().cpu(),
                color=agent_color(i), marker='o'
            )
            ax.plot(
                x[0, 7*i].detach().cpu(), x[0, 7*i+1].detach().cpu(),
                color=agent_color(i), marker='o', markerfacecolor='none'
            )
            r = 0.5
            circle = plt.Circle(
                (x[t, 7*i].detach().cpu(), x[t, 7*i+1].detach().cpu()),
                r, color=agent_color(i), alpha=0.5, zorder=10
            )
            ax.add_patch(circle)
            ax.plot(
                xbar[7*i].detach().cpu(), xbar[7*i+1].detach().cpu(),
                color=agent_color(i), marker='*', markersize=10
            )
            
        pos = torch.stack([x[t, 7*i:7*i+2] for i in range(n_agents)])
        dist = torch.cdist(pos, pos)[~torch.eye(n_agents, dtype=torch.bool)].min()   # closest pair
        if dist < 1:
            ax.set_title('Collision', color='red')
             
        else:
            ax.set_title(f'Time step: {t}')
        """ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)"""
        
        frame_filename = os.path.join(save_folder, f'frame_{t:03d}.png')
        fig.savefig(frame_filename)
        plt.close(fig)


def create_gif_from_frames(frame_folder, gif_filename, duration=0.1):
    frames = []
    for frame_file in sorted(os.listdir(frame_folder)):
        if frame_file.endswith('.png'):
            frame_path = os.path.join(frame_folder, frame_file)
            frames.append(imageio.imread(frame_path))
    imageio.mimsave(gif_filename, frames, duration=duration)