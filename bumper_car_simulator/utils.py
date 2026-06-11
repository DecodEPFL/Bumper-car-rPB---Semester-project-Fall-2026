#!/usr/bin/env python3
import numpy as np
import torch

def normalize_angle(angle):
    """
    Maps an angle in radians to [-pi, pi].

    :param angle: Angle in radians

    :return: Angle in range [-pi, pi].
    """
    if isinstance(angle, torch.Tensor):
        return torch.atan2(torch.sin(angle), torch.cos(angle))
    else:
        return np.arctan2(np.sin(angle), np.cos(angle))

def delta_to_beta(delta, lf, lr):
    """
    Converts the steering angle delta to the slip angle beta.

    :param delta: Steering angle in radians
    :param lf: Distance between center of mass and front wheel axle
    :param lr: Distance between center of mass and rear wheel axle

    :return: Slip angle in radians.
    """
    beta = np.atan2(lr/(lf+lr) * np.sin(delta), np.cos(delta))

    return beta


def beta_to_delta(beta, lf, lr):
    """
    Converts the slip angle beta to the steering angle delta.

    :param beta: Slip angle in radians
    :param lf: Distance between center of mass and front wheel axle
    :param lr: Distance between center of mass and rear wheel axle

    :return: Steering angle in radians.
    """
    delta = np.atan2((lf+lr)/lr * np.sin(beta) , np.cos(beta))

    return delta

def get_nearest_walls(car_state, par):
    # Extract state
    beta_f = car_state[4]
    beta_r = car_state[5]
    beta_cog = np.arctan2(par.lr*np.tan(beta_f) + par.lf*np.tan(beta_r), par.lf + par.lr)
    v = car_state[3] * np.cos(car_state[6])/np.cos(beta_cog)
    vx = v * np.cos(car_state[2] + beta_cog)
    vy = v * np.sin(car_state[2] + beta_cog)

    # Determine nearest walls
    if abs(car_state[0] - par.x_min) < abs(car_state[0] - par.x_max):
        vertical_wall_x = par.x_min
    else:
        vertical_wall_x = par.x_max
    vertical_wall_y = car_state[1]

    if abs(car_state[1] - par.y_min) < abs(car_state[1] - par.y_max):
        horizontal_wall_y = par.y_min
    else:
        horizontal_wall_y = par.y_max
    horizontal_wall_x = car_state[0]

    # Build wall obstacles
    horizontal_wall = np.array([horizontal_wall_x, horizontal_wall_y, 0.0, vx, 0.0, 0.0, 0.0])
    vertical_wall = np.array([vertical_wall_x, vertical_wall_y, np.pi/2, vy, 0.0, 0.0, 0.0])

    return np.array([horizontal_wall, vertical_wall])

def get_nearest_corner(car_state, par):
    if abs(car_state[0] - par.x_min) < abs(car_state[0] - par.x_max):
        corner_x = par.x_min
    else:
        corner_x = par.x_max

    if abs(car_state[1] - par.y_min) < abs(car_state[1] - par.y_max):
        corner_y = par.y_min
    else:
        corner_y = par.y_max

    corner = np.array([corner_x, corner_y, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape(1, 7)
    return corner
