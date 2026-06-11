#!/usr/bin/env python3

class CarParams:
    # Physical parameters
    m: float = 319.6
    lf: float = 0.54
    lr: float = 0.33
    iz: float = 88.07
    cf: float = 5433.0
    cr: float = 7950.9

    # Forces
    f_max_f: float = 638.0
    f_max_r: float = 1592.0
    rx: float = 42.62
    rx_2: float = 17.37

    # Kinematic
    max_speed: float = 2.033
    torque_gain: float = 0.1814
    max_force: float = 400.5
    breaking_gain: float = 587.7
    steering_range: float = 4.024
    steering_speed: float = 28.36
    tl_steering: float = 0.1550
    vel_threshold: float = 1.0
    beta_correction: float = 20.0

    # Arena
    x_min = 0.0
    x_max = 10.0
    y_min = 0.0
    y_max = 10.0

# Initialize parameters
params = CarParams