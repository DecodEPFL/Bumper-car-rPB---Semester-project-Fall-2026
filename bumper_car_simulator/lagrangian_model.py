#!/usr/bin/env python3
import numpy as np
from utils import normalize_angle

class LagrangianModel:
    def __init__(self, initial_state, params):

        # Internal variables
        self.prev_car_state = np.array(initial_state, dtype=np.float32)
        self.car_state = np.array(initial_state, dtype=np.float32)
        self.par = params

    def bumper_car_dynamics(self, car_state, car_input):
        # Extract state and input
        theta = car_state[2]
        vx_b = car_state[3]
        vy_b = car_state[4]
        omega = car_state[5]
        delta = car_state[6]

        # Steering
        delta_ref = car_input[1] * self.par.steering_range/2
        delta_dot = np.clip((delta_ref - delta) / self.par.tl_steering, -self.par.steering_speed, self.par.steering_speed)

        # wheel velocities
        vy_bf = vy_b + self.par.lf*omega
        vy_br = vy_b - self.par.lr*omega
        vx_f = vx_b*np.cos(delta) + vy_bf*np.sin(delta)

        # Slip angles
        beta_f = np.atan2(vy_bf + np.sin(delta)*self.par.vel_threshold, vx_b + np.cos(delta)*self.par.vel_threshold)
        beta_r = np.atan2(vy_br, vx_b + np.sign(vx_b)*self.par.vel_threshold)
        alpha_f = np.clip(beta_f - delta, -np.pi - (beta_f - delta), np.pi - (beta_f - delta))
        alpha_r = np.clip(beta_r, -np.pi - beta_r, np.pi - beta_r)

        # throttle
        v_xf_ref = car_input[0] * self.par.max_speed
        f_motor_t = np.clip(self.par.torque_gain * 1500 * (v_xf_ref - vx_f), 0, self.par.max_force)

        # brake
        brake_in = vx_f * car_input[0]
        f_motor_b = np.clip(brake_in * self.par.breaking_gain, -self.par.max_force, 0)

        # Forces
        f_motor = f_motor_t + f_motor_b

        fy_f = -np.clip(self.par.cf * alpha_f, -self.par.f_max_f, self.par.f_max_f)
        fy_r = -np.clip(self.par.cr * alpha_r, -self.par.f_max_r, self.par.f_max_r)

        fx_f = f_motor - (self.par.rx - self.par.rx_2 * vx_f) * vx_f
        fx_r = 0

        fx = fx_f*np.cos(delta) - fy_f*np.sin(delta) - fx_r
        fy = fy_f*np.cos(delta) + fy_r + fx_f*np.sin(delta)
        mz = self.par.lf*fy_f*np.cos(delta) + self.par.lf*fx_f*np.sin(delta) - self.par.lr*fy_r

        vx_b_dot = fx/self.par.m + omega*vy_b
        vy_b_dot = fy/self.par.m - omega*vx_b
        omega_dot = mz/self.par.iz
        car_state_dot = np.zeros(7)
        car_state_dot[0] = vx_b * np.cos(theta) - vy_b * np.sin(theta)
        car_state_dot[1] = vx_b * np.sin(theta) + vy_b * np.cos(theta)
        car_state_dot[2] = omega
        car_state_dot[3] = vx_b_dot
        car_state_dot[4] = vy_b_dot
        car_state_dot[5] = omega_dot
        car_state_dot[6] = delta_dot

        return car_state_dot

    def lagrangian_forward(self, car_input, delta_t=0.1):

        # Convert to Lagrangian coordinates
        vf = self.car_state[3]
        beta_f = self.car_state[4]
        beta_r = self.car_state[5]

        beta_b_cog = np.arctan2(self.par.lr*np.tan(beta_f) + self.par.lf*np.tan(beta_r), self.par.lf + self.par.lr)
        vx_b = vf * np.cos(beta_f)
        vy_b = vx_b * np.tan(beta_b_cog)
        omega = vf*np.sin(beta_f - beta_r) / ((self.par.lf + self.par.lr)*np.cos(beta_r))

        lagrangian_state = np.array([self.car_state[0], self.car_state[1], self.car_state[2], vx_b, vy_b, omega, self.car_state[6]], dtype=np.float32)

        # RK4 integration in Lagrangian coordinates
        k1 = self.bumper_car_dynamics(lagrangian_state, car_input)
        k2 = self.bumper_car_dynamics(lagrangian_state + 0.5 * delta_t * k1, car_input)
        k3 = self.bumper_car_dynamics(lagrangian_state + 0.5 * delta_t * k2, car_input)
        k4 = self.bumper_car_dynamics(lagrangian_state + delta_t * k3, car_input)
        lagrangian_state = lagrangian_state + (delta_t / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        # Convert back to kinematic state
        vy_bf = lagrangian_state[4] + self.par.lf*lagrangian_state[5]
        vy_br = lagrangian_state[4] - self.par.lr*lagrangian_state[5]
        beta_f = np.atan2(vy_bf + np.sin(lagrangian_state[6])*self.par.vel_threshold, lagrangian_state[3] + np.cos(lagrangian_state[6])*self.par.vel_threshold)
        beta_r = np.atan2(vy_br, lagrangian_state[3] + np.sign(lagrangian_state[3])*self.par.vel_threshold)

        self.car_state[0] = lagrangian_state[0]
        self.car_state[1] = lagrangian_state[1]
        self.car_state[2] = normalize_angle(lagrangian_state[2])
        self.car_state[3] = lagrangian_state[3] / np.cos(beta_f)
        self.car_state[4] = beta_f
        self.car_state[5] = beta_r
        self.car_state[6] = lagrangian_state[6]

    def update(self, car_input):

        # Save previous state
        self.prev_car_state = self.car_state

        # Compute velocities
        self.lagrangian_forward(car_input, 0.05)
        self.lagrangian_forward(car_input, 0.05)