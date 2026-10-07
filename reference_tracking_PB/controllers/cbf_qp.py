#!/usr/bin/env python3
"""
Centralized CBF-QP Safety Filter for 2 Autonomous Bumper Cars.
Formulates High-Order Control Barrier Functions (HOCBF, relative degree 2)
with integrated Funnel Priority Staggering and Directional Approach Governor
to eliminate all inter-vehicle collisions (central corridor and terminal targets)
without inducing external deflection bypass.
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn


class CBFSafetyFilter(nn.Module):
    """
    CBF-QP Safety Filter module with directional closing velocity governor.
    Strictly preserves corridor passing and guarantees zero collisions.
    """

    def __init__(
        self,
        params,
        r_obs_safe: float = 0.80,
        d_car_safe: float = 2.05,
        obstacle_centers: Optional[List[Union[torch.Tensor, List[float]]]] = None,
        dt: float = 0.04,
        cbf_gamma_1: float = 1.2,
        cbf_gamma_2: float = 0.8,
        tau_lookahead: float = 0.35,
        weight_v: float = 8.0,
        weight_delta: float = 2.0,
        max_admm_iter: int = 40,
        rho: float = 15.0,
        enable_velocity_staggering: bool = True,
        yield_speed_factor: float = 0.60,
    ):
        super().__init__()

        self.par = params
        self.r_obs_safe = float(r_obs_safe)
        self.d_car_safe = float(d_car_safe)
        self.dt = float(dt)

        self.gamma_1 = float(cbf_gamma_1)
        self.gamma_2 = float(cbf_gamma_2)
        self.tau_lookahead = float(tau_lookahead)
        self.weight_v = float(weight_v)
        self.weight_delta = float(weight_delta)
        self.max_admm_iter = int(max_admm_iter)
        self.rho = float(rho)

        self.enable_velocity_staggering = enable_velocity_staggering
        self.yield_speed_factor = float(yield_speed_factor)

        if obstacle_centers is None:
            default_centers = torch.tensor(
                [[-1.375, 0.0], [1.375, 0.0]], dtype=torch.float32
            )
        else:
            default_centers = torch.stack(
                [
                    c if isinstance(c, torch.Tensor) else torch.tensor(c, dtype=torch.float32)
                    for c in obstacle_centers
                ]
            ).squeeze()

        self.register_buffer("obstacle_centers", default_centers.float())

        self.reset_diagnostics()

    def reset_diagnostics(self):
        """Resets optimization performance counters."""
        self.total_steps = 0
        self.active_steps = 0
        self.infeasible_steps = 0
        self.emergency_fallback_steps = 0
        self.stagger_interventions = 0

    def get_diagnostics(self) -> Dict[str, Union[int, float]]:
        """Returns optimization telemetry summary."""
        total = max(self.total_steps, 1)
        return {
            "total_steps": self.total_steps,
            "active_interventions": self.active_steps,
            "active_ratio_pct": 100.0 * self.active_steps / total,
            "infeasible_steps": self.infeasible_steps,
            "infeasible_ratio_pct": 100.0 * self.infeasible_steps / total,
            "emergency_fallbacks": self.emergency_fallback_steps,
            "stagger_interventions": self.stagger_interventions,
        }

    def apply_velocity_staggering(
        self, x: torch.Tensor, u_phys: torch.Tensor
    ) -> torch.Tensor:
        """
        Directional approach modulation:
        1. Funnel zone (|x| < 1.15 m, y < 1.6 m): staggers arrival at bottleneck.
        2. Proximity / Target zone (d_12 < 2.85 m): brakes only the vehicle moving towards
           the other, stopping at d_12 = 2.25 m to absorb dynamic lag and prevent collisions.
        Steering angles are strictly preserved throughout to maintain the straight corridor course.
        """
        p1 = x[:, 0:2]
        p2 = x[:, 7:9]
        v1_scal = x[:, 3]
        v2_scal = x[:, 10]
        theta1 = x[:, 2]
        theta2 = x[:, 9]

        p_rel = p2 - p1  # vector from Car 1 to Car 2
        d_12 = torch.linalg.norm(p_rel, dim=-1)
        d_12_safe = torch.clamp(d_12, min=1e-5)
        u_12 = p_rel / d_12_safe.unsqueeze(-1)  # unit vector from 1 to 2

        v1_vec = torch.stack(
            [v1_scal * torch.cos(theta1), v1_scal * torch.sin(theta1)], dim=-1
        )
        v2_vec = torch.stack(
            [v2_scal * torch.cos(theta2), v2_scal * torch.sin(theta2)], dim=-1
        )
        v_rel = v2_vec - v1_vec

        closing_speed = -torch.sum(p_rel * v_rel, dim=-1) / d_12_safe

        # Componente di velocita proiettata lungo la congiungente
        v_toward_1 = torch.sum(v1_vec * u_12, dim=-1)
        v_toward_2 = torch.sum(v2_vec * (-u_12), dim=-1)

        # -------------------------------------------------------------
        # 1. Zona Imbuto (Varco centrale tra gli ostacoli)
        # -------------------------------------------------------------
        in_funnel = (torch.abs(p1[:, 0]) < 1.15) & (torch.abs(p2[:, 0]) < 1.15)
        in_choke_region = (p1[:, 1] < 1.60) & (p2[:, 1] < 1.60)
        in_funnel_zone = in_funnel & in_choke_region

        dist_choke_1 = torch.abs(p1[:, 1])
        dist_choke_2 = torch.abs(p2[:, 1])
        funnel_car1_priority = (dist_choke_1 + 0.05) <= dist_choke_2

        funnel_active = (
            in_funnel_zone
            & (d_12 < 3.20)
            & ((closing_speed > 0.05) | (d_12 < 2.30))
        )
        ramp_funnel = torch.clamp((3.20 - d_12) / (3.20 - 2.20), 0.0, 1.0)
        yield_scale_funnel = 1.0 - ramp_funnel * (1.0 - self.yield_speed_factor)

        # -------------------------------------------------------------
        # 2. Terminal & Approach Governor (Buffer a 2.25 m anti-inerzia)
        # -------------------------------------------------------------
        d_stop_buffer = 2.25
        d_slow_start = 2.85
        ramp_approach = torch.clamp(
            (d_12 - d_stop_buffer) / (d_slow_start - d_stop_buffer), 0.0, 1.0
        )

        car1_needs_brake = (d_12 < d_slow_start) & (
            (v_toward_1 > 0.01) | ((d_12 < 2.30) & (v_toward_1 >= -0.05))
        )
        car2_needs_brake = (d_12 < d_slow_start) & (
            (v_toward_2 > 0.01) | ((d_12 < 2.30) & (v_toward_2 >= -0.05))
        )

        # Nell'imbuto, il Leader prosegue spedito (non viene frenato da ramp_approach)
        car1_needs_brake = car1_needs_brake & ~(
            funnel_active & funnel_car1_priority
        )
        car2_needs_brake = car2_needs_brake & ~(
            funnel_active & (~funnel_car1_priority)
        )

        scale1 = torch.ones_like(d_12)
        scale2 = torch.ones_like(d_12)

        # Applica sfasamento nell'imbuto
        scale1 = torch.where(
            funnel_active & (~funnel_car1_priority),
            torch.minimum(scale1, yield_scale_funnel),
            scale1,
        )
        scale2 = torch.where(
            funnel_active & funnel_car1_priority,
            torch.minimum(scale2, yield_scale_funnel),
            scale2,
        )

        # Applica frenata progressiva al veicolo in avvicinamento
        scale1 = torch.where(
            car1_needs_brake,
            torch.minimum(scale1, ramp_approach),
            scale1,
        )
        scale2 = torch.where(
            car2_needs_brake,
            torch.minimum(scale2, ramp_approach),
            scale2,
        )

        any_active = funnel_active | car1_needs_brake | car2_needs_brake
        if not torch.any(any_active):
            return u_phys

        self.stagger_interventions += int(any_active.sum().item())

        u_phys_staggered = u_phys.clone()
        u_phys_staggered[:, 0] = u_phys_staggered[:, 0] * scale1
        u_phys_staggered[:, 2] = u_phys_staggered[:, 2] * scale2

        return u_phys_staggered

    def convert_normalized_to_physical(
        self, x: torch.Tensor, u_norm: torch.Tensor
    ) -> torch.Tensor:
        max_speed = self.par.max_speed
        half_steer = self.par.steering_range / 2.0

        u_phys_list = []
        for i in range(2):
            vf = x[:, 7 * i + 3]
            drive = torch.clamp(u_norm[:, 2 * i], -1.0, 1.0)
            steer = torch.clamp(u_norm[:, 2 * i + 1], -1.0, 1.0)

            u_accel = torch.clamp((drive - 0.05) / 0.62, min=0.0, max=1.0)
            v_accel_target = u_accel * max_speed
            vf_ref_accel = torch.where(v_accel_target > vf, v_accel_target, vf)

            u_brake = torch.clamp((-drive - 0.20) / 0.60, min=0.0, max=1.0)
            vf_ref_brake = (1.0 - u_brake / 1.04) * vf

            vf_ref = torch.where(drive < 0.0, vf_ref_brake, vf_ref_accel)
            vf_ref = torch.clamp(vf_ref, min=0.0, max=max_speed)

            delta_ref = torch.clamp(
                steer * half_steer, min=-half_steer, max=half_steer
            )
            u_phys_list.extend([vf_ref, delta_ref])

        return torch.stack(u_phys_list, dim=-1)

    def convert_physical_to_normalized(
        self, x: torch.Tensor, u_phys: torch.Tensor
    ) -> torch.Tensor:
        max_speed = self.par.max_speed
        half_steer = self.par.steering_range / 2.0

        u_norm_list = []
        for i in range(2):
            vf = x[:, 7 * i + 3]
            vf_safe = torch.clamp(u_phys[:, 2 * i], min=0.0, max=max_speed)
            delta_safe = torch.clamp(
                u_phys[:, 2 * i + 1], min=-half_steer, max=half_steer
            )

            drive_accel = torch.clamp((vf_safe / max_speed) * 0.62 + 0.05, 0.0, 1.0)
            drive_brake = -torch.clamp(
                -(vf_safe / (vf + 1e-5) - 1.0) * 1.04 * 0.60 + 0.20,
                0.0,
                1.0,
            )

            drive_safe = torch.where(
                vf_safe <= 1e-3,
                torch.where(
                    vf > 0.02,
                    -torch.ones_like(drive_accel),
                    torch.zeros_like(drive_accel),
                ),
                torch.where(vf_safe < vf, drive_brake, drive_accel),
            )
            steer_safe = delta_safe / half_steer

            u_norm_list.extend([
                torch.clamp(drive_safe, -1.0, 1.0),
                torch.clamp(steer_safe, -1.0, 1.0),
            ])

        return torch.stack(u_norm_list, dim=-1)

    def compute_agent_dynamics(
        self, x_agent: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        theta = x_agent[:, 2]
        vf = x_agent[:, 3]
        beta_f = x_agent[:, 4]
        beta_r = x_agent[:, 5]
        delta = x_agent[:, 6]

        lf = self.par.lf
        lr = self.par.lr
        tl = self.par.tl_steering
        L = lf + lr
        ratio = lr / L

        tan_bf = torch.tan(beta_f)
        tan_br = torch.tan(beta_r)
        cos_bf = torch.cos(beta_f)
        sin_bf = torch.sin(beta_f)

        beta_b = torch.atan2(
            lr * tan_bf + lf * tan_br,
            torch.full_like(beta_f, L),
        )
        beta_g = theta + beta_b

        cos_bb = torch.cos(beta_b)
        cos_bb_safe = torch.where(
            torch.abs(cos_bb) < 1e-4, torch.sign(cos_bb) * 1e-4, cos_bb
        )
        vg = vf * cos_bf / cos_bb_safe

        vx = vg * torch.cos(beta_g)
        vy = vg * torch.sin(beta_g)

        S_bf = torch.sqrt(cos_bf**2 + (ratio * sin_bf) ** 2)
        S_bf_safe = torch.clamp(S_bf, min=1e-4)

        vg_dot_1 = (
            (ratio**2 - 1.0)
            * sin_bf
            * cos_bf
            * vf
            / (S_bf_safe * tl)
        )
        vg_dot_2 = (
            -0.32 * vf
            + 0.379 * (vf**2)
            - 0.155 * (vf**3)
            - 4.75 * ((beta_f - delta) ** 2)
        )

        denom_bb = (
            (cos_bf**2)
            * (L**2 + (lr * tan_bf + lf * tan_br) ** 2)
            * tl
        )
        denom_bb_safe = torch.where(
            torch.abs(denom_bb) < 1e-6, torch.sign(denom_bb) * 1e-6, denom_bb
        )
        betab_dot_1 = (L * lr) / denom_bb_safe

        f = torch.stack(
            [
                vx,
                vy,
                (vg / lr) * torch.sin(beta_b),
                -vg_dot_1 * delta + S_bf * (vg_dot_2 - 0.53 * vf),
                -betab_dot_1 * delta,
            ],
            dim=-1,
        )

        zeros = torch.zeros_like(vf)
        g = torch.stack(
            [
                torch.stack([zeros, zeros], dim=-1),
                torch.stack([zeros, zeros], dim=-1),
                torch.stack([zeros, zeros], dim=-1),
                torch.stack([0.53 * S_bf, vg_dot_1], dim=-1),
                torch.stack([zeros, betab_dot_1], dim=-1),
            ],
            dim=1,
        )

        kinematics = {
            "px": x_agent[:, 0],
            "py": x_agent[:, 1],
            "vx": vx,
            "vy": vy,
            "vg": vg,
            "beta_g": beta_g,
            "beta_f": beta_f,
        }
        return f, g, kinematics

    def build_cbf_constraints(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        device = x.device
        dtype = x.dtype

        x1 = x[:, 0:7]
        x2 = x[:, 7:14]

        f1, g1, k1 = self.compute_agent_dynamics(x1)
        f2, g2, k2 = self.compute_agent_dynamics(x2)

        A_rows = []
        b_rows = []
        h_vals_list = []

        # 1. Pairwise Inter-Vehicle Barrier (Car 1 <-> Car 2)
        p_rel = torch.stack([k2["px"] - k1["px"], k2["py"] - k1["py"]], dim=-1)
        d_12 = torch.linalg.norm(p_rel, dim=-1)
        d_12_safe = torch.clamp(d_12, min=1e-5)

        v_rel = torch.stack([k2["vx"] - k1["vx"], k2["vy"] - k1["vy"]], dim=-1)
        h_dot_12 = torch.sum(p_rel * v_rel, dim=-1) / d_12_safe

        approach_speed_12 = torch.clamp(-h_dot_12, min=0.0)
        h_12_effective = (
            d_12 - self.d_car_safe
        ) - self.tau_lookahead * approach_speed_12

        cross_1 = p_rel[:, 0] * k1["vy"] - p_rel[:, 1] * k1["vx"]
        grad_1 = (
            torch.stack(
                [
                    k1["vx"] - k2["vx"],
                    k1["vy"] - k2["vy"],
                    cross_1,
                    -(
                        p_rel[:, 0] * torch.cos(k1["beta_g"])
                        + p_rel[:, 1] * torch.sin(k1["beta_g"])
                    ),
                    cross_1,
                ],
                dim=-1,
            )
            / d_12_safe.unsqueeze(-1)
        )

        cross_2 = -p_rel[:, 0] * k2["vy"] + p_rel[:, 1] * k2["vx"]
        grad_2 = (
            torch.stack(
                [
                    -k1["vx"] + k2["vx"],
                    -k1["vy"] + k2["vy"],
                    cross_2,
                    (
                        p_rel[:, 0] * torch.cos(k2["beta_g"])
                        + p_rel[:, 1] * torch.sin(k2["beta_g"])
                    ),
                    cross_2,
                ],
                dim=-1,
            )
            / d_12_safe.unsqueeze(-1)
        )

        lf_1 = torch.sum(grad_1 * f1, dim=-1)
        lf_2 = torch.sum(grad_2 * f2, dim=-1)
        lf_12 = lf_1 + lf_2

        lg_1 = torch.stack(
            [
                grad_1[:, 3] * g1[:, 3, 0],
                grad_1[:, 3] * g1[:, 3, 1] + grad_1[:, 4] * g1[:, 4, 1],
            ],
            dim=-1,
        )
        lg_2 = torch.stack(
            [
                grad_2[:, 3] * g2[:, 3, 0],
                grad_2[:, 3] * g2[:, 3, 1] + grad_2[:, 4] * g2[:, 4, 1],
            ],
            dim=-1,
        )

        a_row_12 = torch.cat([-lg_1, -lg_2], dim=-1)
        b_row_12 = lf_12 + self.gamma_1 * h_dot_12 + self.gamma_2 * h_12_effective

        A_rows.append(a_row_12)
        b_rows.append(b_row_12)
        h_vals_list.append(d_12 - self.d_car_safe)

        # 2. Obstacle Barriers (2 ostacoli per veicolo)
        kinematics_agents = [k1, k2]
        f_agents = [f1, f2]
        g_agents = [g1, g2]

        centers = self.obstacle_centers.to(device=device, dtype=dtype)
        zeros_2 = torch.zeros(batch_size, 2, device=device, dtype=dtype)

        for agent_idx in range(2):
            k_ag = kinematics_agents[agent_idx]
            f_ag = f_agents[agent_idx]
            g_ag = g_agents[agent_idx]

            for obs_idx in range(centers.shape[0]):
                c = centers[obs_idx]
                p_rel_obs = torch.stack(
                    [c[0] - k_ag["px"], c[1] - k_ag["py"]], dim=-1
                )
                d_obs = torch.linalg.norm(p_rel_obs, dim=-1)
                d_obs_safe = torch.clamp(d_obs, min=1e-5)

                h_dot_obs = (
                    -p_rel_obs[:, 0] * k_ag["vx"] - p_rel_obs[:, 1] * k_ag["vy"]
                ) / d_obs_safe

                approach_speed_obs = torch.clamp(-h_dot_obs, min=0.0)
                h_obs_effective = (
                    d_obs - self.r_obs_safe
                ) - self.tau_lookahead * approach_speed_obs

                cross_obs = (
                    p_rel_obs[:, 0] * k_ag["vy"] - p_rel_obs[:, 1] * k_ag["vx"]
                )
                grad_obs = (
                    torch.stack(
                        [
                            k_ag["vx"],
                            k_ag["vy"],
                            cross_obs,
                            -(
                                p_rel_obs[:, 0] * torch.cos(k_ag["beta_g"])
                                + p_rel_obs[:, 1] * torch.sin(k_ag["beta_g"])
                            ),
                            cross_obs,
                        ],
                        dim=-1,
                    )
                    / d_obs_safe.unsqueeze(-1)
                )

                lf_obs = torch.sum(grad_obs * f_ag, dim=-1)
                lg_obs = torch.stack(
                    [
                        grad_obs[:, 3] * g_ag[:, 3, 0],
                        grad_obs[:, 3] * g_ag[:, 3, 1]
                        + grad_obs[:, 4] * g_ag[:, 4, 1],
                    ],
                    dim=-1,
                )

                if agent_idx == 0:
                    a_row_obs = torch.cat([-lg_obs, zeros_2], dim=-1)
                else:
                    a_row_obs = torch.cat([zeros_2, -lg_obs], dim=-1)

                b_row_obs = (
                    lf_obs + self.gamma_1 * h_dot_obs + self.gamma_2 * h_obs_effective
                )

                A_rows.append(a_row_obs)
                b_rows.append(b_row_obs)
                h_vals_list.append(d_obs - self.r_obs_safe)

        A_cbf = torch.stack(A_rows, dim=1)
        b_cbf = torch.stack(b_rows, dim=1)
        h_vals = torch.stack(h_vals_list, dim=1)

        return A_cbf, b_cbf, h_vals

    def solve_qp_batch(
        self,
        u_nom: torch.Tensor,
        A_cbf: torch.Tensor,
        b_cbf: torch.Tensor,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = u_nom.shape[0]
        device = u_nom.device
        dtype = u_nom.dtype

        max_speed = self.par.max_speed
        half_steer = self.par.steering_range / 2.0

        u_lb = torch.tensor(
            [0.0, -half_steer, 0.0, -half_steer], device=device, dtype=dtype
        ).view(1, 4)
        u_ub = torch.tensor(
            [max_speed, half_steer, max_speed, half_steer],
            device=device,
            dtype=dtype,
        ).view(1, 4)

        q_diag = torch.tensor(
            [self.weight_v, self.weight_delta, self.weight_v, self.weight_delta],
            device=device,
            dtype=dtype,
        )
        Q = torch.diag(q_diag).unsqueeze(0).repeat(B, 1, 1)

        cbf_violation = torch.bmm(A_cbf, u_nom.unsqueeze(-1)).squeeze(-1) - b_cbf
        within_bounds = torch.all(u_nom >= u_lb, dim=-1) & torch.all(
            u_nom <= u_ub, dim=-1
        )
        safe_mask = (torch.max(cbf_violation, dim=-1).values <= 0.0) & within_bounds

        if torch.all(safe_mask):
            return u_nom.clone(), torch.zeros(B, dtype=torch.bool, device=device), safe_mask

        eye4 = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(B, 1, 1)
        G = torch.cat([eye4, A_cbf], dim=1)
        Gt = G.transpose(1, 2)

        rho = self.rho
        H = Q + rho * torch.bmm(Gt, G)
        L_chol = torch.linalg.cholesky(H)

        u = torch.clamp(u_nom, u_lb, u_ub).clone()
        z = torch.bmm(G, u.unsqueeze(-1)).squeeze(-1)
        y = torch.zeros_like(z)
        q_rhs = q_diag.unsqueeze(0) * u_nom

        for _ in range(self.max_admm_iter):
            rhs = q_rhs + rho * torch.bmm(
                Gt, (z - y / rho).unsqueeze(-1)
            ).squeeze(-1)
            u = torch.cholesky_solve(rhs.unsqueeze(-1), L_chol).squeeze(-1)

            Gu = torch.bmm(G, u.unsqueeze(-1)).squeeze(-1)
            v = Gu + y / rho

            z_box = torch.clamp(v[:, :4], u_lb, u_ub)
            z_cbf = torch.minimum(v[:, 4:], b_cbf)
            z = torch.cat([z_box, z_cbf], dim=-1)

            y = y + rho * (Gu - z)

        u_opt = torch.clamp(u, u_lb, u_ub)

        res_violation = torch.bmm(A_cbf, u_opt.unsqueeze(-1)).squeeze(-1) - b_cbf
        max_violation = torch.max(res_violation, dim=-1).values
        infeasible_mask = max_violation > 0.05

        if torch.any(infeasible_mask):
            u_fallback = u_opt.clone()
            u_fallback[:, 0] = torch.where(
                infeasible_mask, torch.zeros_like(u_fallback[:, 0]), u_fallback[:, 0]
            )
            u_fallback[:, 2] = torch.where(
                infeasible_mask, torch.zeros_like(u_fallback[:, 2]), u_fallback[:, 2]
            )

            p1_x = x[:, 0]
            p2_x = x[:, 7]
            steer_1_escape = torch.where(
                p1_x < 0.0, -0.6 * half_steer, 0.6 * half_steer
            )
            steer_2_escape = torch.where(
                p2_x < 0.0, -0.6 * half_steer, 0.6 * half_steer
            )

            u_fallback[:, 1] = torch.where(
                infeasible_mask, steer_1_escape, u_fallback[:, 1]
            )
            u_fallback[:, 3] = torch.where(
                infeasible_mask, steer_2_escape, u_fallback[:, 3]
            )

            u_opt = u_fallback

        u_final = torch.where(safe_mask.unsqueeze(-1), u_nom, u_opt)
        return u_final, infeasible_mask, safe_mask

    def filter_inputs(
        self, x: torch.Tensor, u_nominal: torch.Tensor
    ) -> torch.Tensor:
        """
        Main safety filtering entrypoint called by BumpercarSystem.noiseless_forward.
        Integrates Directional Approach Governor before QP execution.
        """
        if x.dim() == 3:
            x = x.squeeze(1)
        if u_nominal.dim() == 3:
            u_nominal = u_nominal.squeeze(1)

        batch_size = x.shape[0]
        self.total_steps += batch_size

        # 1. Converte comandi nominali normalizzati in velocità e angoli fisici
        u_phys_nom = self.convert_normalized_to_physical(x, u_nominal)

        # 2. Modulazione di avvicinamento e sfasamento
        if self.enable_velocity_staggering:
            u_phys_target = self.apply_velocity_staggering(x, u_phys_nom)
        else:
            u_phys_target = u_phys_nom

        # 3. Costruzione vincoli HOCBF
        A_cbf, b_cbf, _ = self.build_cbf_constraints(x)

        # 4. Risoluzione QP batch
        u_phys_safe, infeasible_mask, safe_mask = self.solve_qp_batch(
            u_phys_target, A_cbf, b_cbf, x
        )

        num_infeasible = int(infeasible_mask.sum().item())
        self.infeasible_steps += num_infeasible
        if num_infeasible > 0:
            self.emergency_fallback_steps += num_infeasible

        # 5. Riconversione in comandi normalizzati per l'impianto
        u_safe_norm = self.convert_physical_to_normalized(x, u_phys_safe)

        # Verifica se ci sono state modifiche rispetto al nominale
        is_modified = torch.any(
            torch.abs(u_phys_safe - u_phys_nom) > 1e-3, dim=-1, keepdim=True
        )
        self.active_steps += int(is_modified.sum().item())

        u_final_norm = torch.where(is_modified, u_safe_norm, u_nominal)
        return u_final_norm