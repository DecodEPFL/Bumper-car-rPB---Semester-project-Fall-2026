# rPB Controller Robustness & Out-Of-Distribution (OOD) Benchmark

This benchmark evaluates the nominal performance and out-of-distribution (OOD) robustness of the official trained rPB checkpoint (`rPB_best.pt`) in closed-loop simulations using the learned neural dynamics (`model_kinematic_mlp.pth`).

## 1. Experimental Setup & Metrics

- **Environment**: Two-car reference tracking in the 10 × 10 m arena with crossing paths.
- **Safety Specification**: Minimum inter-agent safety distance $d_{\text{min}} = 2.0\text{ m}$.
- **Obstacle Buffer**: Center-to-vehicle threshold $r_{\text{obs}} = 1.5\text{ m}$.
- **Horizon**: $T = 300$ steps ($\Delta t = 0.04\text{ s}$, total duration $12.0\text{ s}$).
- **Evaluation Tool**: `reference_tracking_PB/benchmark_robustness.py` (executes the official `BumpercarSystem.rollout()` closed-loop pipeline).

---

## 2. Empirical Results (50 Rollouts per Scenario)

| Scenario | Collision Rate ($d < 2.0\text{ m}$) | Absolute Min Distance | Avg Min Distance | Obstacle Hit Rate ($r < 1.5\text{ m}$) | Avg Final Goal Error |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. Nominal (In-Distribution)** | **100.0%** (50/50) | **1.214 m** | 1.214 m | 100.0% | 0.224 m |
| **2. Heading OOD** ($\theta_0 \pm [-\pi/2, \pi/2]$) | **34.0%** (17/50) | **0.935 m** | 2.246 m | 68.0% | 0.418 m |
| **3. Velocity OOD** ($v_{f,0} = 0.8\text{ m/s}$) | **100.0%** (50/50) | **1.999 m** | 1.999 m | 100.0% | 0.222 m |

---

## 3. Analysis & Theoretical Root Cause

1. **Soft Loss vs Hard Invariance**:
   The controller was trained with a soft barrier penalty:
   $$\mathcal{L}_{\text{col}} = \frac{\alpha_{\text{col}}}{d^2 + 10^{-3}}$$
   Because the objective minimizes the aggregate expectation over entire rollouts, gradient descent settles on a compromise: the cars routinely violate the nominal safety margin ($d \approx 1.21\text{ m} < 2.0\text{ m}$) to preserve reference tracking speed.
2. **Physical Boundary Violations under OOD Heading**:
   Under unmodeled initial headings, the distance drops to **0.935 m**. Considering car physical dimensions ($0.55\text{ m} \times 0.32\text{ m}$), this represents a near-physical collision envelope.
3. **Conclusion**:
   Soft loss penalties during training cannot mathematically guarantee state invariance. A dedicated, deterministic **Control Barrier Function (CBF-QP) safety filter** is necessary at runtime to guarantee $d(t) \ge d_{\text{min}}$ across all operating regimes.