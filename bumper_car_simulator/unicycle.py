import numpy as np
import matplotlib.pyplot as plt

# Controller gains
k_rho = 1.0
k_alpha = 8.0
k_beta = -10

def wrap_angle(angle):
    """Wrap angle to [-pi, pi]"""
    return (angle + np.pi) % (2 * np.pi) - np.pi

def polar_coordinates(x, y, theta, x_d=0.0, y_d=0.0, theta_d=0.0):
    rho = np.sqrt((x-x_d)**2 + (y-y_d)**2)
    angle_to_target = np.arctan2(y_d-y, x_d-x)
    alpha = wrap_angle(angle_to_target - theta)
    beta = wrap_angle(theta_d - theta - alpha)
    return rho, alpha, beta

def controller(rho, alpha, beta):
    # Control law in polar coordinates
    v = k_rho * rho
    omega = k_alpha*alpha + k_rho*(alpha + k_beta*np.sin(beta*0.99))

    return v, omega

# Initial pose (in Cartesian)
x0, y0, theta0 = 0.0, 0.0, np.deg2rad(0.0)
x_d, y_d, theta_d = 0.0, 5.0, np.deg2rad(-90.0)

# Simulation parameters
dt = 0.01
T = 10.0
N = int(T / dt)

# For plotting
trajectory = [[x0, y0, theta0]]
input_trajectory = [[0.0, 0.0]]

# Initialize theta
x = x0
y = y0
theta = theta0

for _ in range(N):
    # Update polar states
    rho, alpha, beta = polar_coordinates(x, y, theta, x_d, y_d, theta_d)
    v, omega = controller(rho, alpha, beta)
    input_trajectory.append([v, omega])

    # Update cartesian state
    x += dt * v*np.cos(theta)
    y += dt * v*np.sin(theta)
    theta += omega * dt
    theta = wrap_angle(theta)
    trajectory.append([x, y, theta])

print(f"Final position: x={x}, y={y}, theta={np.rad2deg(theta)}")

# Convert trajectory to array
trajectory = np.array(trajectory)
input_trajectory = np.array(input_trajectory)

# Plot the path
fig, (ax_path, ax_input) = plt.subplots(nrows=1, ncols=2 ,figsize=(8, 6),dpi=300)
ax_path.plot(trajectory[:, 0], trajectory[:, 1], label='Unicycle Path')
ax_path.scatter([0], [0], color='red', label='Goal (origin)')
ax_path.set_xlabel('x')
ax_path.set_ylabel('y')
plt.title('Unicycle Stabilization using Polar Coordinates')
ax_path.axis('equal')
ax_path.grid(True)
ax_path.legend()

ax_input.plot(input_trajectory[:, 0], label='Velocity')
ax_input.plot(input_trajectory[:, 1], label='Omega')
ax_input.set_xlabel('t')
plt.title('Unicycle Inputs')
ax_input.grid(True)
ax_input.legend()

plt.show()