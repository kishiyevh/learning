"""
Author: Huseyn Kishiyev
------------------
2D error-state (indirect) EKF for a ground robot.
State: [x, y, theta] (position + heading)
Error state: [dx, dy, dtheta]

IMU: provides linear velocity u and angular velocity omega (noisy)
GPS: provides noisy position measurements at 2 Hz

Demonstrates:
  1. Nominal state propagation (pure integration of IMU)
  2. Error-state KF accumulates covariance
  3. GPS update corrects both nominal state and error state
  4. Plot shows nominal vs corrected estimate vs truth

Requires: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt


class ErrorStateEKF2D:
    """
    2D error-state EKF.
    Nominal state: [x, y, theta]
    Error state:   [dx, dy, dtheta]  (3D)
    """

    def __init__(self, Q_vel: float, Q_omega: float, R_gps: float):
        """
        Q_vel:   process noise std for linear velocity
        Q_omega: process noise std for angular velocity
        R_gps:   GPS position noise std (m)
        """
        self.Q_vel   = Q_vel
        self.Q_omega = Q_omega
        self.R_gps   = R_gps

        # Nominal state
        self.x_nom = np.zeros(3)   # [x, y, theta]

        # Error state covariance (3x3)
        self.P = np.diag([1.0, 1.0, 0.1])   # initial uncertainty

        # H matrix: GPS measures x and y (not theta)
        self.H = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])

        self.R = np.eye(2) * R_gps**2

    def predict(self, u_meas: float, omega_meas: float, dt: float):
        """
        Propagate nominal state with IMU measurement.
        Build linearized Jacobian F for error-state covariance update.

        u_meas:     measured forward speed (noisy)
        omega_meas: measured angular rate (noisy)
        """
        theta = self.x_nom[2]
        u_true = u_meas    # ignore noise for nominal propagation
        w_true = omega_meas

        # Nominal state update (nonlinear)
        self.x_nom[0] += u_true * np.cos(theta) * dt
        self.x_nom[1] += u_true * np.sin(theta) * dt
        self.x_nom[2] += w_true * dt

        # Linearized Jacobian of f w.r.t. error state
        # f(x + dx) approx f(x) + F dx
        F = np.array([
            [1.0, 0.0, -u_true * np.sin(theta) * dt],
            [0.0, 1.0,  u_true * np.cos(theta) * dt],
            [0.0, 0.0, 1.0],
        ])

        # Process noise input matrix G (maps noise on u, omega to state)
        G = np.array([
            [np.cos(theta) * dt, 0.0],
            [np.sin(theta) * dt, 0.0],
            [0.0,                dt ],
        ])
        Q = np.diag([self.Q_vel**2, self.Q_omega**2])

        self.P = F @ self.P @ F.T + G @ Q @ G.T

    def update_gps(self, z: np.ndarray):
        """
        GPS measurement z = [x_gps, y_gps] (noisy position).
        """
        # Innovation: difference between measurement and predicted measurement
        y = z - self.H @ self.x_nom   # (2,)

        S = self.H @ self.P @ self.H.T + self.R   # (2,2)
        K = self.P @ self.H.T @ np.linalg.inv(S)  # (3,2)

        # Error state correction
        dx = K @ y    # (3,)

        # Apply correction to nominal state
        self.x_nom += dx

        # Update covariance
        self.P = (np.eye(3) - K @ self.H) @ self.P


def simulate():
    np.random.seed(7)
    dt         = 0.02    # 50 Hz IMU
    gps_every  = 25      # GPS at 2 Hz (every 25 IMU steps)
    total_steps= 1000

    # Process and measurement noise
    Q_vel   = 0.05   # m/s noise
    Q_omega = 0.02   # rad/s noise
    R_gps   = 2.0    # m GPS noise

    ekf = ErrorStateEKF2D(Q_vel, Q_omega, R_gps)

    # True state
    x_true = np.array([0.0, 0.0, 0.0])

    # Storage
    true_traj = [x_true.copy()]
    est_traj  = [ekf.x_nom.copy()]
    nom_traj  = [ekf.x_nom.copy()]  # without corrections
    gps_meas  = []

    x_nom_no_correction = x_true.copy()  # track nominal-only (no GPS)

    times = [0.0]

    for k in range(1, total_steps + 1):
        t = k * dt

        # True commanded inputs (figure-eight-ish path)
        u_cmd = 1.0
        omega_cmd = 0.5 * np.sin(t * 0.5)

        # Add noise for "IMU" measurement
        u_meas     = u_cmd     + np.random.randn() * Q_vel
        omega_meas = omega_cmd + np.random.randn() * Q_omega

        # Propagate true state with true inputs + small extra noise
        theta = x_true[2]
        x_true[0] += u_cmd * np.cos(theta) * dt + np.random.randn() * 0.003
        x_true[1] += u_cmd * np.sin(theta) * dt + np.random.randn() * 0.003
        x_true[2] += omega_cmd * dt               + np.random.randn() * 0.001

        # Nominal-only propagation (no corrections, just for comparison)
        tn = x_nom_no_correction[2]
        x_nom_no_correction[0] += u_meas * np.cos(tn) * dt
        x_nom_no_correction[1] += u_meas * np.sin(tn) * dt
        x_nom_no_correction[2] += omega_meas * dt

        # EKF predict
        ekf.predict(u_meas, omega_meas, dt)

        # GPS update every gps_every steps
        if k % gps_every == 0:
            z = x_true[:2] + np.random.randn(2) * R_gps
            ekf.update_gps(z)
            gps_meas.append((t, z.copy()))

        true_traj.append(x_true.copy())
        est_traj.append(ekf.x_nom.copy())
        nom_traj.append(x_nom_no_correction.copy())
        times.append(t)

    true_traj = np.array(true_traj)
    est_traj  = np.array(est_traj)
    nom_traj  = np.array(nom_traj)

    # Compute errors
    err_ekf = np.linalg.norm(true_traj[:, :2] - est_traj[:, :2], axis=1)
    err_nom = np.linalg.norm(true_traj[:, :2] - nom_traj[:, :2],  axis=1)

    print(f"Mean position error — EKF:     {err_ekf.mean():.3f} m")
    print(f"Mean position error — IMU-only: {err_nom.mean():.3f} m")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.plot(true_traj[:, 0], true_traj[:, 1], "g-",  lw=2, label="Ground truth")
    ax.plot(est_traj[:, 0],  est_traj[:, 1],  "b-",  lw=1.5, label="EKF estimate")
    ax.plot(nom_traj[:, 0],  nom_traj[:, 1],  "r--", lw=1, label="IMU only (no GPS)")
    if gps_meas:
        gx = [g[1][0] for g in gps_meas]
        gy = [g[1][1] for g in gps_meas]
        ax.scatter(gx, gy, s=20, c="orange", zorder=5, label="GPS measurements")
    ax.set_title("2D Trajectory")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    ax = axes[1]
    ax.plot(times, err_ekf, "b-",  label=f"EKF error (mean={err_ekf.mean():.2f}m)")
    ax.plot(times, err_nom,  "r--", label=f"IMU only error (mean={err_nom.mean():.2f}m)")
    ax.set_title("Position Error Over Time")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Error (m)")
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.suptitle("2D Error-State EKF with GPS Updates", fontsize=13)
    plt.tight_layout()
    plt.savefig("error_state_ekf_result.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    simulate()
