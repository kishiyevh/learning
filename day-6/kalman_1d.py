"""
Author: Huseyn Kishiyev
1D Kalman filter tracking position + velocity.
State: [position, velocity]
Measurement: position only (noisy GPS-like sensor)

Demonstrates the predict-update cycle without any library.
Plots: true position, noisy measurements, filter estimate,
and the 1-sigma confidence band from the covariance.

Requires: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt


class KalmanFilter1D:
    """
    2-state (position, velocity) Kalman filter.
    Constant velocity motion model.
    Measurement: position only.
    """

    def __init__(self, dt: float, process_noise_std: float, meas_noise_std: float):
        self.dt = dt

        # State transition matrix (constant velocity model)
        self.F = np.array([
            [1.0, dt],
            [0.0, 1.0],
        ])

        # Measurement matrix (we measure position, not velocity)
        self.H = np.array([[1.0, 0.0]])

        # Process noise covariance (models model uncertainty)
        q = process_noise_std ** 2
        self.Q = q * np.array([
            [dt**4 / 4, dt**3 / 2],
            [dt**3 / 2, dt**2],
        ])

        # Measurement noise covariance
        self.R = np.array([[meas_noise_std ** 2]])

        # Initial state and covariance (uninformed)
        self.x = np.zeros((2, 1))           # [position; velocity]
        self.P = np.eye(2) * 100.0          # large initial uncertainty

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z: float):
        z_vec = np.array([[z]])
        S = self.H @ self.P @ self.H.T + self.R   # innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain
        y = z_vec - self.H @ self.x                # innovation
        self.x = self.x + K @ y
        self.P = (np.eye(2) - K @ self.H) @ self.P

    @property
    def position(self) -> float:
        return float(self.x[0])

    @property
    def velocity(self) -> float:
        return float(self.x[1])

    @property
    def position_std(self) -> float:
        return float(np.sqrt(self.P[0, 0]))


def run_simulation():
    np.random.seed(42)

    dt          = 0.1       # seconds
    total_time  = 30.0      # seconds
    steps       = int(total_time / dt)

    # True motion: accelerate, then cruise, then slow
    true_pos = 0.0
    true_vel = 0.0
    true_positions = []
    measurements   = []
    times          = []

    meas_std    = 3.0   # GPS noise std (m)
    process_std = 0.5   # motion model noise std

    # Generate ground truth with some control input
    for k in range(steps):
        t = k * dt
        # Simple commanded velocity profile
        if t < 5:
            true_vel = 1.0
        elif t < 15:
            true_vel = 3.0
        elif t < 20:
            true_vel = 0.5
        else:
            true_vel = 0.0
        true_pos += true_vel * dt + np.random.randn() * 0.05  # small process noise

        # GPS measurement (5 Hz — only update every 2nd step)
        if k % 2 == 0:
            z = true_pos + np.random.randn() * meas_std
        else:
            z = None

        true_positions.append(true_pos)
        measurements.append(z)
        times.append(t)

    # Run Kalman filter
    kf = KalmanFilter1D(dt=dt,
                        process_noise_std=process_std,
                        meas_noise_std=meas_std)

    kf_pos  = []
    kf_std  = []
    kf_vel  = []
    innovations = []

    for k in range(steps):
        kf.predict()

        z = measurements[k]
        if z is not None:
            innov = z - kf.position
            kf.update(z)
            innovations.append((times[k], innov))

        kf_pos.append(kf.position)
        kf_std.append(kf.position_std)
        kf_vel.append(kf.velocity)

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Position
    ax = axes[0]
    ax.plot(times, true_positions, "g-", label="True position", linewidth=2)
    meas_t = [times[k] for k in range(steps) if measurements[k] is not None]
    meas_v = [measurements[k] for k in range(steps) if measurements[k] is not None]
    ax.scatter(meas_t, meas_v, s=10, color="red", alpha=0.5, label="Measurements (GPS)")
    ax.plot(times, kf_pos, "b-", label="KF estimate", linewidth=2)
    ax.fill_between(
        times,
        [p - s for p, s in zip(kf_pos, kf_std)],
        [p + s for p, s in zip(kf_pos, kf_std)],
        alpha=0.2, color="blue", label="1-sigma bounds"
    )
    ax.set_ylabel("Position (m)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title("1D Kalman Filter — Position Tracking")

    # Velocity
    ax = axes[1]
    ax.plot(times, kf_vel, "b-", label="KF velocity estimate", linewidth=2)
    ax.set_ylabel("Velocity (m/s)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title("Estimated Velocity")

    # Innovation
    ax = axes[2]
    inn_t, inn_v = zip(*innovations) if innovations else ([], [])
    ax.scatter(inn_t, inn_v, s=15, color="purple", alpha=0.7)
    ax.axhline(0, color="k", linewidth=1)
    ax.axhline( meas_std * 2, color="r", linestyle="--", label="+2σ measurement")
    ax.axhline(-meas_std * 2, color="r", linestyle="--", label="-2σ measurement")
    ax.set_ylabel("Innovation (m)")
    ax.set_xlabel("Time (s)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title("Innovation (Measurement Residual)")

    plt.tight_layout()
    plt.savefig("kalman_1d_result.png", dpi=150)
    plt.show()

    # Final stats
    errors = [abs(true_positions[k] - kf_pos[k]) for k in range(steps)]
    print(f"Mean absolute error (KF vs truth): {np.mean(errors):.3f} m")
    print(f"Mean absolute error (raw meas):    {np.mean([abs(true_positions[k] - measurements[k]) for k in range(steps) if measurements[k] is not None]):.3f} m")


if __name__ == "__main__":
    run_simulation()
