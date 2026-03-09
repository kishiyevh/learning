"""
Author: Huseyn Kishiyev
Discrete-time PID controller controlling a simple first-order system.
System model: y[k+1] = y[k] + (u[k] - b*y[k]) * dt
  where b is a drag/damping coefficient.

Plots the step response for given gains.

Usage:
  python3 pid_sim.py

Requires:
  pip install matplotlib numpy
"""

import numpy as np
import matplotlib.pyplot as plt


class PIDController:
    def __init__(self, kp: float, ki: float, kd: float,
                 output_min: float = -1.0, output_max: float = 1.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_min = output_min
        self.output_max = output_max

        self._integral = 0.0
        self._prev_error = 0.0

    def reset(self):
        self._integral = 0.0
        self._prev_error = 0.0

    def compute(self, setpoint: float, measurement: float, dt: float) -> float:
        error = setpoint - measurement

        # Proportional
        p = self.kp * error

        # Integral with anti-windup clamp
        self._integral += error * dt
        i_raw = self.ki * self._integral

        # Derivative (backward difference, no filter for simplicity)
        d = self.kd * (error - self._prev_error) / dt if dt > 0 else 0.0
        self._prev_error = error

        output = p + i_raw + d

        # Clamp output and back-calculate anti-windup
        output_clamped = np.clip(output, self.output_min, self.output_max)
        if self.ki != 0.0:
            # Clamp integral to prevent windup
            self._integral -= (output - output_clamped) / self.ki

        return output_clamped


def simulate(kp, ki, kd, setpoint=1.0, duration=5.0, dt=0.01):
    """
    First-order plant: dx/dt = u - b*x
    Discretized: x[k+1] = x[k] + (u[k] - b*x[k]) * dt
    """
    b = 2.0  # damping / drag

    pid = PIDController(kp=kp, ki=ki, kd=kd, output_min=-5.0, output_max=5.0)

    steps = int(duration / dt)
    time_arr = np.zeros(steps)
    y_arr    = np.zeros(steps)
    u_arr    = np.zeros(steps)
    e_arr    = np.zeros(steps)

    y = 0.0

    for k in range(steps):
        t = k * dt
        u = pid.compute(setpoint, y, dt)

        y = y + (u - b * y) * dt

        time_arr[k] = t
        y_arr[k]    = y
        u_arr[k]    = u
        e_arr[k]    = setpoint - y

    return time_arr, y_arr, u_arr, e_arr


def main():
    configs = [
        # (label, kp, ki, kd)
        ("P only   Kp=1.0", 1.0, 0.0, 0.0),
        ("PI       Kp=1.0 Ki=1.5", 1.0, 1.5, 0.0),
        ("PID      Kp=1.0 Ki=1.5 Kd=0.2", 1.0, 1.5, 0.2),
        ("High Kp  Kp=5.0 Ki=1.5 Kd=0.2", 5.0, 1.5, 0.2),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for ax, (label, kp, ki, kd) in zip(axes, configs):
        t, y, u, e = simulate(kp, ki, kd)
        ax.plot(t, y,              label="Output y(t)", linewidth=2)
        ax.axhline(1.0, color="r", linestyle="--", label="Setpoint", linewidth=1)
        ax.set_title(label)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("y(t)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Annotate steady-state error
        ss_error = abs(1.0 - y[-1])
        ax.text(0.05, 0.1, f"SS error: {ss_error:.4f}",
                transform=ax.transAxes, fontsize=9, color="navy")

    plt.suptitle("PID Step Response — First-Order Plant", fontsize=13)
    plt.tight_layout()
    plt.savefig("pid_step_response.png", dpi=150)
    plt.show()
    print("Saved pid_step_response.png")


if __name__ == "__main__":
    main()
