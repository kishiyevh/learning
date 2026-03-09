# February 21, 2026

## Kalman Filter, Concepts and a 1D Implementation from Scratch

Worked through Roger Labbe's Kalman filter Jupyter book (chapter 1-4) and the Khan Academy probability prereqs. The goal was to understand the predict-update cycle at the math level before looking at any library.

### Idea

The Kalman filter is a recursive estimator. It keeps track of a state estimate and the uncertainty around that estimate. At each step it does two things:

1. **Predict** — use a model of how the state evolves to project the estimate forward in time. Uncertainty grows because the model is imperfect.
2. **Update** — incorporate a new measurement to correct the prediction. Uncertainty shrinks because measurements give information.

The key insight is that both the state and the measurement are modeled as Gaussian distributions. The fusion of two Gaussians is also a Gaussian, and Bayes' theorem tells you exactly how to weight them.

### 1D case, tracking position with a noisy sensor

State: position `x`, velocity `v`. I'll use scalar notation first.

Prediction step (with constant velocity model, dt = 1):

$$\hat{x}_{k|k-1} = \hat{x}_{k-1} + v \cdot \Delta t$$

$$P_{k|k-1} = P_{k-1} + Q$$

Where `P` is the state variance (uncertainty in position) and `Q` is process noise (how much we trust the model).

Update step when a measurement `z_k` arrives:

$$K_k = \frac{P_{k|k-1}}{P_{k|k-1} + R}$$

$$\hat{x}_k = \hat{x}_{k|k-1} + K_k \cdot (z_k - \hat{x}_{k|k-1})$$

$$P_k = (1 - K_k) \cdot P_{k|k-1}$$

`K_k` is the Kalman gain. When measurement noise `R` is large, `K` is small, we trust the model more. When `R` is small, `K` is large, we trust the measurement more.

### The innovation

`z_k - \hat{x}_{k|k-1}` is called the innovation or residual. It's the difference between what the sensor measured and what the filter predicted. If this is consistently large, either your model is wrong or `Q` is too small.

### Extending to multiple dimensions

In the full matrix form, state is a vector `x`, covariance is a matrix `P`, and everything generalizes:

$$\mathbf{x}_{k|k-1} = \mathbf{F} \mathbf{x}_{k-1}$$

$$\mathbf{P}_{k|k-1} = \mathbf{F} \mathbf{P}_{k-1} \mathbf{F}^T + \mathbf{Q}$$

$$\mathbf{K}_k = \mathbf{P}_{k|k-1} \mathbf{H}^T \left(\mathbf{H} \mathbf{P}_{k|k-1} \mathbf{H}^T + \mathbf{R}\right)^{-1}$$

$$\mathbf{x}_k = \mathbf{x}_{k|k-1} + \mathbf{K}_k \left(\mathbf{z}_k - \mathbf{H} \mathbf{x}_{k|k-1}\right)$$

$$\mathbf{P}_k = (\mathbf{I} - \mathbf{K}_k \mathbf{H}) \mathbf{P}_{k|k-1}$$

The `H` matrix maps from state space to measurement space. If you measure position directly, `H = [1, 0]` (picking position out of the [x, v] state vector).

### Why does this matter for drones

PX4's EKF2 is essentially a 23-state Extended Kalman Filter. The same predict-update logic applies but with nonlinear state transitions (quaternion attitude updates) and multiple sensor modalities (IMU, GPS, barometer, magnetometer). The EKF2 fuses all of these to produce the best estimate of the drone's position, velocity, and attitude.

Understanding the 1D case makes the EKF2 docs readable. The matrix dimensions scale, but the logic is identical.

The implementation is in `kalman_1d.py`, it runs a simulated position tracking example and plots the filter output alongside the noisy measurements.

References:

- https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python (chapters 1-4)
- https://www.kalmanfilter.net/default.aspx
- https://docs.px4.io/main/en/advanced_config/tuning_the_ecl_ekf.html
