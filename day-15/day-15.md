# March 2, 2026

## Extended Kalman Filter for Visual-Inertial Odometry

Today went deeper into the EKF for nonlinear systems — specifically oriented toward visual-inertial state estimation. The 1D filter from February 21 assumed linear dynamics. Real drone state estimation involves attitude quaternions, which require an Extended KF.

### The Standard KF 

The standard Kalman filter assumes linear dynamics: `x_{k+1} = F x_k + w`. This works for position and velocity (nearly linear under small motions). Attitude is different. Quaternions have 4 components but only 3 degrees of freedom (the unit norm constraint removes one). The update equations are nonlinear — rotating a body by omega for time dt is:

$$\mathbf{q}_{k+1} = \mathbf{q}_k \otimes \delta\mathbf{q}(\omega, \Delta t)$$

where `delta_q` is the quaternion corresponding to a small rotation. You cannot directly apply the standard KF update here.

### The EKF linearization

The EKF approximates the nonlinear dynamics by their first-order Taylor expansion (Jacobian) at the current estimate:

$$\mathbf{F}_k = \frac{\partial f}{\partial \mathbf{x}} \bigg|_{\hat{\mathbf{x}}_k}$$

The prediction step uses the full nonlinear dynamics:

$$\hat{\mathbf{x}}_{k|k-1} = f(\hat{\mathbf{x}}_{k-1})$$

But the covariance prediction uses the linearized Jacobian:

$$\mathbf{P}_{k|k-1} = \mathbf{F}_k \mathbf{P}_{k-1} \mathbf{F}_k^T + \mathbf{Q}$$

This is valid when the nonlinearity is mild over the prediction step — which holds if the IMU is fast (1 kHz) and the motion is not extreme.

### Error-state (indirect) EKF

The EKF2 in PX4 and most practical VIO systems use the error-state formulation (also called indirect EKF or multiplicative EKF for attitude). Instead of estimating the full state directly, the filter estimates the error between the true state and the nominal state:

$$\delta \mathbf{x} = \mathbf{x}_{true} - \hat{\mathbf{x}}_{nominal}$$

The nominal state is propagated by integrating the IMU at full rate (no uncertainty, just the best estimate). The error state KF runs slower, accumulates covariance from IMU noise, and applies corrections when measurements arrive.

For attitude, the error state is a 3-element rotation vector (small angle) rather than a quaternion. After the KF update computes the error-state correction, it is "injected" back into the quaternion via:

$$\mathbf{q} \leftarrow \mathbf{q} \otimes \delta\mathbf{q}(\delta\theta)$$

This avoids the quaternion normalization problem and keeps the covariance matrix at 15x15 (not 16x16) — the rotation vector has 3 components, not 4.

### Measurement updates for visual odometry

When a visual odometry front-end computes a relative pose between two frames:

$$\mathbf{z} = \begin{bmatrix} \delta \mathbf{p}^{cam} \\ \delta \mathbf{q}^{cam} \end{bmatrix}$$

The measurement model relates this to the current state estimate. The position part is straightforward (difference of positions in world frame, rotated into camera frame). The rotation part is again a quaternion composition.

The H matrix (linearized measurement Jacobian) relates the error state to the measurement innovation. For position measurements:

$$\mathbf{H}_{pos} = \begin{bmatrix} \mathbf{R}^T & \cdots \end{bmatrix}$$

where R is the body-to-world rotation.

### VNAV lectures on this topic

The VNAV course (Lecture 8-9) covers the full derivation of the IMU preintegration factor, which is the batch optimization equivalent of the EKF prediction step. IMU preintegration accumulates the IMU measurements between two keyframes into a single relative motion constraint that is independent of the integration starting point. This is used in factor graph optimizers (GTSAM, g2o) as an alternative to the recursive EKF.

For the EKF path: PX4's ECL EKF is the most battle-tested open implementation. For the factor graph path: OpenVINS is a good open-source reference that implements exactly this preintegration + EKF approach.

The script `error_state_ekf.py` implements a simplified 2D version — position + orientation — to make the error-state concept concrete without the full 15-state complexity.

References:

- Joan Sola "A micro Lie theory for state estimation in robotics" (arXiv:1812.01537) — best reference for quaternion EKF
- https://github.com/rpng/open_vins — OpenVINS codebase
- https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python (Chapter 11)
- https://vnav.mit.edu/lectures.html (Lectures 8-9)
