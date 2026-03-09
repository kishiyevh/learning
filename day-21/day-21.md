# March 8, 2026

## Full State Estimation Loop and Fusing VO with IMU and Reviewing the Stack

The document will be about integrating the visual odometry output with a simple IMU-preintegration-based EKF and doing a full review pass of everything covered since February 16.

### IMU preintegration in brief

Between two keyframes (let's call them at times i and j), the IMU runs at 1 kHz and produces a stream of accelerometer and gyroscope readings. Rather than re-running the EKF prediction for each IMU sample every time the optimizer wants to evaluate a cost, you preintegrate all the IMU samples between i and j into a single relative motion constraint: `(Delta_R, Delta_v, Delta_p)`.

The preintegrated rotation:

$$\Delta R_{ij} = \prod_{k=i}^{j-1} \text{Exp}\left((\tilde{\omega}_k - b_g) \Delta t\right)$$

The preintegrated velocity and position:

$$\Delta v_{ij} = \sum_{k=i}^{j-1} \Delta R_{ik} (\tilde{a}_k - b_a) \Delta t$$

$$\Delta p_{ij} = \sum_{k=i}^{j-1} \left[\Delta v_{ik} \Delta t + \frac{1}{2} \Delta R_{ik} (\tilde{a}_k - b_a) \Delta t^2\right]$$

Here `b_g` and `b_a` are gyro and accel biases, and `Exp(.)` is the exponential map from the Lie algebra so(3) to SO(3) (converts rotation vector to rotation matrix).

The important thing is that these integrals do not depend on the starting pose at frame i. This means if you change the pose estimate at i (during optimization), we do not have to redo the integration we just recompute the residual against the fixed preintegrated measurement. This is why preintegration fits naturally into factor graph optimizers.

### EKF fusion of VO and IMU

Without a full factor graph optimizer (GTSAM, g2o), you can fuse VO and IMU in an EKF:

- **IMU prediction**: propagate state at 1 kHz using accel and gyro
- **VO update**: when a new keyframe pair gives a relative pose (R_vo, t_vo), use it as a measurement to correct position, velocity, and attitude

The VO measurement model for position (assuming VO gives relative position in body frame):

$$\mathbf{z}_{VO} = \mathbf{R}_{world}^T (\mathbf{p}_{k} - \mathbf{p}_{k-1}) + \mathbf{v}_{VO}$$

where `v_VO` is the VO position noise (typically 0.05-0.1 m per keyframe step).

In PX4 this is exactly what happens when you pipe external vision data into EKF2, the VO node publishes `vehicle_visual_odometry` messages and EKF2 treats the position component as a direct state measurement.

### What the full stack would look like (presumably) on a drone

```
Hardware layer:
  IMU (ICM-42688, 8 kHz) -> PX4 uORB -> EKF2 prediction
  GPS (u-blox M9N, 5 Hz) -> PX4 uORB -> EKF2 GPS fusion
  Barometer (MS5611)       -> PX4 uORB -> EKF2 baro fusion
  Camera (global shutter)  -> ROS2 image topic -> VO node

VO node (runs on companion computer, e.g. Raspberry Pi 5, Mango Pi or Jetson Nano):
  SuperPoint + LightGlue (ONNX, GPU) -> Essential matrix -> Relative pose
  Publishes /fmu/in/vehicle_visual_odometry

EKF2 (runs on flight controller, Pixhawk 6C):
  Fuses IMU + GPS + baro + external vision
  Publishes vehicle_odometry (position, velocity, attitude)

Position controller -> Velocity controller -> Rate controller -> ESCs
```

The companion computer talks to the flight controller over a UART or USB serial link via MAVLink/uXRCE-DDS.

### Latency budget

The control loop needs state estimates at >=100 Hz (10 ms budget). The IMU-driven EKF2 runs at this rate internally. The VO update arrives at ~10-20 Hz (50-100 ms) and simply corrects the running estimate without blocking the fast inner loop. This cascaded rate structure is standard in all commercial autopilots.

### Outcomes

The biggest gaps in the current understanding,

1. Full factor graph optimization (GTSAM) — the EKF approximation accumulates linearization errors. Factor graphs with iterative optimization (iSAM2) give better consistency.

2. Place recognition / loop closure — the drift fix. Requires image retrieval (DBoW3 or NetVLAD) + geometric verification + pose graph optimization.

3. Hardware integration — running the ONNX pipeline on a real Jetson with TensorRT, dealing with USB camera latency, hardware trigger synchronization between camera and IMU (crucial for VI-SLAM).

4. PX4 EKF2 parameter tuning on real hardware — the SITL noise parameters are too clean. Real IMUs have temperature drift, vibration bias, and magnetic interference that require careful calibration.

The C++ code in `imu_preintegration.cpp` implements the preintegration equations above with linearized covariance propagation, and `ekf_vo_imu.cpp` shows the EKF fusion loop connecting VO updates into the IMU prediction chain.

References:

- Forster et al. "On-Manifold Preintegration for Real-Time Visual-Inertial Odometry" (TRO 2017) — arXiv:1512.02363
- https://github.com/rpng/open_vins
- https://docs.px4.io/main/en/ros/external_position_estimation.html
- Joan Sola "A micro Lie theory for state estimation in robotics" — arXiv:1812.01537
- https://vnav.mit.edu/lectures.html (Lectures 10-12)
