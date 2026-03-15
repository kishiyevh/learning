# PX4 EKF2 State Vector, Fusion Logic, and Tuning Parameters

### State vector

EKF2 maintains a 23-element state vector:

```
[0:3]   Position (north, east, down) in local NED frame
[3:6]   Velocity (north, east, down)
[6:10]  Orientation quaternion (qw, qx, qy, qz)
[10:13] IMU delta-angle bias (gyro bias in body frame)
[13:16] IMU delta-velocity bias (accelerometer bias in body frame)
[16:19] Earth magnetic field (NED)
[19:21] Body magnetic field bias (x, y)
[21:23] Wind velocity (north, east)
```

The covariance matrix P is 23x23. This is the matrix that quantifies uncertainty, diagonal entries are variances, off-diagonal entries capture correlations.

### Prediction step

The IMU drives the prediction at ~1 kHz. The equations are nonlinear because attitude is represented as a quaternion and the rotation of the accelerometer measurement involves the current attitude estimate. The EKF2 uses the linearized Jacobian (which is why it is an Extended Kalman Filter, not a standard KF).

The process model for velocity:

$$\mathbf{v}_{k+1} = \mathbf{v}_k + \left(\mathbf{R}_{k} (\mathbf{a}_{body} - \mathbf{b}_a) + \mathbf{g}\right) \Delta t$$

where `R_k` is the rotation matrix from body to NED, `a_body` is the accelerometer reading, `b_a` is the accelerometer bias estimate, and `g` is gravity in NED (0, 0, 9.81).

### Sensor fusion

Each sensor update is processed independently as a "measurement update" step. The GPS gives position and velocity. The barometer gives height. The magnetometer gives heading (processed via a 3D magnetic field model). Each update uses the standard EKF update equations but with sensor-specific noise parameters.

The measurement model for GPS position:

$$\mathbf{z}_{GPS} = \mathbf{H}_{pos} \mathbf{x} + \mathbf{v}_{GPS}$$

Where `H_pos = [I_{3x3}, 0_{3x20}]` just picks out the position states.

### Key parameters for tuning

The parameters I spent the most time on:

`EKF2_GPS_NOISE` — GPS position measurement noise. Default 0.5 m. Increasing this makes the filter trust GPS less and rely more on the IMU model between GPS updates.

`EKF2_BARO_NOISE` — Barometer noise, default 3.5 m. In turbulent air, increase this.

`EKF2_GYR_NOISE`, `EKF2_ACC_NOISE` — Process noise for gyro and accelerometer. These control how quickly the bias estimates can change. Too low and the filter cannot track real bias drift; too high and the estimates become noisy.

`EKF2_GBIAS_INIT` — Initial uncertainty in gyro bias. Set this large if you do not have a prior.

### Innovation monitoring

The EKF2 monitors the ratio of innovation (measurement residual) to expected innovation standard deviation. If this ratio (called the "innovation test ratio") exceeds a threshold, the measurement is rejected as an outlier. This is important for GPS spoofing or multipath rejection.

```
innovation_test_ratio = innovation^2 / (H P H^T + R)
```

If this exceeds `EKF2_INNOV_GATE` (default 5.0 in sigma units squared), the update is skipped.

### Where to find this in code

The main EKF2 loop is in:
```
PX4-Autopilot/src/modules/ekf2/EKF2.cpp
PX4-Autopilot/src/lib/ecl/EKF/ekf.cpp
```

The fusion functions are per-sensor:
```
PX4-Autopilot/src/lib/ecl/EKF/gps_yaw_fusion.cpp
PX4-Autopilot/src/lib/ecl/EKF/baro_height_control.cpp
PX4-Autopilot/src/lib/ecl/EKF/optflow_fusion.cpp
```

Reading `ekf.cpp` is dense but worth it the comments explain the math and the derivation of each Jacobian is spelled out. The code for the quaternion attitude update in `predictState()` matches the theory from Labbe's Kalman book chapter on extended filters.

### EKF2 and a simple complementary filter

The complementary filter that Betaflight uses for attitude is; `attitude = alpha * (attitude + gyro * dt) + (1 - alpha) * accel_angle`. This works for attitude estimation but has no formal uncertainty model, no bias estimation, and no GPS/baro fusion. EKF2 is strictly more correct but also more complex and computationally expensive. On an STM32H7 at 480 MHz, EKF2 is too heavy for the full 1 kHz rate — it runs at 400 Hz with the inner attitude loop running from a simpler complementary filter update.

References:

- https://docs.px4.io/main/en/advanced_config/tuning_the_ecl_ekf.html
- https://github.com/PX4/PX4-Autopilot/tree/main/src/lib/ecl/EKF
- Labbe, R. "Kalman and Bayesian Filters in Python" — Chapter 11 (Extended KF)
