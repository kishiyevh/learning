# IMU, GPS, and Barometer

### IMU

An IMU contains an accelerometer and a gyroscope, often with a magnetometer added (then called an AHRS or 9-DOF unit).

The accelerometer measures specific force, the force per unit mass acting on the sensor body, which is the sum of gravity and linear acceleration. In a static unit resting on a table, it reads approximately:

$$a_{measured} = a_{true} + g$$

where `g` points upward in body frame if the IMU is flat. This is the key point that trips people up: an accelerometer does not measure acceleration in the Newtonian sense. It measures the non-gravitational force. A free-falling IMU reads exactly zero.

The gyroscope measures angular velocity about each axis in rad/s. MEMS gyroscopes work by measuring the Coriolis effect on a vibrating resonant mass. The output is very fast and low-noise over short timescales, but it has a bias drift that grows with time. Integrating gyro data to get orientation eventually drifts away from truth — you need magnetometer or vision to correct it.

IMU errors to keep in mind:

- Bias: constant offset, changes with temperature
- Scale factor error: the real sensitivity differs slightly from nominal
- White noise: random, averages out
- Bias instability (flicker noise): slow random walk in bias, does not average out

### GPS

GPS (or GNSS) computes position by measuring time of arrival of signals from multiple satellites. With at least 4 satellites you get 3D position plus clock correction. Each satellite transmits its position and precise timestamp and the receiver measures the signal travel time and computes the pseudorange.

GPS gives position accurate to roughly 1-3 meters with standard L1 civilian signals. Velocity is computed from Doppler shift, which is actually more accurate than differentiating position. Update rate is typically 1-10 Hz.

GPS problems: multipath reflections in urban canyons, signal dropout under dense foliage, poor geometry when satellites are clustered on one side of the sky (high PDOP). In simulation, the GPS plugin in Gazebo models position noise using a Gaussian model with configurable standard deviation.

### Barometer

A barometer measures ambient pressure. Pressure decreases with altitude according to the barometric formula:

$$P(h) = P_0 \cdot \left(1 - \frac{L \cdot h}{T_0}\right)^{\frac{g \cdot M}{R \cdot L}}$$

Where `L` is the temperature lapse rate (0.0065 K/m in the standard atmosphere), `T0` is sea-level temperature, `M` is molar mass of air, and `R` is the gas constant.

In practice, the simpler approximation works fine for low altitudes:

$$\Delta h \approx -\frac{R \cdot T}{g \cdot M} \ln\left(\frac{P}{P_0}\right)$$

Barometer altitude has an absolute error of ±5-10 m depending on weather, but relative altitude changes within a flight are accurate to about 0.1-0.2 m. This is why EKF2 uses it for height hold, it's consistent even if the absolute value drifts.

### How these sensors combine

No single sensor gives us full state. GPS gives position but at low rate with noise. IMU gives high-rate attitude and acceleration but drifts. Barometer gives altitude but slowly and with bias. The whole point of the Kalman filter (coming up in a few days) is to fuse these in a statistically optimal way.

The PX4 EKF2 state vector is 23 states: position (3), velocity (3), orientation quaternion (4), IMU delta-angle bias (3), IMU delta-velocity bias (3), magnetic field body (3), magnetic field earth (2), wind velocity (2).

References:

- Woodman, O. "An Introduction to Inertial Navigation" — Cambridge TR 696
- https://docs.px4.io/main/en/advanced_config/tuning_the_ecl_ekf.html
- "Inertial Navigation Explained" by Starlino (YouTube)
