# February 22, 2026

## Betaflight vs PX4 Flight Control Architecture Differences

Spent the day in the Betaflight Configurator source and comparing it against PX4's control architecture. They solve the same base problem (keep a multirotor stable) but with very different priorities and design decisions.

### Who they are designed for

Betaflight is built for FPV racing and freestyle flying. The priority is raw performance, minimal latency between stick input and motor output, maximum responsiveness, the ability to do flips at full throttle. It does not have autonomous flight, GPS position hold in the traditional sense, or a full navigation stack.

PX4 is designed for autonomous UAVs — missions, waypoints, position hold, return-to-home, payload delivery. It has a full estimation-navigation-control pipeline. The raw performance in manual mode is decent but not the focus.

### Control loop structure

Betaflight runs its entire PID loop at up to 8 kHz. The gyro is sampled at 8 kHz, filtered, and fed into three independent PID controllers for roll, pitch, and yaw. The output goes directly to the ESC mixer with essentially zero abstraction layers.

```
Gyro (8kHz) --> Filters --> PID (roll/pitch/yaw) --> Mixer --> Motor outputs
                ^
            RC input
```

PX4 has a more layered architecture:

```
Mission/Offboard
    |
Navigator (position setpoints)
    |
Position Controller (velocity setpoints)   [~50 Hz]
    |
Velocity Controller (attitude setpoints)   [~50 Hz]
    |
Attitude Controller (rate setpoints)       [~250 Hz]
    |
Rate Controller (motor commands)           [~1000 Hz]
    |
Mixer --> ESCs
```

Each layer runs at a different rate. The inner rate controller runs fastest. This cascaded structure gives cleaner separation of concerns but adds latency between a position command and a motor response.

### PID in Betaflight

In Betaflight, `P` on roll means: if the gyro shows a roll rate error (stick wants 200 deg/s, gyro reads 180 deg/s), apply proportional correction. This is **rate mode** PID by default (ACRO mode). There is no absolute angle reference in rate mode.

In Angle mode (ANGLE or HORIZON in Betaflight), there is an outer loop that converts desired angle to a desired rate setpoint, then feeds the inner rate PID. This is similar to PX4's attitude controller structure, but simplified.

### Betaflight's RPM filtering and notch filters

One thing Betaflight does that PX4 does not natively have is RPM-based dynamic notch filtering. Each ESC reports its motor RPM via bidirectional DSHOT. The filter knows the fundamental frequency of each propeller and creates notch filters at those frequencies (and harmonics) to kill propwash resonance. This is why modern Betaflight quads feel very smooth — the gyro data fed into the PID is extremely clean.

From the Betaflight Configurator source (`src/main/flight/rpm_filter.c`), the notch center frequency for a given motor is:

$$f_{notch} = \frac{RPM}{60}$$

And harmonics at `2f`, `3f`. A Q factor of around 500 is typical, meaning the notch is very narrow.

### PX4 control allocation

PX4 has a control allocation layer (new in v1.14, replacing the old mixer system). Instead of hardcoded mixer files, you define actuator geometry in parameters and the allocator computes motor commands from body-frame forces and moments:

$$\begin{bmatrix} F_z \\ M_x \\ M_y \\ M_z \end{bmatrix} = G \cdot \mathbf{u}$$

Where `G` is the geometry matrix built from motor positions and directions, and `u` is the vector of motor thrusts. The allocator inverts this, with saturation handling for when the desired forces exceed what the actuators can produce.

### Summary comparison

| Property | Betaflight | PX4 |
|---|---|---|
| Loop rate | up to 8 kHz | 1 kHz inner, 50 Hz outer |
| Autonomy | None (GPS rescue only) | Full navigation stack |
| State estimation | Simple complementary filter | EKF2 (23-state) |
| Target users | FPV pilots | Researchers, commercial drones |
| Latency (stick to motor) | ~1 ms | ~10 ms |
| RPM filtering | Yes (bidirectional DSHOT) | No |

References:

- https://github.com/betaflight/betaflight-configurator
- https://github.com/betaflight/betaflight (firmware source)
- https://docs.px4.io/main/en/flight_stack/controller_diagrams.html
- https://docs.px4.io/main/en/concept/control_allocation.html
