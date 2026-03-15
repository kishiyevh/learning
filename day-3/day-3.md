# PID Controller, Theory and a Working 1D Implementation

The PID controller computes a correction signal based on the error between where you are and where you want to be. The three terms each contribute differently.

The proportional term is just the error scaled by Kp:

$$u_P(t) = K_p \cdot e(t)$$

On its own, a P controller almost always has steady-state error because the correction approaches zero as the error approaches zero — but some error has to remain to keep driving the actuator. The only exception is if there is no load on the system.

The integral term accumulates error over time:

$$u_I(t) = K_i \int_0^t e(\tau) \, d\tau$$

This is what eliminates steady-state error. If there is a constant disturbance (like gravity acting on a drone), the integral winds up until the output is large enough to compensate. The downside is integrator windup — if the actuator saturates (hits its max output), the integral keeps growing because error is still nonzero, and then when the system does start responding it overshoots badly. Anti-windup clamps the integrand.

The derivative term reacts to the rate of change of error:

$$u_D(t) = K_d \cdot \frac{de(t)}{dt}$$

This acts as a brake, it resists fast changes. If the system is approaching the setpoint quickly, the derivative term is large and negative, which slows the approach. In practice the derivative is often filtered because it amplifies high-frequency noise.

Full control output:

$$u(t) = K_p \cdot e(t) + K_i \int_0^t e(\tau) d\tau + K_d \cdot \frac{de(t)}{dt}$$

In discrete time (which is what actually runs on hardware):

$$u[k] = K_p \cdot e[k] + K_i \cdot \sum_{j=0}^{k} e[j] \cdot \Delta t + K_d \cdot \frac{e[k] - e[k-1]}{\Delta t}$$

### Tuning intuition

Start with Ki=0, Kd=0 and increase Kp until you get oscillation, then back off. Then add Ki slowly until steady-state error is gone. Then add Kd to dampen overshoot. This is not Ziegler-Nichols, just intuitive sequential tuning.

For a velocity controller on a robot:
- Kp too high: the robot jerks and oscillates around the target speed
- Ki too low: persistent speed error under load
- Kd too high: noisy, erratic corrections from encoder noise

The simulation in `pid_sim.py` shows a first-order system (like a motor) being controlled to a step setpoint. You can tweak the gains and see the step response.

References:

- "PID controller explained" — https://www.youtube.com/watch?v=UR0hOmjaHp0
- Brian Douglas control systems series on YouTube
- https://docs.px4.io/main/en/config_mc/pid_tuning_guide_multicopter.html (good applied reference)
