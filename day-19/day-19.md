# March 6, 2026

## Bayesian Filters, Particle Filter and the Nav2 AMCL Localizer

Labbe's Kalman book chapters 11-12 cover Bayesian filtering beyond the KF. The particle filter is the general case and it makes no Gaussian or linearity assumptions, at the cost of higher computational load.

### Why the Kalman filter is a special case of Bayes

The Kalman filter is optimal for linear systems with Gaussian noise. The belief (posterior distribution over state) is a Gaussian, which can be represented exactly by just a mean and covariance matrix. This is what makes it so computationally efficient.

When either the system model or the measurement model is nonlinear, or when the noise is non-Gaussian, the posterior is no longer Gaussian. The EKF approximates it as Gaussian anyway (linearization). This is usually fine, but it breaks down for highly nonlinear systems or multimodal distributions.

The particle filter represents the posterior as a set of weighted samples ("particles"). No parametric assumption is required.

### Particle filter algorithm

State the belief as N weighted particles: `{x_i, w_i}` where `sum(w_i) = 1`.

**Predict step** — propagate each particle through the motion model with process noise:

```python
for i in range(N):
    x_particles[i] = motion_model(x_particles[i], u) + sample_noise(Q)
```

**Update step** — weight each particle by how likely the measurement is given that state:

```python
for i in range(N):
    w[i] *= measurement_likelihood(z, x_particles[i])
w /= w.sum()  # normalize
```

**Resample step** — sample N particles from the current weighted set. Particles with low weight die out; high-weight particles are duplicated. This prevents weight degeneracy (all weight collapsing onto one particle).

The estimate is the weighted mean: `x_est = sum(w_i * x_i)`.

### AMCL (Adaptive Monte Carlo Localization)

AMCL is the particle filter localizer in Nav2. It takes:
- Lidar scan on `/scan`
- Odometry on `/odom` (motion model)
- A known map (occupancy grid)

And outputs the robot's pose within the map as `map -> odom` TF transform.

The motion model for AMCL is the differential drive odometry model with configurable noise parameters. The measurement model scores each particle by comparing the lidar scan to the occupancy grid — for each particle pose, ray-cast the lidar and compare expected vs actual scan endpoints.

AMCL parameters that matter:

```yaml
amcl:
  ros__parameters:
    min_particles: 500
    max_particles: 2000
    update_min_d: 0.2     # motion needed before filter update (m)
    update_min_a: 0.5     # angle needed before filter update (rad)
    resample_interval: 1
    alpha1: 0.2    # rotation noise from rotation
    alpha2: 0.2    # rotation noise from translation
    alpha3: 0.2    # translation noise from translation
    alpha4: 0.2    # translation noise from rotation
    laser_max_range: 8.0
    laser_model_type: likelihood_field
```

The `min_particles` / `max_particles` range enables the "adaptive" part and when the filter is confident (particles are clustered), it shrinks the particle count to save computation. When uncertain (particles spread), it increases.

### Particle filter for global localization (kidnapped robot problem)

Unlike the EKF (which needs a good initial estimate), a particle filter can handle global localization. You initialize particles uniformly across the entire free space of the map. The filter converges to the true location as measurements accumulate.

In AMCL this is called "global localization" and is triggered via:

```bash
ros2 service call /reinitialize_global_localization std_srvs/srv/Empty {}
```

After that, the robot drives around until particles converge. The convergence speed depends on the map's distinctiveness, a map with many unique features (varied geometry) converges faster than a symmetric or repetitive environment.

### PF and EKF (or UKF) usage for localization

The EKF (or UKF) is better when:
- You have a good initial estimate
- The state space is continuous and unimodal
- Computational budget is tight

The particle filter is better when:
- Global localization needed
- Non-Gaussian noise (e.g., lidar multipath)
- Multimodal posterior (robot could be in one of several similar-looking locations)

For a drone in GPS-denied environments using only visual localization, a particle filter over 6DOF pose is feasible with a fast measurement model (descriptor matching score as likelihood). This is roughly how some visual localization systems work — a top-down image database with particle filter over 2D position.

The script `particle_filter_1d.py` runs a 1D particle filter tracking a robot moving along a corridor with discrete location markers.

References:

- https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python (Chapters 11-12)
- https://docs.nav2.org/configuration/packages/configuring-amcl.html
- Thrun, Burgard, Fox "Probabilistic Robotics" — Chapter 4 (particle filters)
