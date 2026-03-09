# February 28, 2026

## Triangulation and Recovering 3D Points from Two Views

### The triangulation problem

Given two camera projection matrices P0 and P1, and corresponding 2D points p0, p1, find the 3D point X such that:

$$\lambda_0 \mathbf{p}_0 = \mathbf{P}_0 \mathbf{X}, \quad \lambda_1 \mathbf{p}_1 = \mathbf{P}_1 \mathbf{X}$$

In the ideal (noise-free) case, the two rays from each camera center through p0 and p1 intersect exactly at X. In practice, due to pixel quantization and matching errors, the rays are skew lines and you find the midpoint of the shortest segment connecting them, or use a linear algebraic method.

### DLT triangulation

Expanding the cross product `p0 x (P0 X) = 0`:

For each point correspondence, you get 4 equations (2 per view, 2 per view), but only 3 are independent (homogeneous). Stack them as `A X = 0` and solve with SVD — the solution is the last right singular vector.

For the first camera (assume it is the world origin, P0 = K [I | 0]):

$$\mathbf{A} = \begin{bmatrix} u_0 \mathbf{P}_0^{3T} - \mathbf{P}_0^{1T} \\ v_0 \mathbf{P}_0^{3T} - \mathbf{P}_0^{2T} \\ u_1 \mathbf{P}_1^{3T} - \mathbf{P}_1^{1T} \\ v_1 \mathbf{P}_1^{3T} - \mathbf{P}_1^{2T} \end{bmatrix}$$

where P^{iT} is the i-th row of the projection matrix.

OpenCV provides `cv2.triangulatePoints(P0, P1, pts0, pts1)` which uses this approach.

### Scale ambiguity in monocular visual odometry

The recovered t from `cv2.recoverPose` is a unit vector, we know the direction of translation but not the magnitude. This is the fundamental scale ambiguity of monocular vision. Two scenes that differ only by a scale factor produce identical image measurements.

To recover scale we need one of:
- Known 3D reference (a measured object in the scene, like a calibration board on the ground)
- IMU integration — the accelerometer gives real-world acceleration in m/s², which, when double-integrated and compared to the visual trajectory, constrains scale
- GPS — absolute position from GPS combined with relative position from VO

In the PX4 visual-inertial odometry flow (using EKF2 with external vision input), the GPS provides scale, and the visual odometry provides high-rate relative motion between GPS updates.

### Scale from IMU and the visual-inertial alignment

The alignment between the visual trajectory (up to scale) and the IMU trajectory (true scale) solves for the scale factor s, the initial velocity, and the gravity direction simultaneously. This is done in a batch optimization over a window of frames. VINS-Mono and OpenVINS both implement this initialization step.

The optimization minimizes:

$$\sum_k \left\| \mathbf{p}_{k+1}^{imu} - \mathbf{p}_k^{imu} - \mathbf{v}_k \Delta t - \frac{1}{2} \mathbf{g} \Delta t^2 - s \left(\mathbf{p}_{k+1}^{vis} - \mathbf{p}_k^{vis}\right) \right\|^2$$

### Implementation note

The `triangulate_points.py` script demonstrates triangulation on a synthetic stereo pair, it creates two cameras with known relative pose, generates random 3D points, projects them into each camera, adds pixel noise, runs triangulation, and measures the 3D error. This is a clean way to verify the geometry pipeline before connecting to real image data.

References:

- https://vnav.mit.edu/lectures.html (Lecture 6 — Triangulation and Structure)
- Hartley & Zisserman "Multiple View Geometry" — Chapter 12
- https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#gad3fc9a0c82b08df034234979960b778c
- VINS-Mono paper: Qin et al. arXiv:1708.03852
