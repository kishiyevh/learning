# March 7, 2026

## Visual Odometry Front-End and Integrating the Full Pipeline

### The VO pipeline as a state machine

A monocular visual odometry front-end has these stages per frame pair,

```
Frame k-1 (reference)          Frame k (current)
     |                               |
SuperPoint                      SuperPoint
kpts_{k-1}, desc_{k-1}         kpts_k, desc_k
          \                        /
           LightGlue matcher
                |
         matched pairs
                |
         Essential matrix (RANSAC)
                |
         recoverPose -> (R, t)
                |
         Triangulate inlier matches -> 3D points
                |
         Scale from known baseline / IMU / GPS
                |
         Accumulate pose: T_world = T_world * T_{k-1,k}
```

### Keyframe selection

We don't want to run this for every single frame. At high frame rates (60 fps), consecutive frames have almost no baseline and the geometry is ill-conditioned (large rotation uncertainty). Instead, you select keyframes:

A new keyframe is triggered when;
1. The median optical flow magnitude exceeds a threshold (robot has moved enough)
2. The number of tracked features in the current frame drops below a threshold (old features lost, need new detection)
3. A fixed time interval has elapsed

Between keyframes, simpler tracking (KLT optical flow via `cv2.calcOpticalFlowPyrLK`) is used to maintain feature tracks without running the full SuperPoint + LightGlue inference.

### Dealing with pure rotation and degenerate configurations

If the drone hovers and only rotates, `t` from `recoverPose` is unreliable, there is no parallax to triangulate from. The ratio of inliers for a homography (planar motion) vs essential matrix (general motion) tells you which model is more appropriate:

```cpp
// If homography_inliers / essential_inliers > 0.8, likely planar/pure rotation
double ratio = H_inliers / E_inliers;
bool use_homography = (ratio > 0.8);
```

When pure rotation is detected, you update attitude from the rotation but do not update position. This is what the ORBSLAM3 initializer does.

### Scale recovery from known camera height

For a drone flying at known height above a flat ground, you can recover scale by triangulating ground plane points and enforcing the ground plane equation `z = -h`:

```
scale = h / z_triangulated_mean
t_metric = scale * t_unit
```

This only works when the camera has a sufficient ground view (downward-facing or pitched camera).

### Pose accumulation and drift

The accumulated pose after N frames is the composition of relative transforms;

$$T_{0,N} = T_{0,1} \cdot T_{1,2} \cdot \ldots \cdot T_{N-1,N}$$

Each relative transform has estimation error. These errors compound, this is why visual odometry drifts over long trajectories. The drift rate is roughly proportional to the number of frames and the per-frame rotation/translation error.

Loop closure detection (recognizing a previously visited place) and global bundle adjustment are what keep long-term VO systems bounded. Without it, even a well-tuned VO system drifts at roughly 0.5-2% of path length.

### ROS2 integration

The VO node subscribes to `/camera/image_raw` and publishes `geometry_msgs/PoseStamped` on `/vo/pose` and broadcasts the `odom -> camera_link` TF transform. The PX4 EKF2 consumes this as an external vision input via the `/fmu/in/vehicle_visual_odometry` topic.

The C++ implementation `vo_frontend.cpp` runs the ONNX inference via onnxruntime C++ API, does the OpenCV geometry step, and accumulates the trajectory. It reads images from a directory as a stand-in for a live camera topic, connecting to a live ROS2 image topic requires adding a `sensor_msgs::msg::Image` subscriber with `cv_bridge`.

Build with the provided `CMakeLists.txt`.

References:

- Forster et al. "SVO: Fast Semi-Direct Monocular Visual Odometry" (ICRA 2014)
- Engel et al. "DSO: Direct Sparse Odometry" (TPAMI 2018)
- https://github.com/UZ-SLAMLab/ORB_SLAM3
- https://vnav.mit.edu/lectures.html (Lecture 9-10)
- https://onnxruntime.ai/docs/api/c/
