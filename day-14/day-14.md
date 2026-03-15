# ROS2 TF2 Transformations and the Robot State Estimation Chain

Today tied together the ROS2 transform system with the state estimation work. TF2 is the coordinate frame management system in ROS2. It handles the tree of rigid body transforms that connects every sensor frame, body frame, and world frame in a robot system.

### Why TF2 matters

When we have an IMU on a drone, its data is expressed in the IMU body frame. The camera is mounted at a different position and orientation. The GPS antenna is somewhere else. To combine all of this in an EKF, every sensor reading needs to be transformed into a common frame (typically the robot body frame or the world frame).

TF2 provides a time-indexed, interpolated lookup of any transform in the tree. You can ask "what was the transform from `camera_frame` to `base_link` at time T?" and get the answer by composing the chain of static and dynamic transforms.

### TF2 tree structure for a drone

A typical PX4/ROS2 drone TF tree:

```
map
 └── odom
      └── base_link
           ├── imu_link
           ├── gps_link
           └── camera_link
                └── camera_optical_frame
```

Static transforms (sensor placement, does not change) are broadcast once,

```bash
ros2 run tf2_ros static_transform_publisher \
    0.05 0.0 0.02  0.0 0.0 0.0  base_link camera_link
```

This places the camera 5 cm forward, 2 cm up from the base_link origin, no rotation.

Dynamic transforms (robot moving) are broadcast continuously by the odometry/EKF node.

### Listening to transforms in Python

```python
import rclpy
from rclpy.node import Node
import tf2_ros
from tf2_ros import TransformException
from geometry_msgs.msg import PointStamped
import tf2_geometry_msgs  # needed for do_transform_point

class TFListener(Node):
    def __init__(self):
        super().__init__("tf_listener")
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

    def transform_point(self, point_stamped, target_frame):
        try:
            return self.tf_buffer.transform(point_stamped, target_frame)
        except TransformException as e:
            self.get_logger().warn(f"TF error: {e}")
            return None
```

### The odom -> map transform

The `odom` frame is the starting position of the robot — the EKF outputs pose relative to where the robot was when it initialized. The `map` frame is a globally consistent frame that may shift when a new GPS fix comes in or a loop closure is detected.

For a UAV in outdoor flight with GPS, the `map` frame is the GPS origin (first fix). The `odom -> map` transform is initially identity and stays identity if there are no loop closures or GPS re-localizations.

For indoor operation without GPS, the `map` frame is provided by a SLAM system (like RTAB-Map or ORB-SLAM3), and the `odom -> map` transform is updated every time the SLAM system detects a loop closure and corrects accumulated drift.

### Quaternion composition

A common source of bugs is quaternion composition order. In ROS2, quaternions follow the right-hand convention. Composing rotation A then rotation B:

$$\mathbf{q}_{AB} = \mathbf{q}_B \otimes \mathbf{q}_A$$

The order matters — it is not commutative. The TF2 `geometry_msgs.Transform` uses `tf2_ros.transformations` for this.

References:

- https://docs.ros.org/en/humble/Tutorials/Intermediate/Tf2/Tf2-Main.html
- https://vnav.mit.edu/lectures.html (Lecture 7)
- https://docs.ros.org/en/humble/p/tf2_ros/
