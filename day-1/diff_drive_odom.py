"""
Author: Huseyn Kishiyev
Manual differential drive odometry node for ROS2.
Subscribes to /joint_states, computes robot pose using
the two-virtual-wheel kinematic model, and publishes to /odom.

Tested with ROSbot XL parameters (approximate values):
  wheel_radius = 0.05 m
  track_width  = 0.26 m (l2)

Run:
  ros2 run <your_pkg> diff_drive_odom
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion
import math
import time


def euler_to_quaternion(yaw: float) -> Quaternion:
    q = Quaternion()
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw / 2.0)
    q.w = math.cos(yaw / 2.0)
    return q


class DiffDriveOdom(Node):
    def __init__(self):
        super().__init__("diff_drive_odom")

        # Robot parameters — adjust for your hardware
        self.wheel_radius = 0.05       # meters
        self.track_width  = 0.26       # l2 in meters (left-right wheel distance)

        # Wheel name mapping — match what your robot publishes
        self.wheel_names = {
            "fl": "front_left_wheel_joint",
            "fr": "front_right_wheel_joint",
            "rl": "rear_left_wheel_joint",
            "rr": "rear_right_wheel_joint",
        }

        # Robot state
        self.x     = 0.0
        self.y     = 0.0
        self.alpha = 0.0   # heading in radians

        # Previous wheel positions (radians)
        self.prev_phi = {"fl": None, "fr": None, "rl": None, "rr": None}
        self.prev_time = None

        self.sub = self.create_subscription(
            JointState,
            "/joint_states",
            self.joint_state_cb,
            10,
        )

        self.pub = self.create_publisher(Odometry, "/odom", 10)
        self.get_logger().info("diff_drive_odom started")

    def joint_state_cb(self, msg: JointState):
        now = self.get_clock().now().nanoseconds * 1e-9

        # Extract wheel positions from joint state message
        phi = {}
        name_to_key = {v: k for k, v in self.wheel_names.items()}
        for i, name in enumerate(msg.name):
            if name in name_to_key:
                phi[name_to_key[name]] = msg.position[i]

        if len(phi) < 4:
            return  # not all wheels found yet

        if any(self.prev_phi[k] is None for k in phi):
            self.prev_phi = dict(phi)
            self.prev_time = now
            return

        dt = now - self.prev_time
        if dt <= 0.0:
            return

        # Virtual wheel angular displacements (average of front/rear)
        d_phi_L = ((phi["fl"] - self.prev_phi["fl"]) +
                   (phi["rl"] - self.prev_phi["rl"])) / 2.0

        d_phi_R = ((phi["fr"] - self.prev_phi["fr"]) +
                   (phi["rr"] - self.prev_phi["rr"])) / 2.0

        # Linear distance traveled by each virtual wheel
        d_L = d_phi_L * self.wheel_radius
        d_R = d_phi_R * self.wheel_radius

        # Change in heading
        d_alpha = (d_R - d_L) / self.track_width

        # Forward distance (center of robot)
        d_center = (d_L + d_R) / 2.0

        # Update pose
        self.alpha += d_alpha
        self.x += d_center * math.cos(self.alpha)
        self.y += d_center * math.sin(self.alpha)

        # Publish odometry
        odom_msg = Odometry()
        odom_msg.header.stamp = self.get_clock().now().to_msg()
        odom_msg.header.frame_id = "odom"
        odom_msg.child_frame_id = "base_link"

        odom_msg.pose.pose.position.x = self.x
        odom_msg.pose.pose.position.y = self.y
        odom_msg.pose.pose.position.z = 0.0
        odom_msg.pose.pose.orientation = euler_to_quaternion(self.alpha)

        # Velocities
        odom_msg.twist.twist.linear.x  = d_center / dt
        odom_msg.twist.twist.angular.z = d_alpha  / dt

        self.pub.publish(odom_msg)

        self.prev_phi = dict(phi)
        self.prev_time = now


def main(args=None):
    rclpy.init(args=args)
    node = DiffDriveOdom()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
