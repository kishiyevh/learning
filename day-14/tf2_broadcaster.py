"""
Author: Huseyn Kishiyev
------------------
Demonstrates a minimal TF2 dynamic transform broadcaster.
Publishes odom -> base_link transform for a robot moving in a circle.
Also includes a Transform listener that queries camera_link relative to odom.

Static transforms (camera extrinsics) are also published.

Run this node, then in another terminal:
  ros2 run tf2_tools view_frames
to generate a PDF of the transform tree.

Or check a specific transform:
  ros2 run tf2_ros tf2_echo odom camera_link

Requires: ROS2 Humble, tf2_ros, geometry_msgs
"""

import rclpy
from rclpy.node import Node
import tf2_ros
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped
import math
import numpy as np


def euler_to_quat(roll: float, pitch: float, yaw: float) -> tuple:
    """Convert Euler angles (rad) to quaternion (x, y, z, w)."""
    cy = math.cos(yaw   * 0.5)
    sy = math.sin(yaw   * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll  * 0.5)
    sr = math.sin(roll  * 0.5)

    return (
        sr * cp * cy - cr * sp * sy,  # x
        cr * sp * cy + sr * cp * sy,  # y
        cr * cp * sy - sr * sp * cy,  # z
        cr * cp * cy + sr * sp * sy,  # w
    )


def make_transform(parent: str, child: str, stamp,
                   tx=0.0, ty=0.0, tz=0.0,
                   roll=0.0, pitch=0.0, yaw=0.0) -> TransformStamped:
    t = TransformStamped()
    t.header.stamp    = stamp
    t.header.frame_id = parent
    t.child_frame_id  = child
    t.transform.translation.x = tx
    t.transform.translation.y = ty
    t.transform.translation.z = tz
    qx, qy, qz, qw = euler_to_quat(roll, pitch, yaw)
    t.transform.rotation.x = qx
    t.transform.rotation.y = qy
    t.transform.rotation.z = qz
    t.transform.rotation.w = qw
    return t


class TF2Demo(Node):
    def __init__(self):
        super().__init__("tf2_demo")

        # Dynamic broadcaster (robot pose, changes every step)
        self.dyn_broadcaster = TransformBroadcaster(self)

        # Static broadcaster (sensor extrinsics, only published once)
        self.static_broadcaster = StaticTransformBroadcaster(self)
        self._publish_static_transforms()

        # TF listener to query transforms
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.t = 0.0
        self.dt = 0.05
        self.radius = 2.0  # circular trajectory radius

        self.timer = self.create_timer(self.dt, self.step)
        self.get_logger().info("TF2 demo started — robot moving in circle (r=2m)")

    def _publish_static_transforms(self):
        """
        Publish sensor placements once.
        - Camera is 10 cm forward, 5 cm up from base_link, no rotation
        - IMU is at origin of base_link (identity)
        """
        stamp = self.get_clock().now().to_msg()

        cam_tf = make_transform(
            "base_link", "camera_link", stamp,
            tx=0.10, ty=0.0, tz=0.05,
            roll=0.0, pitch=0.0, yaw=0.0,
        )
        imu_tf = make_transform(
            "base_link", "imu_link", stamp,
            tx=0.0, ty=0.0, tz=0.0,
        )
        self.static_broadcaster.sendTransform([cam_tf, imu_tf])
        self.get_logger().info("Static transforms published (camera_link, imu_link)")

    def step(self):
        self.t += self.dt

        # Robot follows a circle
        x   = self.radius * math.cos(self.t * 0.3)
        y   = self.radius * math.sin(self.t * 0.3)
        yaw = self.t * 0.3 + math.pi / 2  # always tangent to circle

        stamp = self.get_clock().now().to_msg()

        # Broadcast odom -> base_link
        robot_tf = make_transform(
            "odom", "base_link", stamp,
            tx=x, ty=y, tz=0.0,
            yaw=yaw,
        )
        self.dyn_broadcaster.sendTransform(robot_tf)

        # Every 2 seconds, look up camera_link in odom frame
        if abs(self.t % 2.0) < self.dt:
            self._query_camera_in_odom(stamp)

    def _query_camera_in_odom(self, stamp):
        try:
            tf_stamped = self.tf_buffer.lookup_transform(
                "odom",
                "camera_link",
                rclpy.time.Time(),   # latest available
            )
            tr = tf_stamped.transform.translation
            ro = tf_stamped.transform.rotation
            self.get_logger().info(
                f"camera_link in odom: "
                f"pos=({tr.x:.3f}, {tr.y:.3f}, {tr.z:.3f})  "
                f"quat=({ro.x:.3f}, {ro.y:.3f}, {ro.z:.3f}, {ro.w:.3f})"
            )
        except tf2_ros.TransformException as e:
            self.get_logger().warn(f"TF lookup failed: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = TF2Demo()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
