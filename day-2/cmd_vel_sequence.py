"""
Author: Huseyn Kishiyev
Publishes a timed sequence of Twist commands to /cmd_vel.
Useful for testing odometry accuracy — run a known movement
and compare the resulting pose to ground truth.

Sequence:
  1. Drive forward 1 m at 0.2 m/s  (5 sec)
  2. Stop 1 sec
  3. Rotate 90 deg at 0.5 rad/s    (~3.14 sec)
  4. Drive forward 1 m at 0.2 m/s  (5 sec)
  5. Stop

Run:
  ros2 run -pkg- cmd_vel_sequence
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import time


class CmdVelSequence(Node):
    def __init__(self):
        super().__init__("cmd_vel_sequence")
        self.pub = self.create_publisher(Twist, "/cmd_vel", 10)

        # Give the system time to connect
        time.sleep(1.0)

        self.run_sequence()

    def send(self, linear_x: float, angular_z: float, duration: float):
        """Publish a constant Twist for the given duration."""
        msg = Twist()
        msg.linear.x  = linear_x
        msg.angular.z = angular_z

        rate_hz = 20.0
        dt = 1.0 / rate_hz
        steps = int(duration * rate_hz)

        for _ in range(steps):
            self.pub.publish(msg)
            time.sleep(dt)

        # Stop
        self.pub.publish(Twist())
        time.sleep(0.2)

    def run_sequence(self):
        self.get_logger().info("Starting sequence: forward 1m")
        self.send(linear_x=0.2, angular_z=0.0, duration=5.0)

        self.get_logger().info("Pause")
        time.sleep(1.0)

        # 90 degrees = pi/2 rad. At 0.5 rad/s => pi/2 / 0.5 = ~3.14 sec
        import math
        turn_duration = (math.pi / 2.0) / 0.5
        self.get_logger().info(f"Rotating 90 deg ({turn_duration:.2f}s)")
        self.send(linear_x=0.0, angular_z=0.5, duration=turn_duration)

        time.sleep(0.5)

        self.get_logger().info("Forward 1m again")
        self.send(linear_x=0.2, angular_z=0.0, duration=5.0)

        self.get_logger().info("Sequence done. Final pose should be at (1, 1, 90 deg).")
        self.get_logger().info("Check /odom for actual values.")


def main(args=None):
    rclpy.init(args=args)
    node = CmdVelSequence()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
