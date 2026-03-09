"""
Author: Huseyn Kishiyev
-------------------
Sends a sequence of navigation goals to Nav2's bt_navigator
using the NavigateToPose action interface.

This is the programmatic equivalent of clicking "Nav2 Goal" in RViz2.
Useful for running automated navigation tests.

Usage:
  1. Start simulation + SLAM + Nav2
  2. ros2 run -your_pkg- nav2_goal_sender

Goals are defined in the WAYPOINTS list below.
Coordinates are in the map frame (meters from map origin).

Requires: ROS2 Humble, nav2_msgs
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped, Quaternion
import math
import time


def yaw_to_quat(yaw_deg: float) -> Quaternion:
    yaw = math.radians(yaw_deg)
    q = Quaternion()
    q.z = math.sin(yaw / 2)
    q.w = math.cos(yaw / 2)
    q.x = 0.0
    q.y = 0.0
    return q


def make_goal(x: float, y: float, yaw_deg: float = 0.0) -> NavigateToPose.Goal:
    goal = NavigateToPose.Goal()
    goal.pose = PoseStamped()
    goal.pose.header.frame_id = "map"
    goal.pose.pose.position.x = x
    goal.pose.pose.position.y = y
    goal.pose.pose.position.z = 0.0
    goal.pose.pose.orientation = yaw_to_quat(yaw_deg)
    return goal


# Waypoints: (x, y, heading_degrees)
WAYPOINTS = [
    ( 1.0,  0.0,   0.0),
    ( 1.0,  1.0,  90.0),
    ( 0.0,  1.0, 180.0),
    ( 0.0,  0.0, -90.0),
]


class Nav2GoalSender(Node):
    def __init__(self):
        super().__init__("nav2_goal_sender")
        self._client = ActionClient(self, NavigateToPose, "navigate_to_pose")

    def wait_for_server(self, timeout_sec=10.0):
        self.get_logger().info("Waiting for NavigateToPose action server...")
        if not self._client.wait_for_server(timeout_sec=timeout_sec):
            self.get_logger().error("Action server not available after timeout")
            return False
        self.get_logger().info("Action server connected")
        return True

    def send_goal_and_wait(self, x: float, y: float, yaw_deg: float = 0.0,
                           timeout_sec: float = 60.0) -> bool:
        goal = make_goal(x, y, yaw_deg)
        goal.pose.header.stamp = self.get_clock().now().to_msg()

        self.get_logger().info(f"Sending goal: x={x:.2f} y={y:.2f} yaw={yaw_deg:.1f}°")

        future = self._client.send_goal_async(goal, feedback_callback=self._feedback_cb)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)

        goal_handle = future.result()
        if not goal_handle or not goal_handle.accepted:
            self.get_logger().error("Goal rejected by Nav2")
            return False

        # Wait for result
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=timeout_sec)

        status = result_future.result().status
        if status == 4:  # SUCCEEDED
            self.get_logger().info("Goal reached successfully")
            return True
        else:
            self.get_logger().warn(f"Goal ended with status: {status}")
            return False

    def _feedback_cb(self, feedback_msg):
        fb = feedback_msg.feedback
        dist = fb.distance_remaining
        # Only log occasionally to avoid spam
        if hasattr(self, "_last_fb_log"):
            if time.time() - self._last_fb_log < 2.0:
                return
        self._last_fb_log = time.time()
        self.get_logger().info(f"  Distance remaining: {dist:.2f} m")

    def run_waypoint_mission(self):
        if not self.wait_for_server():
            return

        for i, (x, y, yaw) in enumerate(WAYPOINTS):
            self.get_logger().info(f"\n--- Waypoint {i+1}/{len(WAYPOINTS)} ---")
            success = self.send_goal_and_wait(x, y, yaw)
            if not success:
                self.get_logger().error(f"Failed at waypoint {i+1}, aborting mission")
                return
            time.sleep(0.5)

        self.get_logger().info("\nMission complete! All waypoints reached.")


def main(args=None):
    rclpy.init(args=args)
    node = Nav2GoalSender()
    node.run_waypoint_mission()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
