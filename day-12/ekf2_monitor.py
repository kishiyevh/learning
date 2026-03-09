"""
Author: Huseyn Kishiyev
Sources: PX4 docs, PX4 repo, Claude for rclpy package.
---------------
Subscribes to PX4 EKF2 output topics (via uXRCE-DDS) and prints
innovation test ratios, sensor health flags, and state estimates
in a terminal dashboard.

Useful for diagnosing EKF2 issues during SITL or real flight tests.

Prerequisites:
  - PX4 SITL running
  - MicroXRCEAgent running
  - px4_msgs built in your ROS2 workspace

Topics monitored:
  /fmu/out/estimator_status_flags   -- sensor health / innovation failures
  /fmu/out/estimator_innovations    -- innovation values per sensor
  /fmu/out/vehicle_odometry         -- EKF2 state output

Run:
  ros2 run <your_pkg> ekf2_monitor
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

try:
    from px4_msgs.msg import (
        EstimatorStatusFlags,
        EstimatorInnovations,
        VehicleOdometry,
    )
    HAS_PX4_MSGS = True
except ImportError:
    HAS_PX4_MSGS = False
    print("WARNING: px4_msgs not found. Running in mock mode.")

import time
import os


def clear():
    os.system("clear")


class EKF2Monitor(Node):
    def __init__(self):
        super().__init__("ekf2_monitor")

        self.latest = {
            "status": None,
            "innovations": None,
            "odometry": None,
        }

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        if HAS_PX4_MSGS:
            self.create_subscription(EstimatorStatusFlags,
                "/fmu/out/estimator_status_flags", self._status_cb, qos)
            self.create_subscription(EstimatorInnovations,
                "/fmu/out/estimator_innovations", self._innov_cb, qos)
            self.create_subscription(VehicleOdometry,
                "/fmu/out/vehicle_odometry", self._odom_cb, qos)
        else:
            self.get_logger().warning("px4_msgs not available — no subscriptions created")

        self.create_timer(0.5, self._print_dashboard)

    def _status_cb(self, msg):
        self.latest["status"] = msg

    def _innov_cb(self, msg):
        self.latest["innovations"] = msg

    def _odom_cb(self, msg):
        self.latest["odometry"] = msg

    def _print_dashboard(self):
        clear()
        print("=" * 60)
        print("  PX4 EKF2 Monitor  (0.5 Hz refresh)")
        print("=" * 60)

        odom = self.latest["odometry"]
        if odom is not None:
            pos = odom.position
            vel = odom.velocity
            print(f"\n  POSITION (m, local NED):")
            print(f"    N={pos[0]:+8.3f}  E={pos[1]:+8.3f}  D={pos[2]:+8.3f}")
            print(f"\n  VELOCITY (m/s):")
            print(f"    N={vel[0]:+8.3f}  E={vel[1]:+8.3f}  D={vel[2]:+8.3f}")
        else:
            print("\n  [Odometry] no data yet")

        innov = self.latest["innovations"]
        if innov is not None:
            print(f"\n  INNOVATIONS:")
            # gps_hvel: GPS horizontal velocity innovation
            try:
                gps_hvel = innov.gps_hvel
                print(f"    GPS hvel innov:  [{gps_hvel[0]:+7.4f}, {gps_hvel[1]:+7.4f}] m/s")
            except AttributeError:
                pass
            try:
                baro_vpos = innov.baro_vpos
                print(f"    Baro vpos innov: {baro_vpos:+7.4f} m")
            except AttributeError:
                pass
            try:
                mag_field = innov.mag_field
                print(f"    Mag field innov: [{mag_field[0]:+7.4f}, {mag_field[1]:+7.4f}, {mag_field[2]:+7.4f}]")
            except AttributeError:
                pass
        else:
            print("\n  [Innovations] no data yet")

        status = self.latest["status"]
        if status is not None:
            print(f"\n  SENSOR STATUS FLAGS:")
            flags = {
                "GPS vel OK":       getattr(status, "gps_hvel_test_ratio_failed", None),
                "Baro OK":          getattr(status, "baro_vpos_test_ratio_failed", None),
                "Mag OK":           getattr(status, "mag_field_test_ratio_failed", None),
                "EKF OK":           getattr(status, "solution_is_high_freq_tilt_corr_type", None),
            }
            for name, failed in flags.items():
                if failed is not None:
                    status_str = "FAIL" if failed else " OK "
                    print(f"    {name:<20} [{status_str}]")
        else:
            print("\n  [Status flags] no data yet")

        print("\n" + "=" * 60)
        print("  Ctrl+C to exit")


def main(args=None):
    rclpy.init(args=args)
    node = EKF2Monitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
