"""
Author: Huseyn Kishiyev
Minimal PX4 offboard control example using ROS2 + uXRCE-DDS.
Sends trajectory setpoints to make the simulated X500 hover at 5m.

Prerequisites:
  - PX4 SITL running (make px4_sitl gz_x500)
  - MicroXRCEAgent running (MicroXRCEAgent udp4 -p 8888)
  - uxrce_dds_client started inside PX4 SITL console

Install px4_msgs:
  git clone https://github.com/PX4/px4_msgs.git ~/ros2_ws/src/px4_msgs
  cd ~/ros2_ws && colcon build --packages-select px4_msgs

Run:
  ros2 run <your_pkg> px4_offboard_takeoff
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleStatus,
)


class OffboardTakeoff(Node):
    def __init__(self):
        super().__init__("offboard_takeoff")

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # Publishers
        self.offboard_mode_pub = self.create_publisher(
            OffboardControlMode, "/fmu/in/offboard_control_mode", qos)
        self.trajectory_pub = self.create_publisher(
            TrajectorySetpoint, "/fmu/in/trajectory_setpoint", qos)
        self.vehicle_cmd_pub = self.create_publisher(
            VehicleCommand, "/fmu/in/vehicle_command", qos)

        # Subscriber
        self.status_sub = self.create_subscription(
            VehicleStatus, "/fmu/out/vehicle_status", self.status_cb, qos)

        self.offboard_setpoint_counter = 0
        self.vehicle_status = VehicleStatus()
        self.armed = False

        self.timer = self.create_timer(0.1, self.timer_cb)  # 10 Hz
        self.get_logger().info("Offboard takeoff node started")

    def status_cb(self, msg: VehicleStatus):
        self.vehicle_status = msg

    def arm(self):
        msg = VehicleCommand()
        msg.command = VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM
        msg.param1 = 1.0  # 1 = arm
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.vehicle_cmd_pub.publish(msg)
        self.get_logger().info("Arm command sent")

    def engage_offboard_mode(self):
        msg = VehicleCommand()
        msg.command = VehicleCommand.VEHICLE_CMD_DO_SET_MODE
        msg.param1 = 1.0   # custom mode
        msg.param2 = 6.0   # PX4 offboard mode number
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.vehicle_cmd_pub.publish(msg)
        self.get_logger().info("Offboard mode command sent")

    def publish_offboard_control_mode(self):
        msg = OffboardControlMode()
        msg.position     = True
        msg.velocity     = False
        msg.acceleration = False
        msg.attitude     = False
        msg.body_rate    = False
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_mode_pub.publish(msg)

    def publish_trajectory_setpoint(self, x=0.0, y=0.0, z=-5.0, yaw=0.0):
        # PX4 NED frame: z is negative for altitude above ground
        msg = TrajectorySetpoint()
        msg.position = [x, y, z]
        msg.yaw = yaw
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.trajectory_pub.publish(msg)

    def timer_cb(self):
        self.publish_offboard_control_mode()
        self.publish_trajectory_setpoint(x=0.0, y=0.0, z=-5.0)

        # After 10 setpoints, arm and switch to offboard
        if self.offboard_setpoint_counter == 10:
            self.engage_offboard_mode()
            self.arm()

        if self.offboard_setpoint_counter < 11:
            self.offboard_setpoint_counter += 1


def main(args=None):
    rclpy.init(args=args)
    node = OffboardTakeoff()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
