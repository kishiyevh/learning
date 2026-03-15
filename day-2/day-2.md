# Visualization with RViz2 and PlotJuggler, cmd_vel Teleop

After working through the kinematics math yesterday, today focused on actually seeing it work. The Husarion tutorial covers PlotJuggler and RViz2 as the two main visualization tools, and both ended up being more useful than I expected.

RViz2 is the standard choice for spatial data — you can visualize the robot model, TF tree, odometry arrows, laser scan points, and so on. PlotJuggler is better for time-series data: wheel velocities, IMU values, PID outputs, any scalar or vector data you want to plot against time. They serve different purposes and are best used together.

### Setting up RViz2 for odometry display

The most common issue is the fixed frame mismatch. By default RViz2 uses `map` as the fixed frame, but odometry data lives in the `odom` frame. Changing this in the `Global Options` panel fixes the blank display. Once set, you add an `Odometry` display, point it at `/odom`, and you see the pose arrow update in real time.

To also see the robot model, you need the URDF loaded and the `RobotModel` display type added. The `robot_description` parameter needs to be on the parameter server, which the launch file usually handles.

### PlotJuggler

PlotJuggler streams ROS2 topics as time-series data. The install on Ubuntu 22.04 with ROS2 Humble:

```bash
sudo apt install ros-humble-plotjuggler-ros
```

Run it:

```bash
ros2 run plotjuggler plotjuggler
```

Then in the UI, select the ROS2 topic streaming plugin, choose your topics (`/odom`, `/cmd_vel`, `/joint_states`), and drag fields into the plot area. I plotted `odom.twist.twist.linear.x` against time while driving the robot with the keyboard — you can see the velocity ramp up and settle clearly.

### Publishing cmd_vel manually

The teleop keyboard node is fine, but writing your own velocity publisher teaches you more. The `geometry_msgs/Twist` message has `linear.x` for forward speed and `angular.z` for turning rate. The motor driver converts these into individual wheel speed commands.

A minimal publisher that drives the robot in a circle:

```python
twist.linear.x  = 0.2   # m/s forward
twist.angular.z = 0.5   # rad/s counterclockwise
```

For a pure rotation in place:

```python
twist.linear.x  = 0.0
twist.angular.z = 1.0
```

The code for a timed sequence publisher is in `cmd_vel_sequence.py`.

### TF tree check

A quick sanity check that is easy to forget:

```bash
ros2 run tf2_tools view_frames
```

This generates a PDF of the current TF tree. If `odom -> base_link` is missing or broken, your RViz2 visualization will be wrong even if `/odom` data looks correct. The TF broadcaster is usually part of the odometry node — in yesterday's node I did not include it, which is something to fix next.

### What I noticed about the ROSbot model

The ROSbot XL uses a `l2` value of roughly 0.26 m and wheel radius of about 0.05 m. These numbers go directly into the kinematic equations. If you get them wrong, the odometry drifts visibly — the robot thinks it's traveling in an arc when it's going straight, for example. Real robots require calibration to get these numbers accurate.

References:

- https://husarion.com/tutorials/ros2-tutorials/4-kinematics-and-visualization/
- https://github.com/PlotJuggler/plotjuggler-ros-plugins
- https://docs.ros.org/en/humble/Tutorials/Intermediate/Tf2/Introduction-To-Tf2.html
