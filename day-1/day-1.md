# February 16, 2026

## ROS2 Differential Drive Kinematics and the ROSbot Model

The main idea is straightforward. The ROSbot has four wheels but is modeled as two virtual wheels, WL and WR, whose axes pass through the robot's geometric center. This lets you use the simpler differential drive kinematic equations without worrying about the geometry of each physical wheel separately. The virtual wheel velocities are just averages of their physical counterparts.

$$\phi_{W_L} = \frac{\phi_{W_{FL}} + \phi_{W_{RL}}}{2}, \quad \phi_{W_R} = \frac{\phi_{W_{FR}} + \phi_{W_{RR}}}{2}$$

$$\omega_{W_L} = \frac{\omega_{W_{FL}} + \omega_{W_{RL}}}{2}, \quad \omega_{W_R} = \frac{\omega_{W_{FR}} + \omega_{W_{RR}}}{2}$$

From those angular velocities and the wheel radius `r`, you get linear velocities:

$$v_L = \omega_{W_L} \cdot r, \quad v_R = \omega_{W_R} \cdot r$$

The robot's angular position and rate:

$$\alpha = (\phi_{W_R} - \phi_{W_L}) \cdot \frac{r}{l_2}$$

$$\dot{\alpha} = \frac{d\alpha}{dt}$$

And then the full pose update:

$$\dot{x}_c = \left(v_L + \dot{\alpha} \cdot \frac{r}{l_2}\right) \cos \alpha$$

$$\dot{y}_c = \left(v_L + \dot{\alpha} \cdot \frac{r}{l_2}\right) \sin \alpha$$

$$x_c = \int_0^t \dot{x}_c \, dt, \quad y_c = \int_0^t \dot{y}_c \, dt$$

The term `l2` is the distance between the left and right wheels (the track width). Starting from (0, 0) the robot integrates its pose over time from wheel encoder data.

One thing worth noting is the non-holonomic constraint. The ROSbot can only move forward/backward and rotate — it cannot slide sideways. This means not all 3 DOF are independently controllable. Mecanum or omni-wheel robots are holonomic and can do lateral translation, but they trade off simplicity for that capability.

I wrote a ROS2 node that subscribes to `/joint_states` and computes odometry by hand using these equations, then publishes to `/odom`. This is basically what the motor driver node does internally, but doing it yourself makes the math concrete. The code is in `diff_drive_odom.py`.

Visualization was the second part. I launched RViz2 and added the `Odometry` display type, then pointed it at `/odom`. Running `teleop_twist_keyboard` and watching the odometry arrow move in RViz2 while the math updates in real time is a good sanity check.


References used today:

- https://husarion.com/tutorials/ros2-tutorials/4-kinematics-and-visualization/
- http://docs.ros.org/en/humble/p/geometry_msgs/
- https://wiki.ros.org/teleop_twist_keyboard
