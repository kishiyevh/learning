# ROS2 SLAM and Nav2 Navigation Stack Architecture

### SLAM

SLAM (Simultaneous Localization and Mapping) solves the chicken-and-egg problem (as per se). We need a map to localize, but we need localization to build a map. The output is a 2D occupancy grid (for ground robots) or a 3D point cloud map (for lidar-based systems), along with a pose estimate within that map.

What SLAM does not give us is a planned path or obstacle avoidance. That is Nav2's job.

### SLAM Toolbox

`slam_toolbox` is the standard SLAM package for ROS2 Humble. It works with a 2D lidar scan on the `/scan` topic and publishes the `/map` occupancy grid and the `map -> odom` transform.

Install:

```bash
sudo apt install ros-humble-slam-toolbox
```

Run with the async mapper (good for real-time on a normal PC):

```bash
ros2 launch slam_toolbox online_async_launch.py \
    slam_params_file:=./my_slam_params.yaml \
    use_sim_time:=false
```

The crucial parameter in the YAML is `resolution` (default 0.05 m/cell) and `max_laser_range`. Too high a resolution with long lidar range means huge maps that slow down loop closure.

Loop closure is what separates SLAM from dead reckoning. When the robot returns to a previously visited location, it matches the current lidar scan against the existing map and corrects accumulated drift. The `map -> odom` transform shifts to reflect this correction.

### Nav2 stack overview

Nav2 is a collection of nodes:

```
map_server (publishes /map from a saved map file)
    |
amcl (localizes robot within map using particle filter)
    |
bt_navigator (behavior tree high-level coordinator)
    |
planner_server (global path planning — NavFn or Smac)
    |
controller_server (local trajectory following — DWB or MPPI)
    |
/cmd_vel -> robot
```

The behavior tree (BT) is what makes Nav2 flexible. Each navigation action (go to pose, spin, wait) is a BT leaf node. The tree handles recovery behaviors: if the robot fails to make progress (stuck), the BT triggers recovery actions (back up, rotate, try again).

### Running Nav2 in simulation

With PX4 and the ROSbot Gazebo simulation, you can test the full stack without hardware. For the ROSbot XL in Gazebo:

```bash
# Terminal 1: Launch Gazebo + ROSbot
ros2 launch rosbot_xl_gazebo simulation.launch.py

# Terminal 2: SLAM Toolbox
ros2 launch slam_toolbox online_async_launch.py

# Terminal 3: Nav2
ros2 launch nav2_bringup navigation_launch.py use_sim_time:=true

# Terminal 4: RViz2 with Nav2 panel
ros2 launch nav2_bringup rviz_launch.py
```

Then use the "Nav2 Goal" tool in RViz2 to set a navigation goal. The planner finds a path, the controller follows it, and if the robot gets stuck, recovery kicks in.

### DWB vs MPPI controller

The default local controller is DWB (Dynamic Window Approach Based). It generates a set of candidate velocity trajectories, scores them against a cost function, and selects the best one. It works but can produce jerky motion in tight spaces.

MPPI (Model Predictive Path Integral) is a newer addition to Nav2. It uses sampling-based optimization — generates a population of random trajectories by adding Gaussian noise to the nominal trajectory, evaluates each one, and computes a weighted average. It handles dynamic obstacles and narrow passages better than DWB and produces smoother motion.

Enable MPPI:

```yaml
# nav2_params.yaml
controller_server:
  ros__parameters:
    controller_plugins: ["FollowPath"]
    FollowPath:
      plugin: "nav2_mppi_controller::MPPIController"
```

References:

- https://husarion.com/tutorials/ros2-tutorials/8-slam/
- https://husarion.com/tutorials/ros2-tutorials/9-navigation/
- https://docs.nav2.org/
- https://github.com/SteveMacenski/slam_toolbox
