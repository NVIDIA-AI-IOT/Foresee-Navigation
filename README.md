# Foresee Navigation

Semantic-Segmentation based autonomous indoor navigation for mobile robots.

## The Problem

The principle sensor for autonomously navigating robots is the LiDAR. However,
low-cost 2D LiDARs cannot detect many man-made obstacles such as mesh-like
railings, glass walls, or descending staircases, and 3D LiDARs that can detect
some of these are prohibitively expensive.

This project is a proof-of-concept showing that Deep Learning can provide a
solution for detecting such obstacles.

## Our Approach

The basis of our strategy is to integrate detected obstacle locations into
proven tools for autonomous navigation from the Robot Operating System.

The process begins by segmenting the floor using DeepLab with ResNet V2 50. To
learn about how this works in more detail, see
[`this`](https://github.com/NVIDIA-Jetson/Foresee-Navigation/tree/master/ros_workspace/src/deep_segmentation).

![Model Inference](https://i.imgur.com/ELv8eyl.png)

A perspective transform is then applied to get top-down view of safe and unsafe
areas. The resulting image is cut into discrete sectors, and the lowest point in
each sector is calculated. These points, after a linear transform to real-world
robot-relative coordinates, form a ROS pointcloud that is fed into our
navigation stack.

![Point Cloud, Top Down](https://i.imgur.com/90SgFs1.jpg)

This data is then fed into SLAM through `gmapping` and we use pathfinder and
nonlinear reference tracking to follow a path autonomously. See more information
about [pathfinder_ros](https://github.com/asinghani/pathfinder_ros) and
[reftracking_ros](https://github.com/asinghani/reftracking_ros).

## Our Platform

This repository includes an implementation of the above based on the Jackal
robot platform with a NVIDIA Jetson TX2 inside. We integrate the above pipeline
with a low-cost RPLiDAR A1. For a camera we used an iBUFFALO BSW20KM11BK mounted
about 25cm above the ground.

![Robot Image](https://i.imgur.com/QBjWTLJ.jpg)

## Repository Structure

The ROS workspace that is deployed to the robot is located in `ros_workspace`.
Inside `src`, `robot_control` contains the coordinating
launch files and safety nodes and `deep_segmentation` contains the Deep Learning
pipeline.

The `superpixel-seg` directory contains an earlier approach to segmenting the
floor based on superpixels and a random forest classifier. `SXLT` contains a
small labeling tool for segmentation.

Since this project depends on OpenCV, (and `superpixel-seg` on OpenCV_contrib),
a version of JetsonHacks's `buildOpenCVTX2` that allows enabling and disabling
certain contrib modules is included.

## Usage

Our code assumes a Jackal development platform with a BNO055 IMU integrated, an
RPLidar A1 with udev rules configured, and a 120-degree-fov iBUFFALO BSW20KM11BK
with distortion and mounting characteristics identical to our own.

Software dependencies include OpenCV, ROS, NumPy, and Tensorflow. To get
segmentation to work, it is highly recommended that you use CUDA 8 and
Tensorflow 1.3. CUDA 8 can be installed via JetPack 3.1 without re-flashing.

Begin by running `provision.sh` in the root directory. This downloads files
that could not be re-distributed for licensing reasons and the binary model
files.

To run the code, first setup the ROS Workspace. `cd ros_workspace`, then execute
`setup.sh`, and finally source `install_isolated.sh`. This will install all
necessary ROS packages and compile certain ones from source inline. If you want
to edit the code, remember to re-source `install_isolated.sh` after each edit.
Finally, `roslaunch robot_control auton.launch` and, in a separate terminal,
`rosrun driver_station ds.py`. To enable, use `C-x C-e` and press space to
disable.

## License

Apache 2.0, see [`LICENSE`](LICENSE).
