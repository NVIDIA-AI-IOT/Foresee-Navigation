#!/usr/bin/env bash

export ROS_DISTRO=kinetic
set -ex

# Install wstool and rosdep.
#sudo apt-get update
sudo apt-get install -y python-wstool python-rosdep ninja-build

set +e
wstool init src
set -e

# Merge the cartographer_ros.rosinstall file and fetch code for dependencies.
wstool merge -t src src/localization.rosinstall
wstool merge -t src src/jackal.rosinstall
wstool update -t src


# Install deb dependencies.
# The command 'sudo rosdep init' will print an error if you have already
# executed it since installing ROS. This error can be ignored.
set +e
sudo rosdep init
set -e
rosdep update
# This command explicitly ignores errors
# There are several packages you cannot get from rosdep that this script
# installs some other way or are vendored.
echo '===== IGNORE ERRORS FROM THE FOLLOWING COMMAND ====='
rosdep install --from-paths src --ignore-src --rosdistro=${ROS_DISTRO} -y -r

# Install the driver for the IMU if it's not installed already
if ! python -c "from Adafruit_BNO055.BNO055 import BNO055"; then
    echo 'Installing IMU driver'
    git clone https://github.com/adafruit/Adafruit_Python_BNO055.git
    cd Adafruit_Python_BNO055
    sudo python setup.py install
    cd ..
else
    echo 'IMU driver already installed.'
fi

# In case the script was sourced
set +ex

echo 'Done.'
