#!/usr/bin/env bash

# Copies open source files that we didn't write entirely

ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
cd "$ROOT_DIR"

# buildOpenCVTX2.sh
git submodule init


# deep_segmentation
DS_DIR="$ROOT_DIR/ros_workspace/src/deep_segmentation/src/"
wget "https://raw.githubusercontent.com/sthalles/deeplab_v3/794f4790edb240ef537ce3dc63ef23f9c3dae0b0/metrics.py" -O "$DS_DIR/metrics.py"
wget "https://raw.githubusercontent.com/sthalles/deeplab_v3/794f4790edb240ef537ce3dc63ef23f9c3dae0b0/network.py" -O "$DS_DIR/network.py"
wget "https://raw.githubusercontent.com/sthalles/deeplab_v3/794f4790edb240ef537ce3dc63ef23f9c3dae0b0/train.py" -O "$DS_DIR/train.py"
wget "https://raw.githubusercontent.com/sthalles/deeplab_v3/794f4790edb240ef537ce3dc63ef23f9c3dae0b0/preprocessing/training.py" -O "$DS_DIR/training.py"
wget "https://raw.githubusercontent.com/sthalles/deeplab_v3/794f4790edb240ef537ce3dc63ef23f9c3dae0b0/preprocessing/inception_preprocessing.py" -O "$DS_DIR/inception_preprocessing.py"
wget "https://raw.githubusercontent.com/sthalles/deeplab_v3/794f4790edb240ef537ce3dc63ef23f9c3dae0b0/resnet/resnet_utils.py" -O "$DS_DIR/resnet_utils.py"
wget "https://raw.githubusercontent.com/sthalles/deeplab_v3/794f4790edb240ef537ce3dc63ef23f9c3dae0b0/resnet/resnet_v2.py" -O "$DS_DIR/resnet_v2.py"
git apply "$ROOT_DIR/resnet-tf-1_3.patch"

mkdir -p "$ROOT_DIR/temp"
if [ ! -f "$ROOT_DIR/temp/model_all.tar.gz" ]; then
    wget "http://download945.mediafire.com/1yjbaybtmbeg/kvhzimialk8rtwn/model_all.tar.gz" -O "$ROOT_DIR/temp/model_all.tar.gz"
fi
tar -xaf "$ROOT_DIR/temp/model_all.tar.gz" -C "$DS_DIR"

# SXLT needs the C++ inih library
INIH_DIR="$ROOT_DIR/SXLT/vendor/inih/"
mkdir -p "$INIH_DIR"
wget "https://raw.githubusercontent.com/jtilly/inih/a3f04ad7bdffd4c407dee47dc8026505166f9ce2/INIReader.h" -O "$INIH_DIR/INIReader.h"
