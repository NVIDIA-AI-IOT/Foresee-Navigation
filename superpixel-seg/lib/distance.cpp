//
// Created by nvidia on 6/18/18.
//

#include "distance.hpp"

// RMS: 0.761314807611
// camera matrix:
// [[702.11942344   0.         976.36908128]
// [  0.         702.9060824  493.78298053]
// [  0.           0.           1.        ]]
// distortion coefficients:  [ 0.0436999  -0.0642659   0.00301327  0.0008171   0.01355263]

using namespace cv;

const CAMERA_MAT CAMERA_MATRIX(
        702.11942344,0.0,976.36908128,
        0.0, 702.9060824, 493.78298053,
        0.0, 0.0, 1.0
);
const DISTORT_MAT DISTORTION_MATRIX(
        0.0436999, -0.0642659, 0.00301327, 0.0008171, 0.01355263
);

const EXTRINSICS_MAT EXTRINSICS_MATRIX(
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0
);

void undistortInternal(InputArray src, OutputArray dst) {
    undistort(src, dst, CAMERA_MATRIX, DISTORTION_MATRIX);
}

void transformToTopDown(InputArray src, OutputArray dst) {

}