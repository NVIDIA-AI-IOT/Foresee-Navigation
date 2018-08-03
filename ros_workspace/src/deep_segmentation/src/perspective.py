#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import cv2

"""
//
// Created by nvidia on 6/18/18.
//
#define DEBUG

#include <opencv2/opencv.hpp>
#include "distance.hpp"
#include "superpixel_seg.hpp"

using namespace cv;
using namespace std;

const int DST_WIDTH = 800;
const int DST_HEIGHT = 500;

const float SCALING = 5.0f;
const float DIST_TO_FRONT_EDGE = 32.25f * SCALING;
const float HALF_WIDTH = 25.0f / 2.0f  * SCALING;
const float HEIGHT = 17.0f  * SCALING;


const Point2f dstLowerRight(DST_WIDTH / 2.0f + HALF_WIDTH, DST_HEIGHT - DIST_TO_FRONT_EDGE);
const Point2f dstUpperRight(DST_WIDTH / 2.0f + HALF_WIDTH, DST_HEIGHT - (DIST_TO_FRONT_EDGE + HEIGHT));
const Point2f dstUpperLeft(DST_WIDTH / 2.0f - HALF_WIDTH, DST_HEIGHT - (DIST_TO_FRONT_EDGE + HEIGHT));
const Point2f dstLowerLeft(DST_WIDTH / 2.0f - HALF_WIDTH, DST_HEIGHT - DIST_TO_FRONT_EDGE);

const int X_MAX = 640;
const int Y_MAX = 480;


int main() {
#ifdef DEBUG
    namedWindow("params", 1);
#endif
    ALGORITHM_PARAM(lowerRightX, 430, X_MAX)
    ALGORITHM_PARAM(lowerRightY, 474, Y_MAX)
    ALGORITHM_PARAM(upperRightX, 391, X_MAX)
    ALGORITHM_PARAM(upperRightY, 391, Y_MAX)
    ALGORITHM_PARAM(upperLeftX, 222, X_MAX)
    ALGORITHM_PARAM(upperLeftY, 390, Y_MAX)
    ALGORITHM_PARAM(lowerLeftX, 187, X_MAX)
    ALGORITHM_PARAM(lowerLeftY, 472, Y_MAX)


    VideoCapture cap(2);
    if(!cap.isOpened()) {
        return -1;
    }
    Mat frame, pin, topDown;

    for (;;) {
//        cap >> frame;
        frame = Mat(Size(640, 480), CV_8UC1);
        circle(frame, Point2f(320, 480), 50, Scalar(255), -1);
        if (frame.empty()) {
            cout << "frame empty" << endl;
            break;
        }
        const Point2f srcLowerRight(lowerRightX, lowerRightY);
        const Point2f srcUpperRight(upperRightX, upperRightY);
        const Point2f srcUpperLeft(upperLeftX, upperLeftY);
        const Point2f srcLowerLeft(lowerLeftX, lowerLeftY);

        undistortInternal(frame, pin);
        SHOW("pin", pin)

        circle(pin, srcLowerRight, 3, Scalar(0,0,255));
        circle(pin, srcUpperRight, 3, Scalar(0,0,255));
        circle(pin, srcLowerLeft, 3, Scalar(0,0,255));
        circle(pin, srcUpperLeft, 3, Scalar(0,0,255));

        SHOW("pinDraw", pin)

        Matx<float, 4, 2> srcQuadrangle(
                srcLowerRight.x, srcLowerRight.y,
                srcUpperRight.x, srcUpperRight.y,
                srcUpperLeft.x, srcUpperLeft.y,
                srcLowerLeft.x, srcLowerLeft.y
        );
        Matx<float, 4, 2> dstQuadrangle(
                dstLowerRight.x, dstLowerRight.y,
                dstUpperRight.x, dstUpperRight.y,
                dstUpperLeft.x, dstUpperLeft.y,
                dstLowerLeft.x, dstLowerLeft.y
        );

        Mat pTransform = getPerspectiveTransform(srcQuadrangle, dstQuadrangle);
        warpPerspective(pin, topDown, pTransform, Size(DST_WIDTH, DST_HEIGHT));
        SHOW("topDown", topDown);

        if (waitKey(15) == 27) {
            break;
        }
    }
}
"""

"""
const EXTRINSICS_MAT EXTRINSICS_MATRIX(
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0
)
"""




CAMERA_MATRIX = np.array([
    [702.11942344,      0.0,                976.36908128],
    [0.0,               702.9060824 * 0.5,        493.78298053 * 0.5],
    [0.0,               0.0,                1.0],
])

CAMERA_MATRIX = CAMERA_MATRIX * 0.75
CAMERA_MATRIX[2][2] = 1.0
print(CAMERA_MATRIX)

DISTORTION_MATRIX = np.array([
    0.0436999, -0.0642659, 0.00301327, 0.0008171, 0.01355263
])


def undistort_internal(img):
    return cv2.undistort(img, CAMERA_MATRIX, DISTORTION_MATRIX)


DST_WIDTH = 800
#DST_HEIGHT = 180
DST_HEIGHT = 500

SCALING = 10.0
DIST_TO_FRONT_EDGE = 0 * SCALING
HALF_WIDTH = 25.0 / 2.0 * SCALING
HEIGHT = 17.0 * SCALING


PERSP_DST = np.array([
    [DST_WIDTH / 2.0 + HALF_WIDTH, DST_HEIGHT - DIST_TO_FRONT_EDGE],
    [DST_WIDTH / 2.0 + HALF_WIDTH, DST_HEIGHT - (DIST_TO_FRONT_EDGE + HEIGHT)],
    [DST_WIDTH / 2.0 - HALF_WIDTH, DST_HEIGHT - (DIST_TO_FRONT_EDGE + HEIGHT)],
    [DST_WIDTH / 2.0 - HALF_WIDTH, DST_HEIGHT - DIST_TO_FRONT_EDGE]
], dtype=np.float32)

PERSP_SRC = np.array([
    [339, 180],
    [303, 109],
    [177, 109],
    [139, 180]
], dtype=np.float32)

PERSP_TRANSFORM = cv2.getPerspectiveTransform(PERSP_SRC, PERSP_DST)

def undistorted_to_top_down(img):
    return cv2.warpPerspective(img, PERSP_TRANSFORM, (DST_WIDTH, DST_HEIGHT), flags=cv2.INTER_NEAREST)



def distorted_cam_to_top_down(img):
    e = undistort_internal(img)
    p_transform = PERSP_TRANSFORM
    return cv2.warpPerspective(e, p_transform, (DST_WIDTH, DST_HEIGHT), flags=cv2.INTER_NEAREST)
