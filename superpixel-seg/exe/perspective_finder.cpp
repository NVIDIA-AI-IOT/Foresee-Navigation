//
// Created by nvidia on 6/18/18.
//
#define DEBUG

#include <opencv2/opencv.hpp>
#include "../lib/distance.hpp"
#include "../lib/superpixel_seg.hpp"

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
const Point2f dstLowerLeft(DST_WIDTH / 2.0f - HALF_WIDTH, DST_HEIGHT - DIST_TO_FRONT_EDGE);
const Point2f dstUpperLeft(DST_WIDTH / 2.0f - HALF_WIDTH, DST_HEIGHT - (DIST_TO_FRONT_EDGE + HEIGHT));

const int X_MAX = 640;
const int Y_MAX = 480;


int main() {
#ifdef DEBUG
    namedWindow("params", 1);
#endif
    ALGORITHM_PARAM(lowerRightX, 441, X_MAX)
    ALGORITHM_PARAM(lowerRightY, 474, Y_MAX)
    ALGORITHM_PARAM(upperRightX, 405, X_MAX)
    ALGORITHM_PARAM(upperRightY, 391, Y_MAX)
    ALGORITHM_PARAM(upperLeftX, 222, X_MAX)
    ALGORITHM_PARAM(upperLeftY, 391, Y_MAX)
    ALGORITHM_PARAM(lowerLeftX, 199, X_MAX)
    ALGORITHM_PARAM(lowerLeftY, 473, Y_MAX)


    VideoCapture cap(1);
    if(!cap.isOpened()) {
        return -1;
    }
    Mat frame, pin, topDown;

    for (;;) {
        cap >> frame;
//        frame = Mat(Size(640, 480), CV_8UC1);
//        circle(frame, Point2f(320, 480), 50, Scalar(255), -1);
//        if (frame.empty()) {
//            cout << "frame empty" << endl;
//            break;
//        }
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