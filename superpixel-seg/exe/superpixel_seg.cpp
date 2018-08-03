//
// Created by nvidia on 6/7/18.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>
#include <algorithm> // for copy
#include <iterator> // for ostream_iterator
#include <boost/graph/graph_utility.hpp>
#include "../lib/SpVec.hpp"
#include "../lib/superpixel_seg.hpp"
#include "../lib/distance.hpp"

#include "../lib/FloodFillSsdSuperpixelAlgorithm.hpp"

using namespace cv;
using namespace std;
using namespace cv::ximgproc;


using namespace cove;

int main(int argc, char** argv) {
    int idx = 1;
#ifdef DEBUG
    namedWindow("params", 1);
#endif
    ALGORITHM_PARAM(blurKernel, 3, 10)
    ALGORITHM_PARAM(iterations, 10, 20)
    ALGORITHM_PARAM(blurDev, 0, 100)
    ALGORITHM_PARAM(regionSize, 7, 50)
    ALGORITHM_PARAM(ratio, 70, 200)
    ALGORITHM_PARAM(cannyLow, 70, 150)
    ALGORITHM_PARAM(ssdEpsilonSlider, 15, 250)
    ALGORITHM_PARAM(meanH, 10, 50)
    ALGORITHM_PARAM(meanL, 10, 50)
    ALGORITHM_PARAM(meanS, 10, 50)
    ALGORITHM_PARAM(stdH, 10, 50)
    ALGORITHM_PARAM(stdL, 10, 50)
    ALGORITHM_PARAM(stdS, 10, 50)
    ALGORITHM_PARAM(circ, 10, 50)



#ifdef CAMERA
    VideoCapture cap(2); // open the default camera
//    VideoCapture cap("sample.mp4"); // open the video
    if(!cap.isOpened()) {
        return -1;
    }
#endif
    int frameno=0;

//    Mat downsampled(60, 80, CV_8U, Scalar(0));
//    Mat final;
    Mat frame;
//    Mat cielab;
//    Mat mask;



    for (;;) {
        FloodFillSsdSuperpixelAlgorithm algorithm(80, 60, regionSize, static_cast<float>(ssdEpsilonSlider) / 10, blurKernel, blurDev, iterations,
                                                  static_cast<float>(ratio) / 1000, cannyLow, cannyLow*2.5);
        weights[0] = static_cast<double>(meanH) / 10;
        weights[1] = static_cast<double>(meanL) / 10;
        weights[2] = static_cast<double>(meanS) / 10;
        weights[3] = static_cast<double>(stdH) / 10;
        weights[4] = static_cast<double>(stdL) / 10;
        weights[5] = static_cast<double>(stdS) / 10;
        weights[6] = static_cast<double>(circ) / 10;





        clock_t start=CLOCK();

#ifdef DEBUG
        if (blurKernel % 2 == 0) {
            blurKernel++;
        }

#endif
#ifdef CAMERA
        cap >> frame; // get a new frame from camera
#else
        frame = imread(argv[idx]);
#endif
        if (frame.empty()) {
            cout << "frame empty" << endl;
            break;
        }

        Mat floorMask;
        Mat resized;
        algorithm.Run(frame, floorMask, resized);

        Mat zeros(floorMask.size(), CV_8UC3, Scalar(0));
        Mat red(floorMask.size(), CV_8UC3, Scalar(0, 0, 255));
        red.copyTo(zeros, floorMask);
        addWeighted(resized, 0.7, zeros, 0.3, 0.0, resized);


        SHOW_ALWAYS("output", resized);


        // =========== time fps and exit ===========
        double dur = CLOCK()-start;
        printf("avg time per frame %f ms. fps %f. frameno = %d\n",avgdur(dur),avgfps(),frameno++ );
        cout << argv[idx] << endl;
        switch(waitKey(1)) {
            case 27:
                exit(0);
            case 108:
                idx++;
                break;
            case 106:
                idx--;
                break;
        }
    }

    return 0;
}
