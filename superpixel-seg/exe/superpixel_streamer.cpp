//
// Created by nvidia on 6/25/18.
//

//
// Created by nvidia on 6/22/18.
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
#include "../lib/RTreesClassifier.hpp"
#include "../lib/util.hpp"

using namespace cv;
using namespace std;
using namespace cv::ximgproc;
using namespace cove;

int main(int argc, char** argv) {
#ifdef DEBUG
    namedWindow("params", 1);
#endif
    ALGORITHM_PARAM(blurKernel, 3, 10)
    ALGORITHM_PARAM(iterations, 10, 20)
    ALGORITHM_PARAM(blurDev, 0, 100)
    ALGORITHM_PARAM(regionSize, 7, 50)
    ALGORITHM_PARAM(ratio, 70, 200)
    ALGORITHM_PARAM(cannyLow, 70, 150)
    ALGORITHM_PARAM(ssdEpsilonSlider, 10, 250)

    int frameno = 0;
    Mat frame;

    RTreesClassifier classifier;
//    Ptr<ml::RTrees> model;
//    model = ml::StatModel::load<ml::RTrees>("RTREES");
    if( classifier.loadModelFromPath("RTREES") ) {
        cout << "The classifier " << "RTREES" << " is loaded.\n";
    } else {
        cout << "Could not read the classifier " << "RTREES" << endl;
        return 1;
    }

    FloodFillSsdSuperpixelAlgorithm algorithm(80, 60, regionSize, static_cast<float>(ssdEpsilonSlider) / 10, blurKernel, blurDev, iterations,
                                              static_cast<float>(ratio) / 1000, cannyLow, cannyLow*2.5);


    if (argc < 2) {
        cout << "specify in arg[1] a camera id" << endl;
        return 1;
    }

    VideoCapture cap(stoi(argv[1], nullptr, 10));
    for (;;) {
        clock_t start=CLOCK();

#ifdef DEBUG
        if (blurKernel % 2 == 0) {
            blurKernel++;
        }
#endif

        cap >> frame;
        if (frame.empty()) {
            cout << "frame empty" << endl;
            break;
        }

        Mat floorMask;
        Mat sizedFrame;
        Mat coloredSizedFrame;

        auto super = algorithm.getSuperPixels(frame, floorMask, sizedFrame);
        cvtColor(sizedFrame, coloredSizedFrame, COLOR_BGR2Lab);
        super->iterate(iterations);
        super->enforceLabelConnectivity(15);
        Mat spxLabels;
        super->getLabels(spxLabels);
        classifier.getFloorMask(spxLabels, coloredSizedFrame, super->getNumberOfSuperpixels(), floorMask);

        dilate(floorMask, floorMask, Mat(), Point(-1, -1), 2);
        erode(floorMask, floorMask, Mat(), Point(-1, -1), 2);

        util::overlayMask(sizedFrame, floorMask, Scalar(0,0,255), sizedFrame);
        SHOW_ALWAYS("mask", sizedFrame);


        // =========== time fps and exit ===========
        double dur = CLOCK()-start;
        printf("avg time per frame %f ms. fps %f. frameno = %d\n",avgdur(dur),avgfps(),frameno++ );

        int key = waitKey(10);
        if (key == 27) {
            break;
        }
    }
    return 0;
}
