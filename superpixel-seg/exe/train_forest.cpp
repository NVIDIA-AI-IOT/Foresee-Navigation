//
// Created by nvidia on 6/22/18.
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

inline TermCriteria TC(int iters, double eps)
{
    return TermCriteria(TermCriteria::MAX_ITER + (eps > 0 ? TermCriteria::EPS : 0), iters, eps);
}

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
    ALGORITHM_PARAM(ssdEpsilonSlider, 10, 250)

    int frameno=0;
    Mat frame;
    Mat label;


    FloodFillSsdSuperpixelAlgorithm algorithm(80, 60, regionSize, static_cast<float>(ssdEpsilonSlider) / 10, blurKernel, blurDev, iterations,
                                              static_cast<float>(ratio) / 1000, cannyLow, cannyLow*2.5);

    vector<double> ius;

    Mat samples = Mat::zeros(0, 10, CV_32F);
    Mat responses = Mat::zeros(0, 1, CV_32S);

    for (int idx = 1; idx < argc*0.8; idx++) {
        clock_t start = CLOCK();

#ifdef DEBUG
        if (blurKernel % 2 == 0) {
            blurKernel++;
        }
#endif

        string path(argv[idx]);
        string filename = util::getFileName(path, true);
        string labelPath = "img/labels/" + filename;
        frame = imread(path);
        label = imread(labelPath);
//        SHOW_ALWAYS("frame", frame)
//        SHOW_ALWAYS("label", label)
        if (frame.empty() || label.empty()) {
            cout << "frame empty" << endl;
            break;
        }

        Mat floorMask;
        Mat resizedLabel;
        algorithm.Run(frame, floorMask, resizedLabel);
        resize(label, resizedLabel, resizedLabel.size(), 0, 0, INTER_NEAREST);
//        SHOW_ALWAYS("resize", resizedLabel)
//        SHOW_ALWAYS("calced floor", floorMask)
        Mat carpet;
        Mat hardfloor;
        Mat lab;
        Mat sizedFrame;
        inRange(resizedLabel, Scalar(0, 0, 255), Scalar(0, 0, 255), carpet);
        inRange(resizedLabel, Scalar(0, 255, 0), Scalar(0, 255, 0), hardfloor);

        auto super = algorithm.getSuperPixels(frame, lab, sizedFrame);
        cvtColor(sizedFrame, sizedFrame, COLOR_BGR2Lab);
        super->iterate(iterations);
        super->enforceLabelConnectivity(15);
        Mat spxLabels;
        super->getLabels(spxLabels);
        for (int i = 0; i < super->getNumberOfSuperpixels(); i++) {
            Mat mask;
            inRange(spxLabels, Scalar(i), Scalar(i), mask);
            // classify on label
            int label;
            Mat temp;
            bitwise_and(carpet, mask, temp);
            if (countNonZero(temp) > 7 * 7 / 2) {
                // is carpet
                label = 1;
                goto add_data;
            }
            bitwise_and(hardfloor, mask, temp);
            if (countNonZero(temp) > 7 * 7 / 2) {
                // is hardfloor
                label = 2;
                goto add_data;
            }
            // is bg
            label = 3;
            goto add_data;


            add_data:
            samples.push_back(RTreesClassifier::getData(mask, sizedFrame));;

            Mat thisResponse = Mat::zeros(1, 1, CV_32S);
            thisResponse.at<int>(0, 0) = label;
            responses.push_back(thisResponse);

        }

        // =========== time fps and exit ===========
        double dur = CLOCK() - start;
        printf("avg time per frame %f ms. fps %f. frameno = %d\n", avgdur(dur), avgfps(), frameno++);
        cout << argv[idx] << endl;

    }
    auto data = ml::TrainData::create(samples, ml::ROW_SAMPLE, responses);

    auto rtrees = ml::RTrees::create();
    rtrees->setMaxDepth(20);
    rtrees->setMinSampleCount(10);
    rtrees->setRegressionAccuracy(0);
    rtrees->setUseSurrogates(false);
    rtrees->setMaxCategories(15);
    rtrees->setPriors(Mat());
    rtrees->setCalculateVarImportance(true);
    rtrees->setActiveVarCount(0);
    rtrees->setTermCriteria(TC(100, 0.01f));
    rtrees->train(data);
    rtrees->save("RTREES");

    return 0;
}
