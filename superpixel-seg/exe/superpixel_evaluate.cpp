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
    bool fastFinish = false;
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
    Mat label;

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

    vector<double> ius;
    for (int idx = static_cast<int>(argc*0.8); idx < argc; idx++) {
        clock_t start=CLOCK();

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
        SHOW_ALWAYS("frame", frame)
        SHOW_ALWAYS("label", label)
        if (frame.empty() || label.empty()) {
            cout << "frame empty" << endl;
            break;
        }

        Mat floorMask;
        Mat resizedLabel;
        Mat sizedFrame;




        Mat coloredSizedFrame;
        auto super = algorithm.getSuperPixels(frame, floorMask, sizedFrame);
        cvtColor(sizedFrame, coloredSizedFrame, COLOR_BGR2Lab);
        super->iterate(iterations);
        super->enforceLabelConnectivity(15);
        Mat spxLabels;
        // floor inference
        super->getLabels(spxLabels);
        cout << "t" << endl;
        classifier.getFloorMask(spxLabels, coloredSizedFrame, super->getNumberOfSuperpixels(), floorMask);
        cout << "t" << endl;

//
//
//        Mat pxSamples = Mat::zeros(0, 10, CV_32F);
//        for (int i = 0; i < super->getNumberOfSuperpixels(); i++) {
//            Mat mask;
//            inRange(spxLabels, Scalar(i), Scalar(i), mask);
//            pxSamples.push_back(RTreesClassifier::getData(mask, coloredSizedFrame));
//            // inference
//        }
//        Mat labels;
//        model->predict(pxSamples, labels);
//        cout << labels.size() << endl;
//        cout << labels.depth() << endl;
//        // compute floorMask
//        Mat temp;
//        floorMask = Mat::zeros(sizedFrame.size(), CV_8U);
//        for (int i = 0; i < super->getNumberOfSuperpixels(); i++) {
//            if (labels.at<float>(0, i) > 2.5) {
//                continue;
//            }
//            inRange(spxLabels, Scalar(i), Scalar(i), temp);
//            bitwise_or(floorMask, temp, floorMask);
//        }

        dilate(floorMask, floorMask, Mat(), Point(-1, -1), 2);
        erode(floorMask, floorMask, Mat(), Point(-1, -1), 2);
        resize(label, resizedLabel, floorMask.size(), 0, 0, INTER_NEAREST);
        SHOW_ALWAYS("resize", resizedLabel)
        SHOW_ALWAYS("calced floor", floorMask)
        util::overlayMask(sizedFrame, floorMask, Scalar(0, 0, 255), sizedFrame);
        SHOW_ALWAYS("display", sizedFrame);

        // get labeled floor mask
        Mat tmp;
        inRange(resizedLabel, Scalar(0,0,255), Scalar(0,0,255), tmp);
        inRange(resizedLabel, Scalar(0,255,0), Scalar(0,255,0), resizedLabel);
        bitwise_or(tmp, resizedLabel, resizedLabel);

        // iu calculation
        bitwise_and(resizedLabel, floorMask, tmp);
        SHOW_ALWAYS("TRUE POS", tmp);
        int truePositive = countNonZero(tmp);
        bitwise_xor(resizedLabel, floorMask, tmp);
        SHOW_ALWAYS("FALSE *", tmp);
        int falsePosandNeg = countNonZero(tmp);
        double iu = truePositive / static_cast<double>(falsePosandNeg + truePositive);
        ius.push_back(iu);
        cout << "IU: " << iu << endl;


//        true positive / (true positive + false positive + false negative)



        // mask the floor and etc

//        Mat zeros(floorMask.size(), CV_8UC3, Scalar(0));
//        Mat red(floorMask.size(), CV_8UC3, Scalar(0, 0, 255));
//        red.copyTo(zeros, floorMask);
//        addWeighted(resized, 0.7, zeros, 0.3, 0.0, resized);

//        SHOW_ALWAYS("output", resized);
        // =========== time fps and exit ===========
        double dur = CLOCK()-start;
        printf("avg time per frame %f ms. fps %f. frameno = %d\n",avgdur(dur),avgfps(),frameno++ );
        cout << argv[idx] << endl;
        if (fastFinish) {
            continue;
        }
        int key = waitKey(1000);
        if (key == 27) {
            break;
        } else if (key == 99) {
            fastFinish = true;
        }

    }
    double sum = std::accumulate(ius.begin(), ius.end(), 0.0);
    double mean = sum / ius.size();
    std::vector<double> diff(ius.size());
    std::transform(ius.begin(), ius.end(), diff.begin(), [mean](double x) { return x - mean; });
    double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
    double stdev = std::sqrt(sq_sum / ius.size());

    cout << "MEAN: " << mean << endl;
    cout << "STDV: " << stdev << endl;




    return 0;
}
