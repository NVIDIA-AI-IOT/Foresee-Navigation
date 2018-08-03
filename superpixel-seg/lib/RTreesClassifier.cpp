//
// Created by nvidia on 6/25/18.
//

#include "RTreesClassifier.hpp"

namespace cove {
using namespace cv;
using namespace std;

Mat RTreesClassifier::getData(InputOutputArray mask, InputArray labColoredSizedFrame) {
    Moments m = moments(mask, false);
    Point center(m.m10/m.m00, m.m01/m.m00);


    Scalar mean, stddev;
    meanStdDev(labColoredSizedFrame, mean, stddev, mask);



    // find squareness and solidity factor
    vector<vector<Point>> contours;
    findContours(mask, contours, noArray(), RETR_LIST, CHAIN_APPROX_SIMPLE);
    float circ;
    float solidity;
    float area;
    if (contours.size() < 1) {
        circ = 0;
        solidity = 0;
        area = 0;
    } else {
        vector<Point> hull;
        convexHull(contours.at(0), hull);
        double hull_area = contourArea(hull);
        area = static_cast<float>(contourArea(contours.at(0)));
        solidity = static_cast<float>(area / hull_area);
        double p = arcLength(contours.at(0), true);
        circ = static_cast<float>(4 * M_PI * area / (p * p));
        if (isnan(circ) || isinf(circ)) {
            circ = 0;
        }
        if (isnan(solidity) || isinf(solidity)) {
            solidity = 0;
        }
        if (isnan(area) || isinf(area)) {
            area = 0;
        }
    }
    Mat thisSample = Mat::zeros(1, 10, CV_32F);
    thisSample.at<float>(0, 0) = area;
    thisSample.at<float>(0, 1) = circ;
    thisSample.at<float>(0, 2) = static_cast<float>(mean[0]);
    thisSample.at<float>(0, 3) = static_cast<float>(mean[1]);
    thisSample.at<float>(0, 4) = static_cast<float>(mean[2]);
    thisSample.at<float>(0, 5) = static_cast<float>(stddev[0]);
    thisSample.at<float>(0, 6) = static_cast<float>(stddev[1]);
    thisSample.at<float>(0, 7) = static_cast<float>(stddev[2]);
    thisSample.at<float>(0, 8) = center.x;
    thisSample.at<float>(0, 9) = center.y;
    return thisSample;
}

bool RTreesClassifier::loadModelFromPath(std::string path) {
    treeModel = ml::StatModel::load<ml::RTrees>(path);
    return !treeModel.empty();
}

void RTreesClassifier::getFloorMask(cv::InputArray spxLabels, cv::InputArray labColoredSizedFrame, int numberOfSuperpixels, cv::OutputArray dst) {
    Mat pxSamples = Mat::zeros(0, 10, CV_32F);
    for (int i = 0; i < numberOfSuperpixels; i++) {
        Mat mask;
        inRange(spxLabels, Scalar(i), Scalar(i), mask);
        pxSamples.push_back(getData(mask, labColoredSizedFrame));
    }
    Mat labels;
    treeModel->predict(pxSamples, labels);
    // compute floorMask
    Mat temp;
    Mat _dst;
    dst.create(spxLabels.size(), CV_8UC1);
    dst.getMat() = Scalar(0);
    for (int i = 0; i < numberOfSuperpixels; i++) {
        if (labels.at<float>(0, i) > 2.5) {
            continue; // not floor
        }
        inRange(spxLabels, Scalar(i), Scalar(i), temp);
//        cout << "d" << endl;
        bitwise_or(dst, temp, dst);
    }
}


}
