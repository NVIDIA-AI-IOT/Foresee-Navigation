//
// Created by nvidia on 6/21/18.
//

#ifndef SUPERPIXEL_SEG_FLOODFILLSSDSUPERPIXELALGORITHM_HPP
#define SUPERPIXEL_SEG_FLOODFILLSSDSUPERPIXELALGORITHM_HPP

#include "SsdSingleShotSuperClassifier.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>


namespace cove {

class FloodFillSsdSuperpixelAlgorithm {
public:
    FloodFillSsdSuperpixelAlgorithm(int width, int height,
                                    int regionSize, double ssdEpsilon, int blurKernel, float blurDev, int iterations,
                                    float ratio, int cannyLow, int cannyHigh);
    cv::Ptr<cv::ximgproc::SuperpixelLSC> getSuperPixels(cv::InputArray src, cv::OutputArray dst, cv::OutputArray _resizedCamera);
    void Run(cv::InputArray src, cv::OutputArray dst, cv::OutputArray _resizedCamera=cv::noArray());
private:
    int width;
    int height;
    double ssdEpsilon;
    int blurKernel;
    float blurDev;
    int iterations;
    int regionSize;
    float ratio;
    int cannyLow;
    int cannyHigh;

    cv::Mat frame;
    cv::Mat cielab;
    cv::Mat mask;
    cv::Mat edges;
    cv::Mat labels;
};

}
#endif //SUPERPIXEL_SEG_FLOODFILLSSDSUPERPIXELALGORITHM_HPP
