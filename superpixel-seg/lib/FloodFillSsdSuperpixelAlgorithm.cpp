//
// Created by nvidia on 6/21/18.
//

#include "FloodFillSsdSuperpixelAlgorithm.hpp"
#include "superpixel_seg.hpp"
#include <opencv2/ximgproc.hpp>

namespace cove {

using namespace cv;
using namespace cv::ximgproc;

FloodFillSsdSuperpixelAlgorithm::FloodFillSsdSuperpixelAlgorithm(int width, int height,
                                                                 int regionSize, double ssdEpsilon,
                                                                 int blurKernel, float blurDev, int iterations,
                                                                 float ratio, int cannyLow,
                                                                 int cannyHigh) : width(width), height(height),
                                                                                  regionSize(regionSize),
                                                                                  ssdEpsilon(ssdEpsilon),
                                                                                  blurKernel(blurKernel),
                                                                                  iterations(iterations),
                                                                                  blurDev(blurDev),
                                                                                  cannyLow(cannyLow),
                                                                                  cannyHigh(cannyHigh),
                                                                                  ratio(ratio)
                                                                                  {
    frame = Mat();
    cielab = Mat();
    mask = Mat();
    edges = Mat();
    labels = Mat();
}

Ptr<SuperpixelLSC> FloodFillSsdSuperpixelAlgorithm::getSuperPixels(InputArray src, OutputArray dst, OutputArray _resizedCamera) {

    SHOW("camera", src)

    cvtColor(src, edges, COLOR_BGR2GRAY);
    Canny(edges, edges, cannyLow, cannyHigh, 3, true);
    dilate(edges, edges, getStructuringElement(MORPH_RECT, Size(5, 5)));
    resize(edges, edges, Size(width, height), 0, 0, INTER_NEAREST);
    SHOW("canny down", edges)


    resize(src, dst, Size(width, height), 0, 0, INTER_AREA);
    SHOW("downsized", dst);


    dst.copyTo(_resizedCamera);
//    Mat resizedCamera = _resizedCamera.getMat();
//    if(resizedCamera.data )
//    {
//        dst.copyTo(resizedCamera);
//        DBP("COPIEDAKLSJDKLASJKLDJASKLDJKLASJLDJASKL")
//    }

    GaussianBlur(dst, dst, Size(blurKernel, blurKernel), blurDev, blurDev);

//    const Mat yellow(dst.size(), CV_8UC3, Scalar(0, 65, 255));
    cvtColor(dst, dst, COLOR_BGR2Lab);
    dst.copyTo(cielab);
    const Mat yellow(dst.size(), CV_8UC3, Scalar(255, 255, 255));
    yellow.copyTo(dst, edges);
    SHOW("colored", dst);

#ifdef THREE_CHANNEL
    Ptr<SuperpixelLSC> super = ximgproc::createSuperpixelLSC(dst, regionSize, ratio);
#else
    Mat ab(dst.size(), CV_8UC2);
    std::vector<int> fromTo = {1, 0, 2, 1};
    mixChannels(dst, ab, fromTo);

    SHOW_CHANNEL("a channel", ab, 0)
    SHOW_CHANNEL("b channel", ab, 1)

    Ptr<SuperpixelLSC> super = ximgproc::createSuperpixelLSC(ab, regionSize, ratio);
#endif
    return super;
}

void FloodFillSsdSuperpixelAlgorithm::Run(InputArray src, OutputArray dst, OutputArray _resizedCamera) {

    auto super = this->getSuperPixels(src, dst, _resizedCamera);

    super->iterate(iterations);
    super->enforceLabelConnectivity(15);

    super->getLabelContourMask(mask);
    SHOW("mask", mask);
    super->getLabels(labels);

    SsdSingleShotSuperClassifier classifier(std::move(super), cielab, regionSize, ssdEpsilon);
    classifier.getFloorMask(dst);
}

}