//
// Created by nvidia on 6/13/18.
//

#ifndef SUPERPIXEL_SEG_PIXELCLASSIFIER_HPP
#define SUPERPIXEL_SEG_PIXELCLASSIFIER_HPP

#include "SpVec.hpp"
#include <opencv2/ximgproc.hpp>
#include <boost/functional/hash.hpp>

namespace cove {

class SsdSingleShotSuperClassifier {
public:
    explicit SsdSingleShotSuperClassifier(cv::Ptr<cv::ximgproc::SuperpixelLSC> &&x, cv::InputArray _frame,
                                          int _regionSize, double _ssdEpsilon);

    void getFloorMask(cv::OutputArray _dst);

private:
    void generateFeatureVectors();

    void buildGraph();

    void fillTrainingData(int rows = 1, int cols = 3);

    void prepareSsdClassifier();

    bool satisfiesSsd(const SpVec &v);

    std::vector<int> doBfs();

    int regionSize;
    cove::SpVec trainingMean;
    cove::SpVec populationStdDev;
    double maxTrainingSsd;
    double ssdEpsilon;
    std::vector<int> trainingLabels;
    std::vector<cove::SpVec> pixelVecs;
    cv::Mat cielab;
    cv::Mat labels;
    cv::Ptr<cv::ximgproc::SuperpixelLSC> super;
    cove::Graph g;
};

}
#endif // SUPERPIXEL_SEG_PIXELCLASSIFIER_HPP
