//
// Created by nvidia on 6/25/18.
//

#ifndef SUPERPIXEL_SEG_RTREESCLASSIFIER_HPP
#define SUPERPIXEL_SEG_RTREESCLASSIFIER_HPP

#include <opencv2/opencv.hpp>

namespace cove {
class RTreesClassifier {
public:
    static cv::Mat getData(cv::InputOutputArray mask, cv::InputArray labColoredSizedFrame);
    bool loadModelFromPath(std::string path);
    void getFloorMask(cv::InputArray spxLabels, cv::InputArray labColoredSizedFrame, int numberOfSuperpixels, cv::OutputArray dst);
private:
    cv::Ptr<cv::ml::RTrees> treeModel;
};


}

#endif //SUPERPIXEL_SEG_RTREESCLASSIFIER_HPP
