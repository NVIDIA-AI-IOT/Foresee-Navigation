//
// Created by nvidia on 6/25/18.
//

#include "util.hpp"

namespace cove {
namespace util {

using std::string;
using namespace cv;

string getFileName(const string& filePath, bool withExtension, char seperator) {
    // Get last dot position
    std::size_t dotPos = filePath.rfind('.');
    std::size_t sepPos = filePath.rfind(seperator);
    size_t firstPos;
    if (sepPos == string::npos) {
        firstPos = 0;
    } else {
        firstPos = sepPos + 1;
    }

    size_t lastPos;
    if (dotPos == string::npos) {
        lastPos = string::npos;
    } else {
        lastPos = withExtension ? string::npos : dotPos;
    }

    return filePath.substr(firstPos, lastPos - firstPos);
}

void overlayMask(cv::InputArray src, cv::InputArray mask, cv::Scalar color, cv::OutputArray dst) {
    CV_Assert(src.size() == mask.size());
    const Mat colorMat(mask.size(), src.type(), color);
    Mat colorMask = Mat::zeros(mask.size(), colorMat.type());
    colorMat.copyTo(colorMask, mask);
    addWeighted(src, 0.7, colorMask, 0.3, 0.0, dst);
}

}
}
