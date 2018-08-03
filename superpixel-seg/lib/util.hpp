//
// Created by nvidia on 6/25/18.
//

#ifndef SUPERPIXEL_SEG_UTIL_HPP
#define SUPERPIXEL_SEG_UTIL_HPP

#include <string>
#include <opencv2/opencv.hpp>

namespace cove {
namespace util {
std::string getFileName(const std::string& filePath, bool withExtension = true, char seperator = '/');


void overlayMask(cv::InputArray src, cv::InputArray mask, cv::Scalar color, cv::OutputArray dst);
}
}


#endif //SUPERPIXEL_SEG_UTIL_HPP
