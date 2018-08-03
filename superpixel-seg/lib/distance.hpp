//
// Created by nvidia on 6/18/18.
//

#ifndef SUPERPIXEL_SEG_DISTANCE_HPP
#define SUPERPIXEL_SEG_DISTANCE_HPP

#include <opencv2/opencv.hpp>
typedef cv::Matx<double, 3, 3> CAMERA_MAT;
typedef cv::Matx<double, 1, 5> DISTORT_MAT;
typedef cv::Matx<double, 3, 4> EXTRINSICS_MAT;

extern const CAMERA_MAT CAMERA_MATRIX;
extern const DISTORT_MAT DISTORTION_MATRIX;
extern const EXTRINSICS_MAT EXTRINSICS_MATRIX;


void undistortInternal(cv::InputArray src, cv::OutputArray dst);

void transformToTopDown(cv::InputArray src, cv::OutputArray dst);

#endif //SUPERPIXEL_SEG_DISTANCE_HPP
