//
// Created by nvidia on 6/7/18.
//

#ifndef SUPERPIXEL_SEG_SUPERPIXEL_SEG_HPP
#define SUPERPIXEL_SEG_SUPERPIXEL_SEG_HPP

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>
#include <utility>
#include "fps.hpp"
#include <algorithm> // for copy
#include <iterator> // for ostream_iterator
#include <cmath>
#include "SpVec.hpp"

//#define DEBUG
//#define CAMERA
//#define THREE_CHANNEL


#ifdef DEBUG
#define ALGORITHM_PARAM(name, default, max)     \
int name = default;                             \
createTrackbar(#name ":", "params", &name, max);
#else
#define ALGORITHM_PARAM(name, default, max)     \
const int name = default;
#endif

#ifdef DEBUG
#define SHOW(name, mat) \
namedWindow(name, WINDOW_NORMAL); \
resizeWindow(name, Size(450, 450)); \
imshow(name, mat);
#define SHOW_CHANNEL(name, mat, channel) { \
Mat tmp; \
extractChannel(mat, tmp, channel); \
SHOW(name, tmp) \
}
#define DBP(x) std::cout << x << std::endl;
#else
#define SHOW(name, mat) ;
#define SHOW_CHANNEL(name, mat, channel) ;
#define DBP(x) ;
#endif

#define SHOW_ALWAYS(name, mat) \
namedWindow(name, WINDOW_NORMAL); \
resizeWindow(name, Size(450, 450)); \
imshow(name, mat);

#endif // SUPERPIXEL_SEG_SUPERPIXEL_SEG_HPP