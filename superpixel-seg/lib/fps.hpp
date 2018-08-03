//
// Created by nvidia on 6/8/18.
//
#ifndef SUPERPIXEL_SEG_FPS_HPP
#define SUPERPIXEL_SEG_FPS_HPP

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv/cv.h>

#include <sys/timeb.h>

namespace cove {

#if defined(_MSC_VER) || defined(WIN32) || defined(_WIN32) || defined(__WIN32__) \
|| defined(WIN64) || defined(_WIN64) || defined(__WIN64__)
int CLOCK()
{
    return clock();
}
#endif

#if defined(unix) || defined(__unix) || defined(__unix__) \
|| defined(linux) || defined(__linux) || defined(__linux__) \
|| defined(sun) || defined(__sun) \
|| defined(BSD) || defined(__OpenBSD__) || defined(__NetBSD__) \
|| defined(__FreeBSD__) || defined __DragonFly__ \
|| defined(sgi) || defined(__sgi) \
|| defined(__MACOSX__) || defined(__APPLE__) \
|| defined(__CYGWIN__)

int CLOCK();

#endif

extern double _avgdur;
extern int _fpsstart;
extern double _avgfps;

extern double _fps1sec;

double avgdur(double newdur);

double avgfps();

}
#endif // SUPERPIXEL_SEG_FPS_HPP
