//
// Created by nvidia on 6/13/18.
//
#include "fps.hpp"

namespace cove {

int CLOCK() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (t.tv_sec * 1000) + (t.tv_nsec * 1e-6);
}

double avgdur(double newdur) {
    _avgdur = 0.98 * _avgdur + 0.02 * newdur;
    return _avgdur;
}

double avgfps() {
    if (CLOCK() - _fpsstart > 1000) {
        _fpsstart = CLOCK();
        _avgfps = 0.7 * _avgfps + 0.3 * _fps1sec;
        _fps1sec = 0;
    }

    _fps1sec++;
    return _avgfps;
}


double _avgdur = 0;
int _fpsstart = 0;
double _avgfps = 0;
double _fps1sec = 0;
}