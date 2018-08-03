//
// Created by nvidia on 6/11/18.
//

#ifndef SUPERPIXEL_UTILS_HPP
#define SUPERPIXEL_UTILS_HPP


#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <boost/functional/hash.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_matrix.hpp>

namespace cove {

constexpr int SPVEC_SIZE = 7;
constexpr int COUT_WIDTH = 7;
typedef cv::Vec<double, SPVEC_SIZE> SpVec;

template<typename T>
void printVec(const std::vector<T> &v);


SpVec multiply(const SpVec &a, const SpVec &b);

void printVecSpVec(const std::vector<SpVec> &v);

void printSpVec(const SpVec &v);

std::vector<int> getTrainingLabels(cv::InputArray _labels, int regionSize, int rows = 1, int cols = 3);

SpVec getParams(cv::InputArray labels, cv::InputArray _frame, int labelVal);

SpVec vecAverage(const std::vector<SpVec> &v);

SpVec vecStdDev(const std::vector<SpVec> &v, const SpVec &mean);

SpVec vecStdDev(const std::vector<SpVec> &v);

extern SpVec weights;

double normSsd(const SpVec &v, const SpVec &mean, const SpVec &stdDev);

cv::Matx<double, 2, 3> translateImg(cv::Mat &img, int offsetx, int offsety);

typedef boost::adjacency_matrix<boost::undirectedS> Graph;

Graph generateGraph(cv::InputArray _labels, unsigned int max);

}
#endif // SUPERPIXEL_UTILS_HPP