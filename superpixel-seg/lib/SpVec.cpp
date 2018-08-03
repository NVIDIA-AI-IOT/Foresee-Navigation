//
// Created by nvidia on 6/13/18.
//

#include "SpVec.hpp"

//
// Created by nvidia on 6/11/18.
//


#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include "superpixel_seg.hpp"
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_matrix.hpp>


namespace cove {

using namespace cv;
using namespace std;


typedef Vec<double, SPVEC_SIZE> SpVec;

template <typename T>
void printVec(const vector<T>& v) {
    for (T a: v) {
        cout << setw(COUT_WIDTH) << a;
    }
    cout << endl;
}


SpVec multiply(const SpVec& a , const SpVec& b) {
    SpVec temp;
    for (int i = 0; i < SPVEC_SIZE; i++) {
        temp[i] = a[i] * b[i];
    }
    return temp;
}

void printVecSpVec(const vector<SpVec> &v) {
    cout << setw(COUT_WIDTH) << "idx" << setw(COUT_WIDTH) << "x" << setw(COUT_WIDTH) << "y" << setw(COUT_WIDTH) << "cL" << setw(COUT_WIDTH) << "ca" << setw(COUT_WIDTH)
         << "cb" << setw(COUT_WIDTH) << "crclr\n";
    for (int i = 0; i < v.size(); i++) {
        cout.precision(4);
        cout << setw(COUT_WIDTH) << i;
        for (int j = 0; j < SPVEC_SIZE; j++) {
            cout << setw(COUT_WIDTH) << v.at(i)[j];
        }
        cout << "\n";
    }
    cout << flush;
}

void printSpVec(const SpVec& v) {
    vector<SpVec> a;
    a.push_back(v);
    printVecSpVec(a);
}

vector<int> getTrainingLabels(InputArray _labels, int regionSize, int rows, int cols) {
    auto labels = _labels.getMat();
    Size s = labels.size();
    int radius = regionSize / 2;
    vector<int> label_vals;
    for (int ydiff = 0; ydiff < rows; ydiff++) {
        label_vals.push_back(labels.at<int>(s.height - radius - ydiff * regionSize, s.width / 2));
        for (int xdiff = 1; xdiff <= cols / 2; xdiff += 1) {
            label_vals.push_back(labels.at<int>(s.height - radius - ydiff * regionSize, s.width / 2 + xdiff * regionSize));
            label_vals.push_back(labels.at<int>(s.height - radius - ydiff * regionSize, s.width / 2 - xdiff * regionSize));
        }
    }
    return label_vals;
}

SpVec getParams(InputArray labels, InputArray _frame, int labelVal) {
    SpVec ret(0.0);
    int idx = 0;
    // mask the thing
    Mat frame = _frame.getMat();
    Mat temp;
    cvtColor(frame, temp, COLOR_BGR2HLS);
    Mat mask;
    inRange(labels, Scalar(labelVal), Scalar(labelVal), mask);

    Scalar mean, stddev;
    meanStdDev(temp, mean, stddev, mask);
    // find color
    for (int i = 0; i < 3; i++) {
        ret[idx++] = mean[i];
    }
    for (int i = 0; i < 3; i++) {
        ret[idx++] = stddev[i];
    }
    // find squareness factor
    vector<vector<Point>> contours;
    findContours(mask, contours, noArray(), RETR_LIST, CHAIN_APPROX_SIMPLE);
    if (contours.size() < 1) {
        ret[idx++] = 0;
        return ret;
    }
    double area = contourArea(contours.at(0));
//    vector<Point> hull;
//    convexHull(contours.at(0), hull);
//    double hull_area = contourArea(hull);
//    double solidity = area / hull_area;
    double p = arcLength(contours.at(0), true);
    ret[idx] = 4 * M_PI * area / (p * p);
//    ret[idx] *= solidity;
    if (isnan(ret[idx]) || isinf(ret[idx])) {
        ret[idx] = 0;
    }
    idx++;
    return ret;
}

SpVec vecAverage(const vector<SpVec>& v) {
    SpVec ret(0.0);
    for(auto vec : v) {
        ret += vec;
    }
    ret /= static_cast<int>(v.size());
    return ret;
}


SpVec vecStdDev(const vector<SpVec>& v, const SpVec& mean) {
    SpVec variance(0.0);
    for (auto vec: v) {
        SpVec t = (vec - mean);
        variance += multiply(t, t);
    }
    variance /= static_cast<int>(v.size());
    sqrt(variance, variance);
    return variance;
}


SpVec vecStdDev(const vector<SpVec>& v) {
    SpVec mean = vecAverage(v);
//    cout << "TEMPORARY MEAN" << endl;
//    printSpVec(mean);
    return vecStdDev(v, mean);
}

//            H   L   S    H  L    S    CIRC
SpVec weights(1, 0.5, 0.5, 1, 0.5, 0.5, 1.5);

double normSsd(const SpVec& v, const SpVec& mean, const SpVec& stdDev) {
    SpVec temp;
    double sum = 0;
    for (int i = 0; i < SPVEC_SIZE; i++) {
        temp[i] = (v[i] - mean[i]) / stdDev[i];
        sum += temp[i] * temp[i] * weights[i];
    }
    return sum;
}


Matx<double, 2, 3> translateImg(Mat &img, int offsetx, int offsety){
    Matx<double, 2, 3> trans_mat(1.0, 0.0, static_cast<double>(offsetx), 0.0, 1.0, static_cast<double>(offsety));
    warpAffine(img, img, trans_mat, img.size());
    return trans_mat;
}

Graph generateGraph(InputArray _labels, unsigned int max) {
    // dilate the contonours
    Mat labels;
    Graph g(max);

    _labels.getMat().convertTo(labels, CV_8UC1);

    CV_Assert(labels.channels() == 1);

    Mat tmp;
    Mat d;

    vector<pair<int,int>> offsets;
    offsets.push_back(make_pair(1, 0));
    offsets.push_back(make_pair(0, 1));
    for (auto pair: offsets) {
        labels.copyTo(tmp);
        translateImg(tmp, pair.first, pair.second);
        SHOW("translated", tmp);
        d = labels - tmp;

        int nRows = tmp.rows;
        int nCols = tmp.cols;

        if (tmp.isContinuous()) {
            CV_Assert(d.isContinuous());
            CV_Assert(labels.isContinuous());
            nCols *= nRows;
            nRows = 1;
        }

        int i, j;
        uchar *l, *dd, *pp;
        for (i = 0; i < nRows; ++i) {
            pp = tmp.ptr<uchar>(i);
            dd = d.ptr<uchar>(i);
            l = labels.ptr<uchar>(i);
            for (j = 0; j < nCols; ++j) {
                if (dd[j] == 0) continue;
                boost::add_edge(l[j], pp[j], g);
            }
        }
    }

    return g;
}

}