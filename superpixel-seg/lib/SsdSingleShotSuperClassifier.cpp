//
// Created by nvidia on 6/13/18.
//

#include <deque>
#include <boost/graph/graph_utility.hpp>
#include <boost/graph/adjacency_matrix.hpp>
#include <boost/graph/edge_list.hpp>
#include "SsdSingleShotSuperClassifier.hpp"
#include "superpixel_seg.hpp"
#include "SpVec.hpp"

using namespace std;
using namespace cv;
using namespace cv::ximgproc;
namespace cove {

SsdSingleShotSuperClassifier::SsdSingleShotSuperClassifier(Ptr<SuperpixelLSC> &&x, InputArray _cielab,
                                                           int _regionSize, double _ssdEpsilon) :
        g(x->getNumberOfSuperpixels()),
        super(x),
        cielab(_cielab.getMat()) {
    super->getLabels(labels);
    labels.convertTo(labels, CV_8UC1);
    regionSize = _regionSize;
    ssdEpsilon = _ssdEpsilon;
}

void SsdSingleShotSuperClassifier::getFloorMask(OutputArray _dst) {
    buildGraph();

#ifdef DEBUG
    boost::print_graph(g,
                       "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*()-=[];',._+{}:\"<>/?");
    cout << flush;
#endif
//    Mat mapped_labels;
//    labels.convertTo(mapped_labels, CV_8UC3);
//    SHOW_ALWAYS("labels", mapped_labels);
//    SHOW_ALWAYS("cielab", cielab);

    fillTrainingData();
    generateFeatureVectors();

//    printVecSpVec(pixelVecs);
    prepareSsdClassifier();

    auto &&floorLabels = doBfs();

    // compute the floor mask
    _dst.createSameSize(labels, CV_8U);
    Mat dst = _dst.getMat();
    dst = Mat::zeros(labels.size(), CV_8U);
    for (int label : floorLabels) {
        Mat temp;
        inRange(labels, Scalar(label), Scalar(label), temp);
        bitwise_or(dst, temp, dst);
    }
    SHOW("floor", dst);
}


void SsdSingleShotSuperClassifier::buildGraph() {
    CV_Assert(labels.channels() == 1);

    Mat tmp;
    Mat d;

    vector<pair<int, int>> offsets;
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
}

void SsdSingleShotSuperClassifier::fillTrainingData(int rows, int cols) {
    trainingLabels.clear();
    Size s = labels.size();
    int radius = regionSize / 2;
    for (int ydiff = 0; ydiff < rows; ydiff++) {
        trainingLabels.push_back(labels.at<unsigned char>(s.height - radius - ydiff * regionSize, s.width / 2));
        for (int xdiff = 1; xdiff <= cols / 2; xdiff += 1) {
            trainingLabels.push_back(labels.at<unsigned char>(s.height - radius - ydiff * regionSize,
                                                              s.width / 2 + xdiff * regionSize));
            trainingLabels.push_back(labels.at<unsigned char>(s.height - radius - ydiff * regionSize,
                                                              s.width / 2 - xdiff * regionSize));
        }
    }
}

void SsdSingleShotSuperClassifier::generateFeatureVectors() {
    int numPx = super->getNumberOfSuperpixels();
    pixelVecs = vector<SpVec>(numPx);
    for (int i = 0; i < numPx; i++) {
        pixelVecs.at(i) = move(getParams(labels, cielab, i));
    }
}

void SsdSingleShotSuperClassifier::prepareSsdClassifier() {
    vector<SpVec> trainingVecs;
    for (int idx: trainingLabels) {
        trainingVecs.push_back(pixelVecs.at(idx));
    }
//    printVecSpVec(trainingVecs);
    trainingMean = vecAverage(trainingVecs);
    populationStdDev = vecStdDev(pixelVecs);
      cout << "MEAN\nDEV" << endl;
//    printSpVec(mean);
//    printSpVec(stdDev);
    vector<double> ssdTrainVals(trainingVecs.size(), 0.0);
    for (int idx: trainingLabels) {
        ssdTrainVals.push_back(normSsd(pixelVecs.at(idx), trainingMean, populationStdDev));
    }
    maxTrainingSsd = *max_element(ssdTrainVals.begin(), ssdTrainVals.end());
}

inline bool SsdSingleShotSuperClassifier::satisfiesSsd(const SpVec &v) {
    return normSsd(v, trainingMean, populationStdDev) <= maxTrainingSsd + ssdEpsilon;
}

vector<int> SsdSingleShotSuperClassifier::doBfs() {
    vector<int> floorLabels;
    if (trainingLabels.empty()) {
        return floorLabels;
    }
    vector<bool> visited(boost::num_vertices(g), false);
    deque<int> toVisit;
    toVisit.push_back(trainingLabels.at(0));

    while (!toVisit.empty()) {
        int node = toVisit.front();
        toVisit.pop_front();

        visited.at(node) = true;
        if (!satisfiesSsd(pixelVecs.at(node))) continue;

        floorLabels.push_back(node);

        auto its = boost::out_edges(node, g);
        for (; its.first != its.second; ++(its.first)) {
            int connected = boost::target(*(its.first), g);
            if (visited.at(connected)) continue;
            toVisit.push_back(connected);
        }
    }

    return floorLabels;
}
}