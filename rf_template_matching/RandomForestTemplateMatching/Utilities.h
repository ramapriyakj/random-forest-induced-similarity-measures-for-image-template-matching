#pragma once
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/core/core_c.h"
#include "opencv2/xfeatures2d/nonfree.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <math.h>
#include <unordered_set>
#include <iomanip>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>

using namespace cv;
using namespace std;

struct DecisionSplit
{
    int p1;
    int p2;
    Mat *dsTrueImageData;
    Mat *dsFalseImageData;
};

struct DecisionNode
{
    int nodeLabel;
    int p1;
    int p2;
    float *predictions;
    DecisionNode *dnTrueBranch;
    DecisionNode *dnFalseBranch;
};

struct Match
{
    float* score;
    int* patch;
    int patchCount;
};
