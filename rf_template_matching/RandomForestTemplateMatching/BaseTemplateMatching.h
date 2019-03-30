#pragma once
#include <unordered_set>
#include <iostream>
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

using namespace cv;
using namespace std;

struct Filter
{
    double filterValue;
    int filterNumber;
    Rect rectA,rectB,rectC,rectD;
};

class BaseTemplateMatching
{
    public:
        int features = 0;
        Point getMatch(Mat *sourceImg,Mat *templateImg,int patchSize);
        double getFeatureDifference(Mat *img,Filter *feature);
        vector<Filter>* getPatches(Mat *img, int patchSize);
        void displayTemplate(string sourcePath,string templatePath,int patchSize);
        void populatePatchFeatures(Mat *img,int patchSize,vector<Filter> *arr,int tmpx,int tmpy);
};
