#pragma once
#include "Utilities.h"
#include "RandomForest.h"

using namespace cv;
using namespace std;

class TemplateMatching
{
    public:
        static const int FAST_KEYS = 2;
        static const int KEYPOINTS_NO = 0;
        static const int KEYPOINTS_YES = 1;
        static const int SIFT_KEYS = 1;
        static const int SURF_KEYS = 0;
        RandomForest randomForest;
        static int patchSize;
        static string randomForestPath;
        static string statisticsOOBResultPath;
        static string trainResultPath;
        TemplateMatching();
        Mat* getBinData(int *classes,string trainFolder);
        Point getBestCenterKey(vector<Point2f> *spoints,vector<Point2f> *tpoints,int trows,int tcols);
        Point getMatch(Mat *sourceImg,Mat *templateImg);
        Point getMatchKey(Mat *sourceImg,Mat *templateImg);
        Point getMatchKey(Mat *sourceImg,Mat *templateImg,int keys,int similarity,int showKeys);
        void getOOBStatistics(string trainFolder,int method);
        void load(int trees,int print);
        void matchTemplate(string sourcePath,string templatePath);
        void matchTemplateKeyPoint(string sourcePath,string templatePath);
        void matchTemplateKeyPoint(string sourcePath,string templatePath,int keys,int similarity,int showKeys);
        void matchVideoKeyPoint(string sourcePath,string templatePath,int keys,int similarity,int showKeys);
        void train(string trainFolder,int trees,int depth,int method);
        vector<Point2f> getBestCenterKeyRot(vector<Point2f> *spoints,vector<Point2f> *tpoints,int trows,int tcols);
        vector<Point2f> getMatchKeyRot(Mat *sourceImg,Mat *templateImg,int keys,int similarity,int showKeys);
        vector<Point2f> getMatchKeyRot(Mat *sourceImg,Mat *templateImg);
};
