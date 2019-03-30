#pragma once
#include "DecisionTreeClassifier.h"

using namespace cv;
using namespace std;

class RandomForest
{
    public:
        static const int PRINT_NO = 0;
        static const int PRINT_YES = 1;
        static const int RF_DEPTH_ALL = -1;
        static const int RF_FIT_ENTROPY = 0;
        static const int RF_FIT_RANDOM = 1;
        static const int RF_SIMILARITY_BY_LEAF_NODE = 0;
        static const int RF_SIMILARITY_BY_PATH = 1;
        static const int RF_TREE_ALL = -1;
        Mat *rfImageData;
        int rfLabelClasses;
        int rfMaxDepth;
        int rfNumberOfTrees;
        int templateVectorSize;
        vector<DecisionNode*> trees;
        vector<vector<int>> *templateVector;
        RandomForest();
        ~RandomForest()
        {
            freeMemory();
        }
        Mat* getSamples(int rows,int maxSamples);
        Match* findSimilarityByLeafNode(Mat *sourceData, Mat *templateData);
        Match* findSimilarityByPath(Mat *sourceData, Mat *templateData);
        int getSimilarityScoreByPath(Mat *sourceData);
        int* getLeafNode(Mat *patchData);
        string* getDecisionPath(Mat *patchData);
        vector<vector<int>>* getDecisionPathVector(Mat *patchData);
        void fit(int method=0);
        void freeMemory();
        void freeTree(DecisionNode *rootNode);
        void initialize(int rfNumberOfTrees, Mat *rfImageData, int rfMaxDepth, int rfLabelCLasses);
        void loadRandomForest(string filePath,int treeCount);
        void printAndSaveStatisticsOOB(string statistics,int method);
        void printDecisionTrees();
        void saveDecisionTrees(string filePath);
        void saveDecisionTreesObject(string filePath);
        void saveTemplateDecisionPathVector(Mat *templateData);
        void freeTemplateDecisionPathVector();
};
