#pragma once
#include "Utilities.h"

using namespace cv;
using namespace std;

class DecisionTreeClassifier
{       
    public:
        int dtLabelClasses;
        static int nodeNumber;
        static int dtleafNodes;
        DecisionTreeClassifier(int dtLabelClasses)
        {
            this->dtLabelClasses = dtLabelClasses;
        }
        DecisionNode* buildTreeEntropy(Mat *imageData,Mat *imageIndex,int depth,int dtMaxDepth);
        DecisionNode* buildTreeRandom(Mat *imageData,Mat *imageIndex,int maxDepth);
        DecisionNode* fit(Mat *imageData,Mat *imageIndex,int dtMaxDepth,int method);
        DecisionNode* getDecisionNode(ifstream& myfile);
        DecisionNode* getDecisionTree(ifstream& myfile);
        DecisionSplit* findSplitEntropy(Mat *imageData,Mat *imageIndex,int depth);
        DecisionSplit* findSplitRandom(Mat *imageData,Mat *imageIndex);
        float entropy(Mat *imageData,Mat *imageIndex);
        float* classPrediction(Mat *imageData,Mat *imageIndex);
        float* getPredictions(DecisionNode *rootNode,Mat *data,Mat *index,int row);
        float* getPredictions(DecisionNode *rootNode,Mat *data,int row);
        int getLeafNode(DecisionNode *rootNode, Mat *data,int row);
        int* classCount(Mat *imageData,Mat *imageIndex);
        string getPath(DecisionNode *rootNode, string path, Mat *data,int row);
        void getPath(DecisionNode *rootNode,vector<int> *path, Mat *data,int row);
        void printDecisionTree(DecisionNode *rootNode, string spacing);
        void saveDecisionTree(DecisionNode *rootNode, ofstream& myfile,string spacing);
        void saveDecisionTreeObject(DecisionNode *rootNode, ofstream& myfile);
};

