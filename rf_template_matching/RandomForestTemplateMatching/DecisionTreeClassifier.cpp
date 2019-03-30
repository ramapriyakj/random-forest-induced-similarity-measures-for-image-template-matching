/*
 * File_name : DecisionTreeClassifier.cpp
 * Purpose   : The file contains all methods related to single decision tree.
 *             Tree creation, traversal etc.
 * Author    : Ramapriya Janardhan
 */
#include "DecisionTreeClassifier.h"

int DecisionTreeClassifier::nodeNumber;
int DecisionTreeClassifier::dtleafNodes;

int* DecisionTreeClassifier::classCount(Mat *imageData,Mat *imageIndex)
{
    int *counts = new int[dtLabelClasses]{0};
    int rows = imageIndex->rows;
    for (int i = 0; i < rows; i++)
    {
        counts[imageData[1].at<uchar>(imageIndex->at<int>(i,0),0)]++;
    }
    return counts;
}

float DecisionTreeClassifier::entropy(Mat *imageData,Mat *imageIndex)
{
    int *counts = classCount(imageData,imageIndex);
    int rows = imageIndex->rows;
    float entropy = 0;
    float classPercent = 0;
    float classPercentLog = 0;
    float classValue = 0;
    for (int i = 0;i < dtLabelClasses; i++)
    {
        if(counts[i] != 0)
        {
            classPercent = static_cast<float>(counts[i]) / rows;
            classPercentLog = log2f(classPercent);
            classValue = (isnan(classPercentLog) || isinf(classPercentLog)) ? 0 : (classPercent*classPercentLog);
            entropy -= classValue;
        }
    }
    delete [] counts;
    return entropy;
}

float* DecisionTreeClassifier::classPrediction(Mat *imageData,Mat *imageIndex)
{
    float *counts = new float[dtLabelClasses]{0};
    int rows = imageIndex->rows;
    for (int i = 0; i < rows; i++)
    {
        counts[imageData[1].at<uchar>(imageIndex->at<int>(i,0),0)]++;
    }

    for (int i = 0; i < dtLabelClasses; i++)
    {
        if(counts[i] != 0)
        {
            counts[i] /= rows;
        }
    }
    return counts;
}

 DecisionSplit*  DecisionTreeClassifier::findSplitEntropy(Mat *imageData,Mat *imageIndex,int depth)
 {
     int tests = (depth == 0) ? 10 : depth*50;
     int rows = imageIndex->rows;
     int cols = imageData->cols;

     float currentScore = entropy(imageData,imageIndex);
     float bestGain = 0;

     DecisionSplit *decisionSplit = nullptr;

     for(int test = 0; test < tests; test++)
     {
         int p1 = rand() % cols;
         int p2 = rand() % cols;
         Mat *trueImageData = new Mat;
         Mat *falseImageData = new Mat;
         for (int i = 0; i < rows; i++)
         {
             if (imageData[0].at<uchar>(imageIndex->at<int>(i,0), p1) <= imageData[0].at<uchar>(imageIndex->at<int>(i,0), p2))
             {
                 trueImageData->push_back(imageIndex->at<int>(i,0));
             }
             else
             {
                 falseImageData->push_back(imageIndex->at<int>(i,0));
             }
         }
         if (trueImageData->empty() || falseImageData->empty())
         {
             trueImageData->release();
             falseImageData->release();
             delete trueImageData;
             delete falseImageData;
             continue;
         }
         float probability = float(trueImageData->rows) / rows;
         float gain = currentScore - (probability*entropy(imageData,trueImageData)) - ((1 - probability)*entropy(imageData,falseImageData));
         if (gain >= bestGain)
         {
             bestGain = gain;
             if(decisionSplit != nullptr)
             {
                 decisionSplit->dsTrueImageData->release();
                 decisionSplit->dsFalseImageData->release();
                 delete decisionSplit->dsTrueImageData;
                 delete decisionSplit->dsFalseImageData;
                 delete decisionSplit;
             }
             decisionSplit = new DecisionSplit;
             decisionSplit->p1 = p1;
             decisionSplit->p2 = p2;
             decisionSplit->dsTrueImageData = trueImageData;
             decisionSplit->dsFalseImageData = falseImageData;
         }
         else
         {
             trueImageData->release();
             falseImageData->release();
             delete trueImageData;
             delete falseImageData;
         }
     }
     return decisionSplit;
 }

 DecisionNode* DecisionTreeClassifier::buildTreeEntropy(Mat *imageData,Mat *imageIndex,int depth,int maxDepth)
 {
     if(depth != maxDepth)
     {
        DecisionSplit *decisionSplit = findSplitEntropy(imageData,imageIndex,depth);
        if(decisionSplit != nullptr)
        {
            DecisionNode *decisionNode = new DecisionNode;
            decisionNode->nodeLabel = nodeNumber++;
            decisionNode->p1 = decisionSplit->p1;
            decisionNode->p2 = decisionSplit->p2;
            decisionNode->predictions = nullptr;
            decisionNode->dnTrueBranch = buildTreeEntropy(imageData,decisionSplit->dsTrueImageData,depth+1,maxDepth);
            decisionNode->dnFalseBranch = buildTreeEntropy(imageData,decisionSplit->dsFalseImageData,depth+1,maxDepth);

            if(decisionNode->dnTrueBranch == nullptr || decisionNode->dnFalseBranch == nullptr)
            {
                decisionNode->predictions = classPrediction(imageData,imageIndex);
                dtleafNodes++;
            }

            decisionSplit->dsTrueImageData->release();
            decisionSplit->dsFalseImageData->release();
            delete decisionSplit->dsTrueImageData;
            delete decisionSplit->dsFalseImageData;
            delete decisionSplit;
            return decisionNode;
        }
        else
        {
            return nullptr;
        }
     }
     else
     {
        return nullptr;
     }
 }

 DecisionSplit*  DecisionTreeClassifier::findSplitRandom(Mat *imageData,Mat *imageIndex)
 {
     int rows = imageIndex->rows;
     int cols = imageData->cols;
     int p1 = rand() % cols;
     int p2 = rand() % cols;
     Mat *trueImageData = new Mat;
     Mat *falseImageData = new Mat;
     for (int i = 0; i < rows; i++)
     {
        if (imageData[0].at<uchar>(imageIndex->at<int>(i,0), p1) <= imageData[0].at<uchar>(imageIndex->at<int>(i,0), p2))
        {
            trueImageData->push_back(imageIndex->at<int>(i,0));
        }
        else
        {
            falseImageData->push_back(imageIndex->at<int>(i,0));
        }
     }
     if (trueImageData->empty() || falseImageData->empty())
     {
         trueImageData->release();
         falseImageData->release();
         delete trueImageData;
         delete falseImageData;
         return nullptr;
     }

     DecisionSplit *decisionSplit = new DecisionSplit;
     decisionSplit->p1 = p1;
     decisionSplit->p2 = p2;
     decisionSplit->dsTrueImageData = trueImageData;
     decisionSplit->dsFalseImageData = falseImageData;
     return decisionSplit;
 }

 DecisionNode* DecisionTreeClassifier::buildTreeRandom(Mat *imageData,Mat *imageIndex,int maxDepth)
 {
     if(maxDepth != 0)
     {
        DecisionSplit *decisionSplit = findSplitRandom(imageData,imageIndex);
        if(decisionSplit != nullptr)
        {
            DecisionNode *decisionNode = new DecisionNode;
            decisionNode->nodeLabel = nodeNumber++;
            decisionNode->p1 = decisionSplit->p1;
            decisionNode->p2 = decisionSplit->p2;
            decisionNode->predictions = nullptr;
            decisionNode->dnTrueBranch = buildTreeRandom(imageData,decisionSplit->dsTrueImageData,maxDepth-1);
            decisionNode->dnFalseBranch = buildTreeRandom(imageData,decisionSplit->dsFalseImageData,maxDepth-1);

            if(decisionNode->dnTrueBranch == nullptr || decisionNode->dnFalseBranch == nullptr)
            {
                decisionNode->predictions = classPrediction(imageData,imageIndex);
                dtleafNodes++;
            }
            decisionSplit->dsTrueImageData->release();
            decisionSplit->dsFalseImageData->release();
            delete decisionSplit->dsTrueImageData;
            delete decisionSplit->dsFalseImageData;
            delete decisionSplit;
            return decisionNode;
        }
        else
        {
            return nullptr;
        }
     }
     else
     {
        return nullptr;
     }
 }

 DecisionNode* DecisionTreeClassifier::fit(Mat *imageData,Mat *imageIndex,int dtMaxDepth,int method)
 {
     nodeNumber = 0;
     dtleafNodes = 0;
     switch(method)
     {
        case 0:
            return buildTreeEntropy(imageData,imageIndex,0,dtMaxDepth);
        case 1:
            return buildTreeRandom(imageData,imageIndex,dtMaxDepth);
     }
     return nullptr;
 }

 void DecisionTreeClassifier::saveDecisionTreeObject(DecisionNode *rootNode, ofstream& myfile)
 {
     if (!rootNode)
     {
        myfile << " # ";
     }
     else
     {
         myfile << rootNode->nodeLabel << " "
                << rootNode->p1 << " "
                << rootNode->p2 << " | ";
         saveDecisionTreeObject(rootNode->dnTrueBranch,myfile);
         saveDecisionTreeObject(rootNode->dnFalseBranch,myfile);
     }
 }

 DecisionNode* DecisionTreeClassifier::getDecisionNode(ifstream& myfile)
 {
     string data;
     int tokCount = 0;
     int nodeLabel;
     int p1;
     int p2;
     while(myfile >> data)
     {
         if(data.compare("|") == 0)
         {
             DecisionNode *decisionNode = new DecisionNode;
             decisionNode->nodeLabel = nodeLabel;
             decisionNode->p1 = p1;
             decisionNode->p2 = p2;
             decisionNode->predictions = nullptr;
             decisionNode->dnTrueBranch = getDecisionNode(myfile);
             decisionNode->dnFalseBranch = getDecisionNode(myfile);

             return decisionNode;
         }
         else if(data.compare("#") == 0)
         {
             return nullptr;
         }
         else
         {
             switch(tokCount)
             {
                case 0: nodeLabel = stoi(data);
                        break;
                case 1: p1 = stoi(data);
                        break;
                case 2: p2 = stoi(data);
                        break;
             }
             tokCount++;
         }
     }
 }

 DecisionNode* DecisionTreeClassifier::getDecisionTree(ifstream& myfile)
 {
     return getDecisionNode(myfile);
 }

 string DecisionTreeClassifier::getPath(DecisionNode *rootNode, string path, Mat *data,int row)
 {
     if (rootNode != nullptr)
     {
         path = path + " " + to_string(rootNode->nodeLabel);
         if (data->at<uchar>(row, rootNode->p1) <= data->at<uchar>(row, rootNode->p2))
         {
             if(rootNode->dnTrueBranch != nullptr)
             {
                 path = getPath(rootNode->dnTrueBranch, path, data, row);
             }
         }
         else
         {
             if(rootNode->dnFalseBranch != nullptr)
             {
                 path = getPath(rootNode->dnFalseBranch, path, data, row);
             }
         }
     }
     return path;
 }

 void DecisionTreeClassifier::getPath(DecisionNode *rootNode,vector<int> *path, Mat *data,int row)
 {
     if (rootNode != nullptr)
     {
         path->push_back(rootNode->nodeLabel);
         if (data->at<uchar>(row, rootNode->p1) <= data->at<uchar>(row, rootNode->p2))
         {
             if(rootNode->dnTrueBranch != nullptr)
             {
                 getPath(rootNode->dnTrueBranch, path, data, row);
             }
         }
         else
         {
             if(rootNode->dnFalseBranch != nullptr)
             {
                 getPath(rootNode->dnFalseBranch, path, data, row);
             }
         }
     }
 }

 int DecisionTreeClassifier::getLeafNode(DecisionNode *rootNode,Mat *data,int row)
 {
     int nodeLabel = -1;
     if (rootNode != nullptr)
     {
         nodeLabel = rootNode->nodeLabel;
         if (data->at<uchar>(row, rootNode->p1) <= data->at<uchar>(row, rootNode->p2))
         {
             if(rootNode->dnTrueBranch != nullptr)
             {
                 nodeLabel = getLeafNode(rootNode->dnTrueBranch, data,row);
             }
         }
         else
         {
             if(rootNode->dnFalseBranch != nullptr)
             {
                 nodeLabel = getLeafNode(rootNode->dnFalseBranch, data,row);
             }
         }
     }
     return nodeLabel;
 }

 float* DecisionTreeClassifier::getPredictions(DecisionNode *rootNode,Mat *data,Mat *index,int row)
 {
     float *pred = nullptr;
     if (rootNode != nullptr)
     {
         pred = rootNode->predictions;
         if (data[0].at<uchar>(index->at<int>(row,0), rootNode->p1) <= data[0].at<uchar>(index->at<int>(row,0), rootNode->p2))
         {
             if(rootNode->dnTrueBranch != nullptr)
             {
                 pred = getPredictions(rootNode->dnTrueBranch,data,index,row);
             }
         }
         else
         {
             if(rootNode->dnFalseBranch != nullptr)
             {
                 pred = getPredictions(rootNode->dnFalseBranch,data,index,row);
             }
         }
     }
     return pred;
 }

 float* DecisionTreeClassifier::getPredictions(DecisionNode *rootNode,Mat *data,int row)
 {
     float *pred = nullptr;
     if (rootNode != nullptr)
     {
         pred = rootNode->predictions;
         if (data[0].at<uchar>(row, rootNode->p1) <= data[0].at<uchar>(row, rootNode->p2))
         {
             if(rootNode->dnTrueBranch != nullptr)
             {
                 pred = getPredictions(rootNode->dnTrueBranch,data,row);
             }
         }
         else
         {
             if(rootNode->dnFalseBranch != nullptr)
             {
                 pred = getPredictions(rootNode->dnFalseBranch,data,row);
             }
         }
     }
     return pred;
 }

 void DecisionTreeClassifier::saveDecisionTree(DecisionNode *rootNode, ofstream& myfile,string spacing)
 {
     if (rootNode != nullptr)
     {
		 myfile << spacing;
		 myfile << "L:" << rootNode->nodeLabel;
         myfile << ",P1:" << rootNode->p1;
         myfile << ",P2:" << rootNode->p2 << endl;
         saveDecisionTree(rootNode->dnTrueBranch, myfile, spacing + " ");
         saveDecisionTree(rootNode->dnFalseBranch, myfile, spacing + " ");
	 }
 }

 void DecisionTreeClassifier::printDecisionTree(DecisionNode *rootNode,string spacing)
 {
     if (rootNode != nullptr)
     {
         cout << spacing;
         cout << "L:" << rootNode->nodeLabel;
         cout << ",P1:" << rootNode->p1;
         cout << ",P2:" << rootNode->p2 << endl;
         if(rootNode->predictions != nullptr)
         {
             cout << spacing << "[";
             float *counts = rootNode->predictions;
             for (int i = 0; i < dtLabelClasses; i++)
             {
                 if(counts[i] != 0)
                 {
                     printf("(%d,%f)",i,counts[i]);
                 }
             }
             cout << "]";
             cout << endl;
         }
         printDecisionTree(rootNode->dnTrueBranch, spacing + " ");
         printDecisionTree(rootNode->dnFalseBranch, spacing + " ");
     }
 }
