/*
 * File_name : RandomForestTemplateMatching.cpp
 * Purpose   : The file contains all methods related to random forest.
 *             Random forest creation, training, testing, traversal etc.
 * Author    : Ramapriya Janardhan
 */
#include "RandomForest.h"

RandomForest::RandomForest()
{
    this->rfNumberOfTrees = 0;
    this->rfImageData = nullptr;
    this->rfMaxDepth = 0;
    this->rfLabelClasses = 0;
}

void RandomForest::initialize(int rfNumberOfTrees,Mat *rfImageData,int rfMaxDepth, int rfLabelClasses)
{
	this->rfNumberOfTrees = rfNumberOfTrees;
	this->rfImageData = rfImageData;
	this->rfMaxDepth = rfMaxDepth;
    this->rfLabelClasses = rfLabelClasses;
}

void RandomForest::freeMemory()
{
    if(trees.size() != 0)
    {
        for (int i = 0; i < rfNumberOfTrees; i++)
        {
            freeTree(trees[i]);
        }
    }
}

void RandomForest::freeTree(DecisionNode *rootNode)
{
    if (rootNode != nullptr)
    {
        if(rootNode->predictions != nullptr)
        {
            delete [] rootNode->predictions;
        }
        freeTree(rootNode->dnTrueBranch);
        freeTree(rootNode->dnFalseBranch);
        delete rootNode;
    }
}

Mat* RandomForest::getSamples(int rows,int maxSamples)
{
    Mat *randomImageData = new Mat;
    if (rows < maxSamples)
    {
        maxSamples = rows;
    }
    for (int i = 0; i < maxSamples; i++)
    {
        int randomNumber = rand() % rows;
        randomImageData->push_back(randomNumber);
    }
    return randomImageData;
}

void RandomForest::fit(int method)
{
    DecisionTreeClassifier decisionTreeClassifier(rfLabelClasses);
    int maxSamples = (0.63)*rfImageData[0].rows;
	for (int i = 0; i < rfNumberOfTrees; i++)
	{
        Mat *imageIndex = getSamples(rfImageData->rows,maxSamples);
        trees.push_back(decisionTreeClassifier.fit(rfImageData, imageIndex, rfMaxDepth, method));
        imageIndex->release();
        delete imageIndex;
        cout << i+1 << " tree generated." << endl;
    }
    rfImageData[0].release();
    rfImageData[1].release();
    delete [] rfImageData;
}

void RandomForest::saveDecisionTreesObject(string filePath)
{
    DecisionTreeClassifier decisionTreeClassifier(rfLabelClasses);
    ofstream myfile;
    myfile.open(filePath);
    if(myfile)
    {
        myfile << rfNumberOfTrees << endl;
        myfile << rfMaxDepth << endl;
        myfile << rfLabelClasses << endl;
        for (int i = 0; i < rfNumberOfTrees; i++)
        {
            decisionTreeClassifier.saveDecisionTreeObject(trees[i], myfile);
            myfile << endl;
        }
        myfile.close();
    }
}

void RandomForest::loadRandomForest(string filePath,int treeCount)
{
    ifstream myfile;
    myfile.open(filePath);
    if(myfile)
    {
        string word;
        myfile >> word;
        this->rfNumberOfTrees = stoi(word);
        if(treeCount != -1 && treeCount > 0)
        {
            this->rfNumberOfTrees = treeCount;
        }
        myfile >> word;
        this->rfMaxDepth = stoi(word);
        myfile >> word;
        this->rfLabelClasses = stoi(word);
        DecisionTreeClassifier decisionTreeClassifier(rfLabelClasses);
        for (int i = 0; i < rfNumberOfTrees; i++)
        {
            trees.push_back(decisionTreeClassifier.getDecisionTree(myfile));
        }
        myfile.close();
    }
}

void RandomForest::saveDecisionTrees(string filePath)
{
    DecisionTreeClassifier decisionTreeClassifier(rfLabelClasses);
    ofstream myfile;
    myfile.open(filePath);
    for (int i = 0; i < rfNumberOfTrees; i++)
    {
        myfile << "---------------DecisionTree[" << i << "]---------------" << endl;
        decisionTreeClassifier.saveDecisionTree(trees[i], myfile, "");
    }
    myfile.close();
}

void RandomForest::printDecisionTrees()
{
    DecisionTreeClassifier decisionTreeClassifier(rfLabelClasses);
    for (int i = 0; i < rfNumberOfTrees; i++)
    {
        cout << "---------------DecisionTree[" << i << "]---------------" << endl;
        decisionTreeClassifier.printDecisionTree(trees[i], "");
    }
}

void RandomForest::printAndSaveStatisticsOOB(string statistics,int method)
{
    static clock_t start;
    static clock_t stop;
    static double elapsed_secs;
    string myfilePath = statistics + (method == 0? "entropy_" : "random_") + "_statisticsOOB.txt";
    ofstream myfile;
    myfile.open(myfilePath);
    int size = rfImageData[0].rows;
    int subTrainSize = (0.63)*size;
    cout << "----- Random Forest Statistics -----" << endl;
    cout << "Total samples : " << size << endl;
    cout << "Bagging samples : " << subTrainSize << endl;
    cout << "Trees,Depth,LeafNodes,Training Time,Testing Accuracy" << endl;

    myfile << "----- Random Forest Statistics -----" << endl;
    myfile << "Total samples : " << size << endl;
    myfile << "Bagging samples : " << subTrainSize << endl;
    myfile << "Trees,Depth,LeafNodes,Training time,Testing accuracy" << endl;
    int depthData[6] = {10,12,14,16,18,20};
    int depthSize = 0;
    if(method == RF_FIT_ENTROPY)
    {
        depthSize = 3;
    }
    else
    {
        depthSize = 6;
    }
    for(int depth = 0; depth < depthSize ; depth++)
    {
        int totalTrees = 0;
        if(method == 0)
        {
            totalTrees = 50;
        }
        else
        {
            totalTrees = 100;
        }
        elapsed_secs = 0;
        DecisionTreeClassifier decisionTreeClassifier(rfLabelClasses);
        vector<DecisionNode*> trees;
        Mat oob = Mat::ones(size,totalTrees,CV_8SC1);
        for(int tree = 0; tree < totalTrees ; tree++)
        {
            Mat subTrainData;
            for(int a = 0; a< subTrainSize; a++)
            {
                int random = rand() % size;
                subTrainData.push_back(random);
                oob.at<uchar>(random,tree) = 0;
            }
            start = clock();
            trees.push_back(decisionTreeClassifier.fit(rfImageData,&subTrainData, depthData[depth],method));
            int leafNodes = decisionTreeClassifier.dtleafNodes;
            stop = clock();
            elapsed_secs += double(stop - start) / CLOCKS_PER_SEC;
            subTrainData.release();
            int errors = 0;
            int totalTestSamples = 0;
            for(int b = 0; b < size; b++)
            {
                int treeCount = 0;
                float max = 0;
                int classLabel = -1;
                float *p_o = new float[rfLabelClasses] {0};
                for (int g = 0; g <= tree; g++)
                {
                    if(oob.at<uchar>(b,tree) == 1)
                    {
                        treeCount++;
                        float *p_i = decisionTreeClassifier.getPredictions(trees[g],rfImageData,b);
                        if(p_i != nullptr)
                        {
                            for (int i_i = 0; i_i < rfLabelClasses; i_i++)
                            {
                                if(p_i[i_i] != 0)
                                {
                                    p_o[i_i] += p_i[i_i];
                                }
                            }
                        }
                    }
                }
                if(treeCount != 0)
                {
                    totalTestSamples++;
                    for(int i_o = 0; i_o < rfLabelClasses; i_o++)
                    {
                        if(p_o[i_o] != 0)
                        {
                            p_o[i_o] /= treeCount;
                        }
                    }
                    for(int i_o = 0; i_o < rfLabelClasses; i_o++)
                    {
                        if(p_o[i_o] != 0 && p_o[i_o] > max)
                        {
                            max = p_o[i_o];
                            classLabel = i_o;
                        }
                    }
                    if(rfImageData[1].at<uchar>(b,0) != classLabel)
                    {
                        errors++;
                    }
                }
                delete [] p_o;
            }
            float accuracy = (static_cast<float>(totalTestSamples-errors)/static_cast<float>(totalTestSamples))*100.0;
            cout << tree+1 << "," << depthData[depth] << "," << leafNodes << "," << elapsed_secs << "," << accuracy << endl;
            myfile << tree+1 << "," << depthData[depth] << "," << leafNodes << "," << elapsed_secs << "," << accuracy << endl;
        }
        oob.release();
        for (int h = 0; h < totalTrees; h++)
        {
            freeTree(trees[h]);
        }
    }
    rfImageData[0].release();
    rfImageData[1].release();
    delete [] rfImageData;
    myfile.close();
    cout << "File saved under " << myfilePath << endl;
}

string* RandomForest::getDecisionPath(Mat *patchData)
{
    DecisionTreeClassifier decisionTreeClassifier(rfLabelClasses);
    int size = patchData->rows;
    string *path = new string[size * rfNumberOfTrees];
    int count = 0;
    for (int i = 0; i < rfNumberOfTrees; i++)
    {
        for(int j = 0; j < size; j++)
        {
            path[count++] = decisionTreeClassifier.getPath(trees[i],"",patchData,j);
        }
    }
    return path;
}

int* RandomForest::getLeafNode(Mat *patchData)
{
    DecisionTreeClassifier decisionTreeClassifier(rfLabelClasses);
    int size = patchData->rows;
    int *path = new int[size * rfNumberOfTrees];
    int count = 0;
    for (int i = 0; i < rfNumberOfTrees; i++)
    {
        for(int j = 0; j < size; j++)
        {
            path[count++] = decisionTreeClassifier.getLeafNode(trees[i],patchData,j);
        }
    }
    return path;
}

vector<vector<int>>* RandomForest::getDecisionPathVector(Mat *patchData)
{
    DecisionTreeClassifier decisionTreeClassifier(rfLabelClasses);
    int size = patchData->rows;
    vector<vector<int>> *path = new vector<vector<int>>();
    for (int i = 0; i < rfNumberOfTrees; i++)
    {
        for(int j = 0; j < size; j++)
        {
            vector<int> data;
            decisionTreeClassifier.getPath(trees[i],&data,patchData,j);
            path->push_back(data);
        }
    }
    return path;
}

void RandomForest::saveTemplateDecisionPathVector(Mat *templateData)
{
    templateVector = getDecisionPathVector(templateData);
    templateVectorSize = templateVector->size();
}

void RandomForest::freeTemplateDecisionPathVector()
{
    if(templateVector != nullptr)
    {
        delete templateVector;
    }
}

Match* RandomForest::findSimilarityByPath(Mat *sourceData, Mat *templateData)
{
    DecisionTreeClassifier decisionTreeClassifier(rfLabelClasses);

    int sSize = sourceData->rows;
    int tSize = templateData->rows;

    Match *match = new Match;
    match->patch = new int[tSize];
    match->score = new float[tSize];

    float **score = new float*[tSize];
    for (int i = 0; i < tSize; i++)
    {
        score[i] = new float[sSize]{0};
    }

    vector<vector<int>> *ss = getDecisionPathVector(sourceData);
    vector<vector<int>> *ts = getDecisionPathVector(templateData);

    int tcount = 0;
    int scount = 0;
    for (int k = 0; k < rfNumberOfTrees; k++)
    {
        for (int i = 0; i < tSize; i++)
        {
            int te_size = ts->at(tcount).size();
            for (int j = 0; j < sSize; j++)
            {
                int sindex = j+scount;
                int so_size = ss->at(sindex).size();
                int countMin;
                int countMax;
                int countMatch = 0;
                if(so_size > te_size)
                {
                    countMax = so_size;
                    countMin = te_size;
                }
                else
                {
                    countMin = so_size;
                    countMax = te_size;
                }
                for (int s = 0; s < countMin; s++)
                {
                    if(ss->at(sindex).at(s) == ts->at(tcount).at(s))
                    {
                        countMatch++;
                    }
                    else
                    {
                        break;
                    }
                }
                score[i][j] += countMatch/countMax;
            }
            tcount = tcount + 1;
        }
        scount = scount + sSize;
    }

    for (int i = 0; i < tSize; i++)
    {
        for (int j = 0; j < sSize; j++)
        {
            score[i][j] = score[i][j] / rfNumberOfTrees;
        }
    }

    for (int i = 0; i < tSize; i++)
    {
        float bestScore = score[i][0];
        int bestPatch = 0;
        for (int j = 1; j < sSize; j++)
        {
            if (score[i][j] > bestScore)
            {
                bestScore = score[i][j];
                bestPatch = j;
            }
        }
        match->patch[i] = bestPatch;
        match->score[i] = bestScore;
    }
    match->patchCount = tSize;

    delete ss;
    delete ts;
    for (int i = 0; i < tSize; i++)
    {
        delete [] score[i];
    }
    delete [] score;
    return match;
}

Match* RandomForest::findSimilarityByLeafNode(Mat *sourceData, Mat *templateData)
{
    DecisionTreeClassifier decisionTreeClassifier(rfLabelClasses);

    int sSize = sourceData->rows;
    int tSize = templateData->rows;

    Match *match = new Match;
    match->patch = new int[tSize];
    match->score = new float[tSize];

    float **score = new float*[tSize];
    for (int i = 0; i < tSize; i++)
    {
        score[i] = new float[sSize]{0};
    }

    int *ss = getLeafNode(sourceData);
    int *ts = getLeafNode(templateData);

    int tcount = 0;
    int scount = 0;
    for (int k = 0; k < rfNumberOfTrees; k++)
    {
        for (int i = 0; i < tSize; i++)
        {
            for (int j = 0; j < sSize; j++)
            {
                int sindex = j+scount;
                if(ts[tcount] == ss[sindex])
                {
                    score[i][j] +=1;
                }
            }
            tcount = tcount + 1;
        }
        scount = scount + sSize;
    }

    for (int i = 0; i < tSize; i++)
    {
        for (int j = 0; j < sSize; j++)
        {
            score[i][j] = score[i][j] / rfNumberOfTrees;
        }
    }

    for (int i = 0; i < tSize; i++)
    {
        float bestScore = score[i][0];
        int bestPatch = 0;
        for (int j = 1; j < sSize; j++)
        {
            if (score[i][j] > bestScore)
            {
                bestScore = score[i][j];
                bestPatch = j;
            }
        }
        match->patch[i] = bestPatch;
        match->score[i] = bestScore;
    }
    match->patchCount = tSize;

    delete [] ss;
    delete [] ts;
    for (int i = 0; i < tSize; i++)
    {
        delete[] score[i];
    }
    delete [] score;
    return match;
}

int RandomForest::getSimilarityScoreByPath(Mat *sourceData)
{
    int result = 0;
    vector<vector<int>> *sourceVector = getDecisionPathVector(sourceData);
    for (int i = 0; i < templateVectorSize; i++)
    {
        int te_size = templateVector->at(i).size();
        int so_size = sourceVector->at(i).size();
        int countMin;
        int countMatch = 0;
        countMin = (so_size > te_size) ? te_size : so_size;
        for (int j = 0; j < countMin; j++)
        {
            if(sourceVector->at(i).at(j) == templateVector->at(i).at(j))
            {
                countMatch++;
            }
            else
            {
                break;
            }
        }
        result += countMatch;
    }
    delete sourceVector;
    return result;
}
