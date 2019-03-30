/*
 * File_name : BaseTemplateMatching.cpp
 * Purpose   : The file contains methods to evaluate the work proposed.
 *
 * Author    : Ramapriya Janardhan
 */
#pragma once
#include "TemplateMatching.h"
#include "BaseTemplateMatching.h"
#include "LegacyTemplateMatching.h"
#include "RandomForest.h"
#include <random>
#include "math.h"

static string trainingFolderPath = "../helpers/dataset/train/";
static string testFolderPath = "../helpers/dataset/test/";
static string evaluateFolderPath = "../helpers/dataset/evaluate/";
static string resultFolder = "../helpers/rf/statistics/performance/";

void statisticsRF(string trainingFolderPath,int fitMethod)
{
    TemplateMatching templateMatching;
    templateMatching.getOOBStatistics(trainingFolderPath,fitMethod);
}

void trainRF(string trainingFolderPath,int trees,int depth,int fitMethod)
{
    TemplateMatching templateMatching;
    templateMatching.train(trainingFolderPath,trees,depth,fitMethod);
}

TemplateMatching* loadRF(int trees,int print)
{
    TemplateMatching *templateMatching = new TemplateMatching();
    templateMatching->load(trees,print);
    return templateMatching;
}

void test(int method,string sourceLocation,string templateLocation,TemplateMatching *templateMatching)
{
    LegacyTemplateMatching legacyTemplateMatching;
    BaseTemplateMatching baseTemplateMatching;
    switch(method)
    {
        case 0:
            legacyTemplateMatching.displayTemplate(sourceLocation,templateLocation);
            break;
        case 1:
            baseTemplateMatching.displayTemplate(sourceLocation,templateLocation,24);
            break;
        case 2:
            templateMatching->matchTemplate(sourceLocation,templateLocation);
            break;
        case 3:
            templateMatching->matchTemplateKeyPoint(sourceLocation,templateLocation);
            break;
        case 4:
            templateMatching->matchTemplateKeyPoint(sourceLocation,templateLocation,TemplateMatching::SURF_KEYS,RandomForest::RF_SIMILARITY_BY_PATH,TemplateMatching::KEYPOINTS_NO);
            break;
    }
}

void demonstrate(int method,TemplateMatching *templateMatching)
{
    string sources[5] = {
        "FruitSourceA.jpg",
        "FruitSourceB.jpg",
        "RoadSignSource.jpg",
        "FaceSource.jpg",
        "BirdSource.jpg"
    };

    string templates[13] = {
        "FruitTemplateB.jpg",
        "FruitTemplateA.jpg",
        "FruitTemplateC.jpg",
        "FruitTemplateD.jpg",
        "FruitTemplateE.jpg",
        "RoadSignTemplate.jpg",
        "RoadSignTemplate_occlusion.jpg",
        "RoadSignTemplate_resize.jpg",
        "FaceTemplateA.jpg",
        "FaceTemplateB.jpg",
        "FaceTemplateC.jpg",
        "BirdTemplateA.jpg",
        "BirdTemplateB.jpg",
    };

    string sourceVideoLocation = "boat.mp4";
    string templateVideoLocation = "boat.jpg";

    LegacyTemplateMatching legacyTemplateMatching;
    BaseTemplateMatching baseTemplateMatching;

    if(method == 5)
    {
        templateMatching->matchVideoKeyPoint(testFolderPath+sourceVideoLocation,testFolderPath+templateVideoLocation,TemplateMatching::SURF_KEYS,RandomForest::RF_SIMILARITY_BY_PATH,TemplateMatching::KEYPOINTS_NO);
    }
    else
    {
        for(int i = 0; i< 13; i++)
        {
            string sourceLocation;
            string templateLocation = templates[i];
            if(i < 4)
            {
                sourceLocation = sources[0];
            }
            else if(i == 4)
            {
                sourceLocation = sources[1];
            }
            else if(i < 8)
            {
                sourceLocation = sources[2];
            }
            else if(i < 11)
            {
                sourceLocation = sources[3];
            }
            else if(i < 13)
            {
                sourceLocation = sources[4];
            }

            sourceLocation = testFolderPath + sourceLocation;
            templateLocation = testFolderPath + templateLocation;

            switch(method)
            {
                case 0:
                    legacyTemplateMatching.displayTemplate(sourceLocation,templateLocation);
                    break;
                case 1:
                    baseTemplateMatching.displayTemplate(sourceLocation,templateLocation,24);
                    break;
                case 2:
                    templateMatching->matchTemplate(sourceLocation,templateLocation);
                    break;
                case 3:
                    templateMatching->matchTemplateKeyPoint(sourceLocation,templateLocation);
                    break;
                case 4:
                    templateMatching->matchTemplateKeyPoint(sourceLocation,templateLocation,TemplateMatching::SURF_KEYS,RandomForest::RF_SIMILARITY_BY_PATH,TemplateMatching::KEYPOINTS_NO);
                    break;
            }
        }
    }
}

void addNoise(Mat *source, Mat *destination,double mean, double stdDev)
{
    Mat gaussian = Mat((*source).size(),CV_8UC3);
    randn(gaussian,Scalar::all(mean), Scalar::all(stdDev));
    addWeighted(*source, 1.0, gaussian, 1.0, 0.0, *destination);
    (*destination).convertTo(*destination,CV_8UC3);
}

float getIOU(Point2f A1,Point2f A2,Point B1,Point B2)
{
    Rect A = Rect(A1,A2);
    Rect B = Rect(B1,B2);
    Rect INS = A & B;
    float AREA_A = A.height*A.width;
    float AREA_B = B.height*B.width;
    float INS_AREA = INS.height*INS.width;
    float IOU = INS_AREA / float(AREA_A + AREA_B - INS_AREA);
    return IOU;
}

float getIOURot(vector<Point2f> points,Point2f a1,Point2f a2,int tcols,int trows)
{
    if(points.size() == 0) return 0;
    RotatedRect A = RotatedRect((points[0]+points[2])*0.5,Size(tcols,trows),0);
    RotatedRect B = RotatedRect((a1+a2)*0.5,Size(tcols,trows),0);
    float AREA_A = A.size.height*A.size.width;
    float AREA_B = B.size.height*B.size.width;
    float IOU = 0;
    vector<Point2f> m_points;
    rotatedRectangleIntersection(A,B,m_points);
    if(m_points.size() >= 4)
    {
        float INS_AREA = cv::norm(m_points[0]-m_points[1])*cv::norm(m_points[1]-m_points[2]);
        IOU = INS_AREA / float(AREA_A + AREA_B - INS_AREA);
    }
    return IOU;
}

void evaluateApplicationA(TemplateMatching *templateMatching)
{
    string dataFolder = evaluateFolderPath;
    string resultFile = resultFolder + "ApplicationA_Performance.txt";
    clock_t start;
    clock_t stop;
    BaseTemplateMatching baseTemplateMatching;
    Point2f pointRef,pointA,pointB;
    float iouA,iouB;
    double timeA,timeB;

    boost::filesystem::create_directories(resultFolder);
    ofstream myfile;
    myfile.open(resultFile);

    cout << "Image,Test,Class,FEA,IOUA,IOUB,TIMEA,TIMEB" << endl;
    myfile << "Image,Test,Class,FEA,IOUA,IOUB,TIMEA,TIMEB" << endl;

    int ImgNumber = 1;
    for(int c = 1; c <= 4; c++)
    {
        int trows = 0;
        int tcols = 0;
        string cls(1, char(64+c));
        string folder = dataFolder + to_string(c);
        switch(c)
        {
            case 1:trows = 132;tcols = 134;break;//480,640
            case 2:trows = 202;tcols = 202;break;//640,800
            case 3:trows = 254;tcols = 298;break;//640,800
            case 4:trows = 302;tcols = 356;break;//960,1280
        }
        vector<string> files;
        for (boost::filesystem::directory_iterator di(folder);di != boost::filesystem::directory_iterator(); di++)
        {
            files.push_back(di->path().c_str());
        }
        sort(files.begin(),files.end());
        for (int f=0; f < files.size(); f++)
        {
            string path = files[f];
            Mat srcImg = imread(path);
            int srows = srcImg.rows;
            int scols = srcImg.cols;
            pointRef = Point2f(tcols+rand()%10,trows+rand()%10);

            Mat tmpImg = srcImg(Rect(pointRef,Size(tcols,trows))).clone();
            int htrows = trows>>1;
            int htcols = tcols>>1;

            for(int t=0; t < 7; t++)
            {
                Mat src,tmp,R;
                string test;
                Point2f pt_a1 = pointRef,pt_a2 = Point2f(pt_a1.x+tcols,pt_a1.y+trows);
                tmp = tmpImg;
                switch(t)
                {
                    case 0: test = "ORIGINAL";
                            src = srcImg.clone();
                            break;
                    case 1: test = "GNO_0_30";
                            addNoise(&srcImg,&src,0.0,30.0);
                            break;
                    case 2: test = "GBR_1.5_0";
                            GaussianBlur(srcImg,src,Size(3,3),1.5);
                            break;
                    case 3: test = "GBR_2.5_0";
                            GaussianBlur(srcImg,src,Size(5,5),2.5);
                            break;
                    case 4: test = "ROTATE_5";
                            {
                                R = cv::getRotationMatrix2D(Point2f(scols>>1,srows>>1),5,1);
                                cv::warpAffine(srcImg,src,R,Size(scols,srows));
                                vector<Point2f> oldV,newV;
                                oldV.push_back(pt_a1);
                                oldV.push_back(pt_a2);
                                transform(oldV, newV, R);
                                pt_a1 = newV[0];
                                pt_a2 = newV[1];
                            }
                            break;
                    case 5: test = "ROTATE_10";
                            {
                                R = cv::getRotationMatrix2D(Point2f(scols>>1,srows>>1),10,1);
                                cv::warpAffine(srcImg,src,R,Size(scols,srows));
                                vector<Point2f> oldV,newV;
                                oldV.push_back(pt_a1);
                                oldV.push_back(pt_a2);
                                transform(oldV, newV, R);
                                pt_a1 = newV[0];
                                pt_a2 = newV[1];
                            }
                            break;
                    case 6: test = "OCCLUDED";
                            {
                                Point tmpRef;
                                src = srcImg.clone();
                                tmp = tmpImg.clone();
                                while(true)
                                {
                                    int x = rand() % tcols;
                                    int y = rand() % trows;
                                    if( x+htcols < tcols && y+htrows < trows)
                                    {
                                        tmpRef = Point2f(x,y);
                                        break;
                                    }
                                }
                                Mat roi = tmp(Rect(tmpRef,Size(htcols,htrows)));
                                roi.setTo(0);
                            }
                            break;
                }

                //cout << "Start : Image correlation" << endl;
                Mat srcGray;
                Mat tmpGray;
                cvtColor(src,srcGray,COLOR_BGR2GRAY);
                cvtColor(tmp,tmpGray,COLOR_BGR2GRAY);
                start = clock();
                pointA = baseTemplateMatching.getMatch(&srcGray,&tmpGray,24);
                stop = clock();
                timeA = double(stop - start) / CLOCKS_PER_SEC;
                iouA = getIOU(pt_a1,pt_a2,pointA,Point2f(pointA.x+tcols,pointA.y+trows));

                //cout << "Start : Image correlation with RF distance" << endl;
                start = clock();
                pointB = templateMatching->getMatch(&src,&tmp);
                stop = clock();
                timeB = double(stop - start) / CLOCKS_PER_SEC;
                iouB = getIOU(pt_a1,pt_a2,pointB,Point2f(pointB.x+tcols,pointB.y+trows));

                cout << ImgNumber << "," << test << "," << cls << "," << baseTemplateMatching.features;
                cout << "," << iouA << "," << iouB << "," << timeA << "," << timeB << endl;

                myfile << ImgNumber << "," << test << "," << cls << "," << baseTemplateMatching.features;
                myfile << "," << iouA << "," << iouB << "," << timeA << "," << timeB << endl;

                ImgNumber++;
                src.release();
                tmp.release();
            }
            srcImg.release();
            tmpImg.release();
        }
    }
    myfile.close();
    cout << "Evaluation complete. File saved under : " << resultFile << endl;
}

void evaluateApplicationB(TemplateMatching *templateMatching)
{
    string dataFolder = evaluateFolderPath;
    string resultFile = resultFolder + "ApplicationB_Performance.txt";
    clock_t start;
    clock_t stop;
    Point2f pointRef,pointA,pointB;
    float iouA,iouB;
    double timeA,timeB;

    boost::filesystem::create_directories(resultFolder);
    ofstream myfile;
    myfile.open(resultFile);

    cout << "Image,Test,Class,IOUC,IOUD,TIMEC,TIMED" << endl;
    myfile << "Image,Test,Class,IOUC,IOUD,TIMEC,TIMED" << endl;

    int ImgNumber = 1;
    for(int c = 1; c <= 4; c++)
    {
        int trows = 0;
        int tcols = 0;
        string cls(1, char(64+c));
        string folder = dataFolder + to_string(c);
        switch(c)
        {
            case 1:trows = 132;tcols = 134;break;//480,640
            case 2:trows = 202;tcols = 202;break;//640,800
            case 3:trows = 254;tcols = 298;break;//640,800
            case 4:trows = 302;tcols = 356;break;//960,1280
        }
        vector<string> files;
        for (boost::filesystem::directory_iterator di(folder);di != boost::filesystem::directory_iterator(); di++)
        {
            files.push_back(di->path().c_str());
        }
        sort(files.begin(),files.end());
        for (int f=0; f < files.size(); f++)
        {
            string path = files[f];
            Mat srcImg = imread(path);
            int srows = srcImg.rows;
            int scols = srcImg.cols;
            pointRef = Point2f(tcols+rand()%100,trows+rand()%100);

            Mat tmpImg = srcImg(Rect(pointRef,Size(tcols,trows))).clone();
            int htrows = trows>>1;
            int htcols = tcols>>1;

            for(int t=0; t < 7; t++)
            {
                Mat src,tmp,R;
                string test;
                Point2f pt_a1 = pointRef,pt_a2 = Point2f(pt_a1.x+tcols,pt_a1.y+trows);
                tmp = tmpImg;
                switch(t)
                {
                    case 0: test = "ORIGINAL";
                            src = srcImg.clone();
                            break;
                    case 1: test = "GNO_0_30";
                            addNoise(&srcImg,&src,0.0,30.0);
                            break;
                    case 2: test = "GBR_1.5_0";
                            GaussianBlur(srcImg,src,Size(3,3),1.5);
                            break;
                    case 3: test = "GBR_2.5_0";
                            GaussianBlur(srcImg,src,Size(5,5),2.5);
                            break;
                    case 4: test = "ROTATE_5";
                            {
                                R = cv::getRotationMatrix2D(Point2f(scols>>1,srows>>1),5,1);
                                cv::warpAffine(srcImg,src,R,Size(scols,srows));
                                vector<Point2f> oldV,newV;
                                oldV.push_back(pt_a1);
                                oldV.push_back(pt_a2);
                                transform(oldV, newV, R);
                                pt_a1 = newV[0];
                                pt_a2 = newV[1];
                            }
                            break;
                    case 5: test = "ROTATE_10";
                            {
                                R = cv::getRotationMatrix2D(Point2f(scols>>1,srows>>1),10,1);
                                cv::warpAffine(srcImg,src,R,Size(scols,srows));
                                vector<Point2f> oldV,newV;
                                oldV.push_back(pt_a1);
                                oldV.push_back(pt_a2);
                                transform(oldV, newV, R);
                                pt_a1 = newV[0];
                                pt_a2 = newV[1];
                            }
                            break;
                    case 6: test = "OCCLUDED";
                            {
                                Point tmpRef;
                                src = srcImg.clone();
                                tmp = tmpImg.clone();
                                while(true)
                                {
                                    int x = rand() % tcols;
                                    int y = rand() % trows;
                                    if( x+htcols < tcols && y+htrows < trows)
                                    {
                                        tmpRef = Point2f(x,y);
                                        break;
                                    }
                                }
                                Mat roi = tmp(Rect(tmpRef,Size(htcols,htrows)));
                                roi.setTo(0);
                            }
                            break;
                }

                if(t == 4 || t == 5)
                {
                    //cout << "Start : SURF key patch matching with SURF descriptor distance" << endl;
                    start = clock();
                    vector<Point2f> pointsA = templateMatching->getMatchKeyRot(&src,&tmp);
                    stop = clock();
                    timeA = double(stop - start) / CLOCKS_PER_SEC;
                    iouA = getIOURot(pointsA,pt_a1,pt_a2,tcols,trows);

                    //cout << "Start : SURF key patch matching with RF distance" << endl;
                    start = clock();
                    vector<Point2f> pointsB = templateMatching->getMatchKeyRot(&src,&tmp,TemplateMatching::SURF_KEYS,RandomForest::RF_SIMILARITY_BY_PATH,TemplateMatching::KEYPOINTS_NO);
                    stop = clock();
                    timeB = double(stop - start) / CLOCKS_PER_SEC;
                    iouB = getIOURot(pointsB,pt_a1,pt_a2,tcols,trows);
                }
                else
                {
                    //cout << "Start : SURF key patch matching with SURF descriptor distance" << endl;
                    start = clock();
                    pointA = templateMatching->getMatchKey(&src,&tmp);
                    stop = clock();
                    timeA = double(stop - start) / CLOCKS_PER_SEC;
                    iouA = getIOU(pt_a1,pt_a2,pointA,Point2f(pointA.x+tcols,pointA.y+trows));

                    //cout << "Start : SURF key patch matching with RF distance" << endl;
                    start = clock();
                    pointB = templateMatching->getMatchKey(&src,&tmp,TemplateMatching::SURF_KEYS,RandomForest::RF_SIMILARITY_BY_PATH,TemplateMatching::KEYPOINTS_NO);
                    stop = clock();
                    timeB = double(stop - start) / CLOCKS_PER_SEC;
                    iouB = getIOU(pt_a1,pt_a2,pointB,Point2f(pointB.x+tcols,pointB.y+trows));
                }

                cout << ImgNumber << "," << test << "," << cls;
                cout << "," << iouA << "," << iouB << "," << timeA << "," << timeB << endl;

                myfile << ImgNumber << "," << test << "," << cls;
                myfile << "," << iouA << "," << iouB << "," << timeA << "," << timeB << endl;

                ImgNumber++;
                src.release();
                tmp.release();
            }
            srcImg.release();
            tmpImg.release();
        }
    }
    myfile.close();
    cout << "Evaluation complete. File saved under : " << resultFile << endl;
}

int main()
{
    srand(time(nullptr));
    TemplateMatching *templateMatching = nullptr;
    int train = 0;
    while(true)
    {
        string error = "Please pass the correct arguments!!";
        int flag = 0;
        cout << endl << endl;
        cout << "Generate OOB statistics - 1" << endl;
        cout << "Train Random forest - 2" << endl;
        cout << "Evaluate applications - 3" << endl;
        cout << "Demonstrate - 4" << endl ;
        cout << "Test - 5" << endl ;
        cout << "Please select : ";
        int arg1; cin >> arg1; cout << endl;

        if(arg1 == 1)
        {
            cout << "Entropy statistics - 0, Random statistics - 1 : ";
            int arg2; cin >> arg2; cout << endl;

            if(arg2 == 0)
            {
                statisticsRF(trainingFolderPath,RandomForest::RF_FIT_ENTROPY);
            }
            else if(arg2 == 1)
            {
                statisticsRF(trainingFolderPath,RandomForest::RF_FIT_RANDOM);
            }
            else
            {
                flag = 1;
            }
        }
        else if(arg1 == 2)
        {
            cout << "Trees to train : " ;
            int treesTrain; cin >> treesTrain; cout << endl;
            cout << "Trees to load : " ;
            int treesLoad; cin >> treesLoad; cout << endl;
            cout << "Tree depth : " ;
            int depth; cin >> depth; cout << endl;
            cout << "Training method - Entropy (0), Random (1) : ";
            int method; cin >> method; cout << endl;
            cout << "Print trees - Yes (1), No (0) : " ;
            int print; cin >> print; cout << endl;

            if(method == 0)
            {
                trainRF(trainingFolderPath,treesTrain,depth,RandomForest::RF_FIT_ENTROPY);
                templateMatching = loadRF(treesLoad,print == 1?RandomForest::PRINT_YES:RandomForest::PRINT_NO);
                train = 1;
            }
            else if(method == 1)
            {
                trainRF(trainingFolderPath,treesTrain,depth,RandomForest::RF_FIT_RANDOM);
                templateMatching = loadRF(treesLoad,print == 1?RandomForest::PRINT_YES:RandomForest::PRINT_NO);
                train = 1;
            }
            else
            {
                flag = 1;
            }
        }
        else if(arg1 == 3)
        {
            if(train)
            {
                cout << "Evaluate applications A and B - 0,  Evaluate applications C and D - 1 : ";
                int arg2; cin >> arg2; cout << endl;

                if(arg2 == 0)
                {
                    evaluateApplicationA(templateMatching);
                }
                else if(arg2 == 1)
                {
                    evaluateApplicationB(templateMatching);
                }
                else
                {
                    flag = 1;
                }
            }
            else
            {
                error = "Please run training atleast once before evaluating!!";
                flag = 1;
            }
        }
        else if(arg1 == 4)
        {
            if(train)
            {
                cout << "Test methods" << endl;
                cout << "OpenCV - 0" << endl;
                cout << "Application A - 1" << endl;
                cout << "Application B - 2" << endl;
                cout << "Application C - 3" << endl;
                cout << "Application D - 4" << endl;
                cout << "Real time demo using RF distance - 5" << endl;
                cout << "Please select : ";
                int arg2; cin >> arg2; cout << endl;

                if(arg2 >=0 && arg2 <=5)
                {
                    demonstrate(arg2,templateMatching);
                }
                else
                {
                    flag = 1;
                }
            }
            else
            {
                error = "Please run training atleast once before evaluating!!";
                flag = 1;
            }
        }
        else if(arg1 == 5)
        {
            if(train)
            {
                cout << "Test methods" << endl;
                cout << "OpenCV - 0" << endl;
                cout << "Application A - 1" << endl;
                cout << "Application B - 2" << endl;
                cout << "Application C - 3" << endl;
                cout << "Application D - 4" << endl;
                cout << "Please select : ";
                int arg2; cin >> arg2; cout << endl;
                cout << "Source location : ";
                string arg3; cin >> arg3; cout << endl;
                cout << "Template location : ";
                string arg4; cin >> arg4; cout << endl;

                if(arg2 >=0 && arg2 <=4)
                {
                    test(arg2,arg3,arg4,templateMatching);
                }
                else
                {
                    flag = 1;
                }
            }
            else
            {
                error = "Please run training atleast once before evaluating!!";
                flag = 1;
            }
        }
        else
        {
            flag = 1;
        }
        if(flag)
        {
            cout << error << endl << endl;
        }
    }
    return 0;
}


