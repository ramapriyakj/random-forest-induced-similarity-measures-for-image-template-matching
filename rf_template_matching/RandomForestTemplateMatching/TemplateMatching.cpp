/*
 * File_name : TemplateMatching.cpp
 * Purpose   : The file contains methods to perform template matching
 *
 * Author    : Ramapriya Janardhan
 */

#include "TemplateMatching.h"
#define BEGIN start = clock();
#define END(message) stop = clock();elapsed_secs = double(stop - start) / CLOCKS_PER_SEC; cout << message << elapsed_secs << endl;

static clock_t start;
static clock_t stop;
static double elapsed_secs;
int TemplateMatching::patchSize = 24;
string TemplateMatching::trainResultPath = "../helpers/rf/train/";
string TemplateMatching::statisticsOOBResultPath = "../helpers/rf/statistics/oob/";
string TemplateMatching::randomForestPath = "../helpers/rf/train/RandomForestDecisionTrees.txt";

TemplateMatching::TemplateMatching()
{
    boost::filesystem::create_directories(trainResultPath);
    boost::filesystem::create_directories(statisticsOOBResultPath);
}

Mat* TemplateMatching::getBinData(int *classes,string trainFolder)
{
    cout << "Loading images" << endl;
    vector<string> binFiles;
    int len = 0;
    {
        for (boost::filesystem::directory_iterator di(trainFolder);di != boost::filesystem::directory_iterator(); di++)
        {
            string path = di->path().c_str();
            if (boost::filesystem::is_regular_file(path))
            {
                if(boost::algorithm::ends_with(path, ".bin"))
                {
                    binFiles.push_back(path);
                }
            }
        }
        len = binFiles.size();
    }
    Mat *imageData = new Mat[2];
    unordered_set<char> labels;
    int images = 10000;
    int isize = 32;

    for(int i = 0; i < len ; i++)
    {
        ifstream myfile;
        myfile.open(binFiles[i],ios::in|ios::binary);
        if(myfile)
        {
            for(int s=0; s<images; s++)
            {
                char label_data;
                myfile.read(&label_data, 1);
                labels.insert(label_data);
                int label = static_cast<int>(label_data);

                Mat image = Mat::zeros(isize, isize, CV_8UC3);
                vector<Mat> channels;
                for(int ch=0; ch<3; ch++)
                {
                    Mat cmat = Mat::zeros(isize,isize,CV_8UC1);
                    for(int r=0; r < isize; r++)
                    {
                        for(int c=0; c < isize; c++)
                        {
                            char data;
                            myfile.read(&data, 1);
                            cmat.at<uchar>(r,c) = data;
                        }
                    }
                    channels.push_back(cmat);
                }
                merge(channels,image);
                cv::resize(image,image,Size(patchSize,patchSize));
                imageData[0].push_back(image.reshape(1,1));
                imageData[1].push_back(label);

                //Transformations
                //Fip
                Mat t1;
                cv::flip(image,t1,1);
                imageData[0].push_back(t1.reshape(1,1));
                imageData[1].push_back(label);

                //Rotate
                Mat t2;
                Mat R = cv::getRotationMatrix2D(Point2f(patchSize>>1,patchSize>>1),90,1);
                cv::warpAffine(image,t2,R,Size(patchSize,patchSize));
                imageData[0].push_back(t2.reshape(1,1));
                imageData[1].push_back(label);
            }
        }
        myfile.close();
     }
    *classes = labels.size();
    cout << "Loading images completed" << endl;
    return imageData;
}

void TemplateMatching::train(string trainFolder,int trees,int depth,int method)
{
    RandomForest rf;
    int classes = 0;
    Mat *imageData = getBinData(&classes,trainFolder);
    rf.initialize(trees,imageData,depth,classes);
    cout << "Training Random Forest" << endl;
    BEGIN
    rf.fit(method);
    END("Training Random Forest completed. Total training time : ")
    rf.saveDecisionTreesObject(randomForestPath);
    cout << "Forest saved at : " << randomForestPath << endl;
}

void TemplateMatching::load(int trees,int print)
{
    BEGIN
    cout << "Loading Random Forest" << endl;
    randomForest.loadRandomForest(randomForestPath,trees);
    END("Loading Random Forest completed. Total loading time : ");
    if(print)
    {
        randomForest.printDecisionTrees();
    }
}

void TemplateMatching::getOOBStatistics(string trainFolder,int method)
{
    RandomForest rf;
    int classes = 0;
    Mat *imageData = getBinData(&classes,trainFolder);
    rf.initialize(0,imageData,0,classes);
    cout << "Generating OOB statistics for Random Forest" << endl;
    rf.printAndSaveStatisticsOOB(statisticsOOBResultPath,method);
    cout << "Generating OOB statistics for Random Forest completed" << endl;
}

Point TemplateMatching::getBestCenterKey(vector<Point2f> *spoints,vector<Point2f> *tpoints,int trows,int tcols)
{
    std::vector<Point2f> scene_corners(4);
    std::vector<Point2f> obj_corners(4);
    obj_corners[0] = cvPoint(0,0);
    obj_corners[1] = cvPoint(tcols, 0 );
    obj_corners[2] = cvPoint(tcols,trows);
    obj_corners[3] = cvPoint(0,trows);
    int flag = 0;
    if(tpoints->size() > 0)
    {
        Mat H = findHomography(*tpoints, *spoints, CV_RANSAC );
        if(H.rows > 0)
        {
            perspectiveTransform(obj_corners, scene_corners, H);
            flag = 1;
        }
    }
    if(flag == 1)
    {
        return scene_corners[0];
    }
    else
    {
        return Point2f(0,0);
    }
}

Point TemplateMatching::getMatchKey(Mat *sourceImg,Mat *templateImg,int keys,int similarity,int showKeys)
{
    int trows = templateImg->rows;
    int tcols = templateImg->cols;
    int srows = sourceImg->rows;
    int scols = sourceImg->cols;
    Point2f ref = Point2f(patchSize>>1,patchSize>>1);
    cv::Ptr<cv::FeatureDetector> KeyPointDetector;
    if(keys == TemplateMatching::SURF_KEYS)
    {
        KeyPointDetector = cv::xfeatures2d::SurfFeatureDetector::create(300);
    }
    else if(keys == TemplateMatching::SIFT_KEYS)
    {
        KeyPointDetector = cv::xfeatures2d::SiftFeatureDetector::create(700);
    }
    else
    {
        KeyPointDetector = cv::FastFeatureDetector::create(40);
    }

    Mat templateData;
    vector<Point2f> templatePoints;
    vector<KeyPoint> templateKeyPoints;
    KeyPointDetector->detect(*templateImg, templateKeyPoints);
    for (vector<KeyPoint>::iterator it = templateKeyPoints.begin(); it != templateKeyPoints.end(); it++)
    {
        Point2f p = it->pt - ref;
        if ((p.x < 0) || (p.y < 0) || ((p.x + patchSize) >= tcols) || ((p.y + patchSize) >= trows))
        {
            continue;
        }
        templatePoints.push_back(it->pt);
        Mat data = (*templateImg)(Rect(p.x, p.y, patchSize, patchSize)).clone();
        templateData.push_back(data.reshape(1,1));
    }

    Mat sourceData;
    vector<Point2f> sourcePoints;
    vector<KeyPoint> sourceKeyPoints;
    KeyPointDetector->detect(*sourceImg, sourceKeyPoints);
    for (vector<KeyPoint>::iterator it = sourceKeyPoints.begin(); it != sourceKeyPoints.end(); it++)
    {
        Point2f p = it->pt - ref;
        if ((p.x < 0) || (p.y < 0) || ((p.x + patchSize) >= scols) || ((p.y + patchSize) >= srows))
        {
            continue;
        }
        sourcePoints.push_back(it->pt);
        Mat data = (*sourceImg)(Rect(p.x, p.y, patchSize, patchSize)).clone();
        sourceData.push_back(data.reshape(1,1));
    }
    Match *match = nullptr;
    if(similarity == RandomForest::RF_SIMILARITY_BY_LEAF_NODE)
    {
        match = randomForest.findSimilarityByLeafNode(&sourceData,&templateData);
    }
    else if(similarity == RandomForest::RF_SIMILARITY_BY_PATH)
    {
        match = randomForest.findSimilarityByPath(&sourceData,&templateData);
    }
    vector<Point2f> matchSourcePoints;
    vector<Point2f> matchTemplatePoints;
    for(int i = 0; i < match->patchCount; i++)
    {
        //if(match->score[i] >= 0.4)
        //{
            matchSourcePoints.push_back(sourcePoints[match->patch[i]]);
            matchTemplatePoints.push_back(templatePoints[i]);
        //}
    }
    if(showKeys == TemplateMatching::KEYPOINTS_YES)
    {
        for(int i = 0;i < templatePoints.size();i++)
        {
            rectangle(*templateImg, Rect(templatePoints[i],Size(patchSize , patchSize)), Scalar(0, 255, 0));
        }
        for(int i = 0;i < sourcePoints.size();i++)
        {
            rectangle(*sourceImg, Rect(sourcePoints[i], Size(patchSize , patchSize)), Scalar(0, 255, 0));
        }
        for(int i = 0;i < matchSourcePoints.size();i++)
        {
            rectangle(*sourceImg, Rect(matchSourcePoints[i], Size(patchSize , patchSize)), Scalar(0, 0, 255));
        }
    }
    delete [] match->score;
    delete [] match->patch;
    delete match;
    return getBestCenterKey(&matchSourcePoints,&matchTemplatePoints,trows,tcols);
}

void TemplateMatching::matchTemplateKeyPoint(string sourcePath,string templatePath,int keys,int similarity,int showKeys)
{
    BEGIN
    Mat sourceImg = imread(sourcePath);
    Mat templateImg = imread(templatePath);
    Point2f point = getMatchKey(&sourceImg,&templateImg,keys,similarity,showKeys);
    END("Source[" + to_string(sourceImg.rows) + "x" + to_string(sourceImg.cols) + "],Template[" + to_string(templateImg.rows) + "x" + to_string(templateImg.cols) +  "], Processing time(s) : ")
    rectangle(sourceImg, Rect(point,Size(templateImg.cols , templateImg.rows)), Scalar::all(0), 2, 8, 0 );
    namedWindow("SOURCE", WINDOW_NORMAL);
    imshow("SOURCE", sourceImg);
    namedWindow("TEMPLATE", WINDOW_NORMAL);
    imshow("TEMPLATE", templateImg);
    waitKey(10);
}

void TemplateMatching::matchVideoKeyPoint(string sourcePath,string templatePath,int keys,int similarity,int showKeys)
{
    VideoCapture cap(sourcePath);
    Mat temp = imread(templatePath);
    namedWindow("SOURCEVIDEO", 200);
    namedWindow("TEMPLATEFRAME", 200);
    int count = 0;
    if(cap.isOpened())
    {
        while(1)
        {
            cout << "frame : " << count++ << endl;
            Mat templateImg;
            temp.copyTo(templateImg);
            Mat sourceImg;
            cap >> sourceImg;
            if (sourceImg.empty())
            {
                break;
            }
            Point2f point = getMatchKey(&sourceImg,&templateImg,keys,similarity,showKeys);
            rectangle(sourceImg, Rect(point,Size(templateImg.cols , templateImg.rows)), Scalar::all(0), 2, 8, 0 );
            imshow("SOURCEVIDEO", sourceImg);
            imshow("TEMPLATEFRAME", templateImg);
            waitKey(10);
        }
        cap.release();
        destroyAllWindows();
    }
    else
    {
        cout << "Error opening video stream or file" << endl;
    }
}

Point TemplateMatching::getMatchKey(Mat *sourceImg,Mat *templateImg)
{
    int trows = templateImg->rows;
    int tcols = templateImg->cols;

    Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create( 300 );
    vector<KeyPoint> templateKeyPoints,sourceKeyPoints;
    Mat templateDescriptors, sourceDescriptors;
    detector->detectAndCompute( *templateImg, noArray(), templateKeyPoints, templateDescriptors );
    detector->detectAndCompute( *sourceImg, noArray(), sourceKeyPoints, sourceDescriptors );

    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    std::vector< std::vector<DMatch> > knn_matches;
    matcher->knnMatch( templateDescriptors, sourceDescriptors, knn_matches, 2 );

    const float ratio_thresh = 0.75f;
    std::vector<DMatch> good_matches;
    for (int i = 0; i < knn_matches.size(); i++)
    {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
        {
            good_matches.push_back(knn_matches[i][0]);
        }
    }

    vector<Point2f> matchSourcePoints;
    vector<Point2f> matchTemplatePoints;

    for( int i = 0; i < good_matches.size(); i++ )
    {
        matchTemplatePoints.push_back( templateKeyPoints[ good_matches[i].queryIdx ].pt );
        matchSourcePoints.push_back( sourceKeyPoints[ good_matches[i].trainIdx ].pt );
    }

    std::vector<Point2f> scene_corners(4);
    std::vector<Point2f> obj_corners(4);
    obj_corners[0] = cvPoint(0,0);
    obj_corners[1] = cvPoint(tcols, 0 );
    obj_corners[2] = cvPoint(tcols,trows);
    obj_corners[3] = cvPoint(0,trows);
    int flag = 0;
    if(matchTemplatePoints.size() > 0)
    {
        Mat H = findHomography(matchTemplatePoints, matchSourcePoints, CV_RANSAC );
        if(H.rows > 0)
        {
            perspectiveTransform(obj_corners, scene_corners, H);
            flag = 1;
        }
    }

    Point2f point = Point2f(0,0);
    if(flag == 1)
    {
        point = scene_corners[0];
    }
    return point;
}

void TemplateMatching::matchTemplateKeyPoint(string sourcePath,string templatePath)
{
    BEGIN
    Mat sourceImg = imread(sourcePath);
    Mat templateImg = imread(templatePath);
    Point2f point = getMatchKey(&sourceImg,&templateImg);
    END("Source[" + to_string(sourceImg.rows) + "x" + to_string(sourceImg.cols) + "],Template[" + to_string(templateImg.rows) + "x" + to_string(templateImg.cols) +  "], Processing time(s) : ")
    rectangle(sourceImg, Rect(point,Size(templateImg.cols , templateImg.rows)), Scalar::all(0), 2, 8, 0 );
    namedWindow("SOURCE", WINDOW_NORMAL);
    imshow("SOURCE", sourceImg);
    namedWindow("TEMPLATE", WINDOW_NORMAL);
    imshow("TEMPLATE", templateImg);
    waitKey(10);
}

Point TemplateMatching::getMatch(Mat *sourceImg,Mat *templateImg)
{
    int trows = templateImg->rows;
    int tcols = templateImg->cols;
    int srows = sourceImg->rows;
    int scols = sourceImg->cols;

    Mat templateData;
    vector<Point2f> templatePoints;
    int patchCount = 5;
    int i = patchCount;
    while(1)
    {
        int x = rand()%trows;
        int y = rand()%tcols;
        if ((x + patchSize) >= tcols || (y + patchSize) >= trows)
        {
            continue;
        }
        if(i != 0)
        {
            templatePoints.push_back(Point2f(x,y));
            Mat data = (*templateImg)(Rect(x, y, patchSize, patchSize)).clone();
            templateData.push_back(data.reshape(1,1));
            i--;
        }
        else
        {
            break;
        }
    }
    randomForest.saveTemplateDecisionPathVector(&templateData);
    Point2f point = Point2f(0,0);
    int max = 0;
    for (int i = 0; i < srows; i++)
    {
        if (!(i + trows > srows))
        {
            for (int j = 0; j < scols; j++)
            {
                if (!(j + tcols > scols))
                {
                    Mat sourceData;
                    for(int k = 0; k < patchCount; k++)
                    {
                        Mat srcImg = (*sourceImg)(Rect(j,i,tcols,trows));
                        Mat data = srcImg(Rect(templatePoints[k],Size(patchSize,patchSize))).clone();
                        sourceData.push_back(data.reshape(1,1));
                    }
                    int score = randomForest.getSimilarityScoreByPath(&sourceData);
                    if(score > max)
                    {
                        max = score;
                        point = Point2f(j,i);
                    }
                    sourceData.release();
                }
            }
        }
    }
    randomForest.freeTemplateDecisionPathVector();
    return point;
}

void TemplateMatching::matchTemplate(string sourcePath,string templatePath)
{
    BEGIN
    Mat sourceImg = imread(sourcePath);
    Mat templateImg = imread(templatePath);
    Point2f point = getMatch(&sourceImg,&templateImg);
    END("Source[" + to_string(sourceImg.rows) + "x" + to_string(sourceImg.cols) + "],Template[" + to_string(templateImg.rows) + "x" + to_string(templateImg.cols) +  "], Processing time(s) : ")
    rectangle(sourceImg, Rect(point,Size(templateImg.cols , templateImg.rows)), Scalar::all(0), 2, 8, 0 );
    namedWindow("SOURCE", WINDOW_NORMAL);
    imshow("SOURCE", sourceImg);
    namedWindow("TEMPLATE", WINDOW_NORMAL);
    imshow("TEMPLATE", templateImg);
    waitKey(10);
}

vector<Point2f> TemplateMatching::getBestCenterKeyRot(vector<Point2f> *spoints,vector<Point2f> *tpoints,int trows,int tcols)
{
    std::vector<Point2f> scene_corners(4);
    std::vector<Point2f> obj_corners(4);
    obj_corners[0] = cvPoint(0,0);
    obj_corners[1] = cvPoint(tcols, 0 );
    obj_corners[2] = cvPoint(tcols,trows);
    obj_corners[3] = cvPoint(0,trows);
    if(tpoints->size() > 0)
    {
        Mat H = findHomography(*tpoints, *spoints, CV_RANSAC );
        if(H.rows > 0)
        {
            perspectiveTransform(obj_corners, scene_corners, H);
        }
    }
    return scene_corners;
}

vector<Point2f> TemplateMatching::getMatchKeyRot(Mat *sourceImg,Mat *templateImg,int keys,int similarity,int showKeys)
{
    int trows = templateImg->rows;
    int tcols = templateImg->cols;
    int srows = sourceImg->rows;
    int scols = sourceImg->cols;
    Point2f ref = Point2f(patchSize>>1,patchSize>>1);
    cv::Ptr<cv::FeatureDetector> KeyPointDetector;
    if(keys == TemplateMatching::SURF_KEYS)
    {
        KeyPointDetector = cv::xfeatures2d::SurfFeatureDetector::create(300);
    }
    else if(keys == TemplateMatching::SIFT_KEYS)
    {
        KeyPointDetector = cv::xfeatures2d::SiftFeatureDetector::create(700);
    }
    else
    {
        KeyPointDetector = cv::FastFeatureDetector::create(40);
    }

    Mat templateData;
    vector<Point2f> templatePoints;
    vector<KeyPoint> templateKeyPoints;
    KeyPointDetector->detect(*templateImg, templateKeyPoints);
    for (vector<KeyPoint>::iterator it = templateKeyPoints.begin(); it != templateKeyPoints.end(); it++)
    {
        Point2f p = it->pt - ref;
        if ((p.x < 0) || (p.y < 0) || ((p.x + patchSize) >= tcols) || ((p.y + patchSize) >= trows))
        {
            continue;
        }
        templatePoints.push_back(it->pt);
        Mat data = (*templateImg)(Rect(p.x, p.y, patchSize, patchSize)).clone();
        templateData.push_back(data.reshape(1,1));
    }

    Mat sourceData;
    vector<Point2f> sourcePoints;
    vector<KeyPoint> sourceKeyPoints;
    KeyPointDetector->detect(*sourceImg, sourceKeyPoints);
    for (vector<KeyPoint>::iterator it = sourceKeyPoints.begin(); it != sourceKeyPoints.end(); it++)
    {
        Point2f p = it->pt - ref;
        if ((p.x < 0) || (p.y < 0) || ((p.x + patchSize) >= scols) || ((p.y + patchSize) >= srows))
        {
            continue;
        }
        sourcePoints.push_back(it->pt);
        Mat data = (*sourceImg)(Rect(p.x, p.y, patchSize, patchSize)).clone();
        sourceData.push_back(data.reshape(1,1));
    }
    Match *match = nullptr;
    if(similarity == RandomForest::RF_SIMILARITY_BY_LEAF_NODE)
    {
        match = randomForest.findSimilarityByLeafNode(&sourceData,&templateData);
    }
    else if(similarity == RandomForest::RF_SIMILARITY_BY_PATH)
    {
        match = randomForest.findSimilarityByPath(&sourceData,&templateData);
    }
    vector<Point2f> matchSourcePoints;
    vector<Point2f> matchTemplatePoints;
    for(int i = 0; i < match->patchCount; i++)
    {
        //if(match->score[i] >= 0.4)
        //{
            matchSourcePoints.push_back(sourcePoints[match->patch[i]]);
            matchTemplatePoints.push_back(templatePoints[i]);
        //}
    }
    if(showKeys == TemplateMatching::KEYPOINTS_YES)
    {
        for(int i = 0;i < templatePoints.size();i++)
        {
            rectangle(*templateImg, Rect(templatePoints[i],Size(patchSize , patchSize)), Scalar(0, 255, 0));
        }
        for(int i = 0;i < sourcePoints.size();i++)
        {
            rectangle(*sourceImg, Rect(sourcePoints[i], Size(patchSize , patchSize)), Scalar(0, 255, 0));
        }
        for(int i = 0;i < matchSourcePoints.size();i++)
        {
            rectangle(*sourceImg, Rect(matchSourcePoints[i], Size(patchSize , patchSize)), Scalar(0, 0, 255));
        }
    }
    delete [] match->score;
    delete [] match->patch;
    delete match;
    return getBestCenterKeyRot(&matchSourcePoints,&matchTemplatePoints,trows,tcols);
}

vector<Point2f> TemplateMatching::getMatchKeyRot(Mat *sourceImg,Mat *templateImg)
{
    int trows = templateImg->rows;
    int tcols = templateImg->cols;

    Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create( 300 );
    vector<KeyPoint> templateKeyPoints,sourceKeyPoints;
    Mat templateDescriptors, sourceDescriptors;
    detector->detectAndCompute( *templateImg, noArray(), templateKeyPoints, templateDescriptors );
    detector->detectAndCompute( *sourceImg, noArray(), sourceKeyPoints, sourceDescriptors );

    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    std::vector< std::vector<DMatch> > knn_matches;
    matcher->knnMatch( templateDescriptors, sourceDescriptors, knn_matches, 2 );

    const float ratio_thresh = 0.75f;
    std::vector<DMatch> good_matches;
    for (int i = 0; i < knn_matches.size(); i++)
    {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
        {
            good_matches.push_back(knn_matches[i][0]);
        }
    }

    vector<Point2f> matchSourcePoints;
    vector<Point2f> matchTemplatePoints;

    for( int i = 0; i < good_matches.size(); i++ )
    {
        matchTemplatePoints.push_back( templateKeyPoints[ good_matches[i].queryIdx ].pt );
        matchSourcePoints.push_back( sourceKeyPoints[ good_matches[i].trainIdx ].pt );
    }

    std::vector<Point2f> scene_corners(4);
    std::vector<Point2f> obj_corners(4);
    obj_corners[0] = cvPoint(0,0);
    obj_corners[1] = cvPoint(tcols, 0 );
    obj_corners[2] = cvPoint(tcols,trows);
    obj_corners[3] = cvPoint(0,trows);
    if(matchTemplatePoints.size() > 0)
    {
        Mat H = findHomography(matchTemplatePoints, matchSourcePoints, CV_RANSAC );
        if(H.rows > 0)
        {
            perspectiveTransform(obj_corners, scene_corners, H);
        }
    }
    return scene_corners;
}
