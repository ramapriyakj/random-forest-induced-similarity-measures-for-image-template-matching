/*
 * File_name : BaseTemplateMatching.cpp
 * Purpose   : Re-implementation of work "Patch-based Image Correlation with Rapid Filtering"
 *             by Guodong Guo and Charles R. Dyer
 * Author    : Ramapriya Janardhan
 */

#include "BaseTemplateMatching.h"
double BaseTemplateMatching::getFeatureDifference(Mat *img,Filter *feature)
{
    double value = 0;
    switch(feature->filterNumber)
    {
        case 0:
        case 1:
                value = sum((*img)(feature->rectA))[0] - sum((*img)(feature->rectB))[0];
                break;
        case 2:
        case 3:
                value = sum((*img)(feature->rectA))[0] - sum((*img)(feature->rectB))[0] + sum((*img)(feature->rectC))[0];
                break;
        case 4:
                value = sum((*img)(feature->rectA))[0] - sum((*img)(feature->rectB))[0] - sum((*img)(feature->rectC))[0] + sum((*img)(feature->rectD))[0];
                break;
    }
    return abs(feature->filterValue - value);
}

void BaseTemplateMatching::populatePatchFeatures(Mat *img,int patchSize,vector<Filter> *arr,int tmpx,int tmpy)
{
    int filterRow,filterCol,subFilterSize;
    Size size;
    Point ref = Point(tmpx,tmpy);

    //Filter 1
    filterRow = 8;
    filterCol = 12;
    subFilterSize = 6;
    size = Size(subFilterSize, filterRow);
    for (int i = 0; i < patchSize; i = i + 4)
    {
        if (!(i + filterRow > patchSize))
        {
            for (int j = 0; j < patchSize; j = j + 4)
            {
                if (!(j + filterCol > patchSize))
                {
                    Point a = Point(j,i);
                    Point b = Point(j + subFilterSize, i);

                    Filter fe;
                    fe.filterNumber = 0;
                    fe.filterValue = sum((*img)(Rect(a,size)))[0] - sum((*img)(Rect(b,size)))[0];
                    fe.rectA = Rect(a+ref,size);
                    fe.rectB = Rect(b+ref,size);
                    arr->push_back(fe);
                }
            }
        }
    }

    //Filter 2
    filterRow = 12;
    filterCol = 8;
    subFilterSize = 6;
    size = Size(filterCol, subFilterSize);
    for (int i = 0; i < patchSize; i = i + 4)
    {
        if (!(i + filterRow > patchSize))
        {
            for (int j = 0; j < patchSize; j = j + 4)
            {
                if (!(j + filterCol > patchSize))
                {
                    Point a = Point(j,i);
                    Point b = Point(j, i + subFilterSize);

                    Filter fe;
                    fe.filterNumber = 1;
                    fe.filterValue = sum((*img)(Rect(a,size)))[0] - sum((*img)(Rect(b,size)))[0];
                    fe.rectA = Rect(a+ref,size);
                    fe.rectB = Rect(b+ref,size);
                    arr->push_back(fe);
                }
            }
        }
    }

    //Filter3
    filterRow = 8;
    filterCol = 12;
    subFilterSize = 4;
    size = Size(subFilterSize, filterRow);
    for (int i = 0; i < patchSize; i = i + 4)
    {
        if (!(i + filterRow > patchSize))
        {
            for (int j = 0; j < patchSize; j = j + 4)
            {
                if (!(j + filterCol > patchSize))
                {
                    Point a = Point(j,i);
                    Point b = Point(j + subFilterSize, i);
                    Point c = Point(j + subFilterSize + subFilterSize, i);

                    Filter fe;
                    fe.filterNumber = 2;
                    fe.filterValue = sum((*img)(Rect(a,size)))[0] - sum((*img)(Rect(b,size)))[0] + sum((*img)(Rect(c,size)))[0];
                    fe.rectA = Rect(a+ref,size);
                    fe.rectB = Rect(b+ref,size);
                    fe.rectC = Rect(c+ref,size);
                    arr->push_back(fe);
                }
            }
        }
    }

    //Filter 4
    filterRow = 12;
    filterCol = 8;
    subFilterSize = 4;
    size = Size(filterCol, subFilterSize);
    for (int i = 0; i < patchSize; i = i + 4)
    {
        if (!(i + filterRow > patchSize))
        {
            for (int j = 0; j < patchSize; j = j + 4)
            {
                if (!(j + filterCol > patchSize))
                {
                    Point a = Point(j,i);
                    Point b = Point(j, i + subFilterSize);
                    Point c = Point(j, i + subFilterSize + subFilterSize);

                    Filter fe;
                    fe.filterNumber = 3;
                    fe.filterValue = sum((*img)(Rect(a,size)))[0] - sum((*img)(Rect(b,size)))[0] + sum((*img)(Rect(c,size)))[0];
                    fe.rectA = Rect(a+ref,size);
                    fe.rectB = Rect(b+ref,size);
                    fe.rectC = Rect(c+ref,size);
                    arr->push_back(fe);
                }
            }
        }
    }

    //Filter 5
    filterRow = 8;
    filterCol = 8;
    subFilterSize = 4;
    size = Size(subFilterSize, subFilterSize);
    for (int i = 0; i < patchSize; i = i + 4)
    {
        if (!(i + filterRow > patchSize))
        {
            for (int j = 0; j < patchSize; j = j + 4)
            {
                if (!(j + filterCol > patchSize))
                {
                    Point a = Point(j,i);
                    Point b = Point(j + subFilterSize, i);
                    Point c = Point(j, i + subFilterSize);
                    Point d = Point(j + subFilterSize, i + subFilterSize);

                    Filter fe;
                    fe.filterNumber = 4;
                    fe.filterValue = sum((*img)(Rect(a,size)))[0] - sum((*img)(Rect(b,size)))[0] - sum((*img)(Rect(c,size)))[0] + sum((*img)(Rect(d,size)))[0];
                    fe.rectA = Rect(a+ref,size);
                    fe.rectB = Rect(b+ref,size);
                    fe.rectC = Rect(c+ref,size);
                    fe.rectD = Rect(d+ref,size);
                    arr->push_back(fe);
                }
            }
        }
    }
}

vector<Filter>* BaseTemplateMatching::getPatches(Mat *img, int patchSize)
{
    vector<Filter> *patches = new vector<Filter>();
    if (!img->empty() && patchSize > 0)
    {
        int rows = img->rows;
        int cols = img->cols;
        if(!(rows < patchSize || cols < patchSize))
        {
            for (int i = 0; i < rows; i = i + patchSize)
            {
                if (!(i + patchSize > rows))
                {
                    for (int j = 0; j < cols; j = j + patchSize)
                    {
                        if (!(j + patchSize > cols))
                        {
                            Mat tmpImg = (*img)(Rect(j, i, patchSize, patchSize));
                            populatePatchFeatures(&tmpImg, patchSize, patches,j,i);
                        }
                    }
                }
            }
        }
    }
    return patches;
}

Point BaseTemplateMatching::getMatch(Mat *sourceImg,Mat *templateImg,int patchSize)
{
    Point2f point = Point2f(0,0);
    if (!sourceImg->empty() && !templateImg->empty())
    {
        vector<Filter> *templatePatches = getPatches(templateImg, patchSize);
        double min = 0;
        vector<Filter> filterFeatures;
        double max = templatePatches->at(0).filterValue;
        double threshold;
        for(int i = 1; i < templatePatches->size(); i++)
        {
            if(max < templatePatches->at(i).filterValue)
            {
               max = templatePatches->at(i).filterValue;
            }
        }
        threshold = 0.95*max;
        for(vector<Filter>::iterator i = templatePatches->begin(); i != templatePatches->end(); i++)
        {
            if((*i).filterValue > threshold)
            {
               filterFeatures.push_back(*i);
            }
        }
        features = filterFeatures.size();
        int sourceRows = sourceImg->rows;
        int sourceCols = sourceImg->cols;
        int templateRows = templateImg->rows;
        int templateCols = templateImg->cols;
        for (int i = 0; i < sourceRows; i++)
        {
            if(!(i + templateRows > sourceRows))
            {
                for (int j = 0; j < sourceCols; j++)
                {
                    if(!(j + templateCols > sourceCols))
                    {
                        double score = 0;
                        for(vector<Filter>::iterator it = filterFeatures.begin(); it != filterFeatures.end(); it++)
                        {
                            Mat tmpImg = (*sourceImg)(Rect(j,i,templateCols,templateRows));
                            score = score + getFeatureDifference(&tmpImg,&(*it));
                        }
                        if(score < min || min == 0)
                        {
                            min = score;
                            point = Point2f(j,i);
                        }
                    }
                }
            }
        }
        delete templatePatches;
    }
    return point;
}

void BaseTemplateMatching::displayTemplate(string sourcePath,string templatePath,int patchSize)
{
    clock_t start;
    clock_t stop;
    double elapsed_secs;
    start = clock();
    Mat sourceImg = imread(sourcePath);
    Mat templateImg = imread(templatePath);
    Mat s;
    Mat t;
    cvtColor(sourceImg,s,COLOR_BGR2GRAY);
    cvtColor(templateImg,t,COLOR_BGR2GRAY);
    Point2f point = getMatch(&s,&t,patchSize);
    stop = clock();
    elapsed_secs = double(stop - start) / CLOCKS_PER_SEC;
    cout << "Source[" + to_string(sourceImg.rows) + "x" + to_string(sourceImg.cols) + "],Template[" + to_string(templateImg.rows) + "x" + to_string(templateImg.cols) +  "], Processing time(s) : " << elapsed_secs << endl;
    rectangle(sourceImg, Rect(point.x, point.y, templateImg.cols , templateImg.rows), Scalar::all(0), 2, 8, 0 );
    namedWindow("SOURCE",WINDOW_NORMAL);
    namedWindow("TEMPLATE",WINDOW_NORMAL);
    imshow("SOURCE", sourceImg);
    imshow("TEMPLATE", templateImg);
    waitKey(10);
}
