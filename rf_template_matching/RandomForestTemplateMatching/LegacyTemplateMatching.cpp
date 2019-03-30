/*
 * File_name : LegacyTemplateMatching.cpp
 * Purpose   : Template matching using opencv library
 * Author    : Ramapriya Janardhan
 */
#include "LegacyTemplateMatching.h"

void LegacyTemplateMatching::displayTemplate(string sourcePath,string templatePath)
{
    clock_t start;
    clock_t stop;
    double elapsed_secs;
    start = clock();
    Mat sourceImg = imread(sourcePath);
    Mat templateImg = imread(templatePath);
    Mat img_display;
    Mat result;

    int match_method = CV_TM_CCORR_NORMED;
    sourceImg.copyTo( img_display );
    int result_cols =  sourceImg.cols - templateImg.cols + 1;
    int result_rows = sourceImg.rows - templateImg.rows + 1;

    result.create( result_rows, result_cols, CV_32FC1 );

    /// Do the Matching and Normalize
    matchTemplate( sourceImg, templateImg, result, match_method);
    normalize( result, result, 0, 1, NORM_MINMAX, -1, Mat() );

    /// Localizing the best match with minMaxLoc
    double minVal; double maxVal; Point minLoc; Point maxLoc;
    Point2f matchLoc;

    minMaxLoc( result, &minVal, &maxVal, &minLoc, &maxLoc, Mat() );

    /// For SQDIFF and SQDIFF_NORMED, the best matches are lower values. For all the other methods, the higher the better
    if( match_method  == CV_TM_SQDIFF || match_method == CV_TM_SQDIFF_NORMED )
    {
        matchLoc = minLoc;
    }
    else
    {
        matchLoc = maxLoc;
    }

    /// Show me what you got
    rectangle(img_display, matchLoc, Point2f( matchLoc.x + templateImg.cols , matchLoc.y + templateImg.rows ), Scalar::all(0), 2, 8, 0 );

    stop = clock();
    elapsed_secs = double(stop - start) / CLOCKS_PER_SEC;
    cout << "Source[" + to_string(sourceImg.rows) + "x" + to_string(sourceImg.cols) + "],Template[" + to_string(templateImg.rows) + "x" + to_string(templateImg.cols) +  "], Processing time(s) : " << elapsed_secs << endl;
    namedWindow("SOURCE", WINDOW_NORMAL);
    namedWindow("TEMPLATE", WINDOW_NORMAL);
    imshow("SOURCE", img_display);
    imshow("TEMPLATE", templateImg);
    waitKey(10);
}
