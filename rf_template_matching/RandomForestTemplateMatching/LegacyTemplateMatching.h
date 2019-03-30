#pragma once
#include <iostream>
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

using namespace cv;
using namespace std;

class LegacyTemplateMatching
{
    public:
        void displayTemplate(string sourcePath,string templatePath);
};
