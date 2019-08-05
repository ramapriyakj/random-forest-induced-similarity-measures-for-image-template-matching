--------------------------------------------------------------------------------------------------
# Random Forest Induced Similarity Measures for Image Template Matching

Author  - Ramapriya Janardhan


Email   - raghuiyengar.kj@gmail.com


Purpose - The main goal of this work is to demonstrate the use of Random forest similarity measure in the context of image template matching.
__________________________________________________________________________________________________

Pre-requisites to run this work:
---------------------------------------------------------------------------------------------------
The following modules are required to run this work:


OpenCV 3


OpenCV extra modules


Boost libraries


Run the commands in configuration.txt in Ubuntu 18.04 to install all the modules mentioned above.
__________________________________________________________________________________________________

Note:
---------------------------------------------------------------------------------------------------
rf_template_matching/RandomForestTemplateMatching - This folder contains source code to run this work.
rf_template_matching/helpers - This folder contains data set needed for training and testing.
__________________________________________________________________________________________________

Run the below commands to build and run the project:
---------------------------------------------------------------------------------------------------
Build:
-------
```
cd rf_template_matching/build
rm -rf *
cmake ..
make
```
--------
Run:
--------
```
./RandomForestTemplateMatching
```
__________________________________________________________________________________________________

Explanations for options shown when the project is run:
---------------------------------------------------------------------------------------------------

Note:

Entropy mode - In Entropy mode, the tests in decison nodes are chosen to maximize information gain.


Random mode - In Random mode the tests in decison nodes are randomly generated without maximize information gain.


1. Generate OOB statistics:
---------------------------
This option generates OOB statistics for random forest trained on CIFAR-10 dataset.


The statistics can be generated in two modes - Entropy mode and Random mode


2. Train Random forest:
------------------------
This option is to train the Random forest. This option has to be run atleast once before choosing options 3,4 and 5.


Arguments to pass: Number of trees to train, Number of trees to load for testing, Depth of each tree (-1 for maximum depth) and an option to print the trees.


The random forest can be trained in two modes - Entropy mode and Random mode


3. Evaluate applications:
------------------------
1. Application A - Template matching using Rectangle filters
2. Application B - Template matching using Random forest similarity score
3. Application C - Template matching using SIFT key point matching
4. Application D - Template matching using Random forest similarity distance

THis option evaluate above applications and generate performance statistics.
Applciation A will be evaluated against application B.
Applciation C will be evaluated against application D.

4. Demonstrate:
------------------------
This option demonstrates all the above applications using some temporary test images.


5. Test:
------------------------
This option is to test all the above applications by passing source and tempate location.













