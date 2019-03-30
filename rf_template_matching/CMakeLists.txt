cmake_minimum_required(VERSION 2.8)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

project(RandomForestTemplateMatching)

find_package(OpenCV 3 REQUIRED)
find_package(Boost 1.45.0 COMPONENTS filesystem regex system)

add_executable( RandomForestTemplateMatching 
	RandomForestTemplateMatching/DecisionTreeClassifier.cpp 
	RandomForestTemplateMatching/RandomForest.cpp 
	RandomForestTemplateMatching/RandomForestTemplateMatching.cpp 
	RandomForestTemplateMatching/TemplateMatching.cpp 
	RandomForestTemplateMatching/BaseTemplateMatching.cpp 
	RandomForestTemplateMatching/LegacyTemplateMatching.cpp )

# Link your application with OpenCV libraries
target_link_libraries(RandomForestTemplateMatching ${OpenCV_LIBS})
target_link_libraries(RandomForestTemplateMatching ${Boost_LIBRARIES})
