TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += \
    DecisionTreeClassifier.cpp \
    RandomForest.cpp \
    RandomForestTemplateMatching.cpp \
    TemplateMatching.cpp \
    BaseTemplateMatching.cpp \
    LegacyTemplateMatching.cpp

HEADERS += \
    DecisionTreeClassifier.h \
    RandomForest.h \
    Utilities.h \
    TemplateMatching.h \
    BaseTemplateMatching.h \
    LegacyTemplateMatching.h

LIBS += `pkg-config --libs opencv`
LIBS += -lboost_filesystem -lboost_regex -lboost_system
