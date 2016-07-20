PROJECT(pedestriandetector)

CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

# Where to search for cmake scripts
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# OpenCV
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

# Build subdirectories
ADD_SUBDIRECTORY(thirdparty/libsvm-3.14)
ADD_SUBDIRECTORY(thirdparty/svmlight)

# Building the project
ADD_EXECUTABLE(pd main.cpp)

target_link_libraries( pd ${OpenCV_LIBS} svmlight )
