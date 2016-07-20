CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

add_library(svmlight STATIC svm_common.c svm_learn.c svm_hideo.c)
