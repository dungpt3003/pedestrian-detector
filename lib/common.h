#ifndef COMMON_H
#define COMMON_H

// Standard library includes
#define _USE_MATH_DEFINES // To get M_PI back on windows
#include <vector>
#include <iostream>
#include <iomanip>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>
#include <cassert>
#include <stdexcept>
#include <set>
#include <algorithm>
#include <cstdlib>

// libsvm
#include "libsvm-3.14/svm.h"

// OpenCV 
#include <opencv2/opencv.hpp>

// Logging macro
#define PRINT_MSG(msg) std::cout << msg << std::endl

// Debugging macros, output suppressed in release mode
#ifdef NDEBUG
#define PRINT_EXPR(expr)
#else
#define PRINT_EXPR(expr) std::cout << #expr << " = " << (expr) << std::endl
#endif

#if defined(_WIN32) || defined(_WIN64)
  #define snprintf _snprintf
  #define vsnprintf _vsnprintf
  #define strcasecmp _stricmp
  #define strncasecmp _strnicmp
#endif

#endif
