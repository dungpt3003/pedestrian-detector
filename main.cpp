#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "lib/common.h"
#include "lib/ImageDatabase.h"

using namespace cv;
using namespace std;


int readDatabse(int argc, char** argv){
  const char* dbFName = argv[1];
  ImageDatabase db(dbFName);
  cout << db << endl;
  return EXIT_SUCCESS;
}

int main(int argc, char** argv )
{
    if ( argc != 2 )
    {
        printf("Usage: ./pd path_to_database");
        return -1;
    }

    return readDatabse(argc, argv);
}
