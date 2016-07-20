#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include "lib/common.h"
#include "lib/ImageDatabase.h"

using namespace cv;
using namespace std;


int readDatabse(int argc, char** argv){
  const char* dbFName = argv[1];
  ImageDatabase db(dbFName);
  for(int i=0; i<db.getLabels().size(); ++i)
    cout << db.getFilenames()[i] << " " << db.getLabels()[i] << ' ' << endl;
  cout << db<< endl;
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
