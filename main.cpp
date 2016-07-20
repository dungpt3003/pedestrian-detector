#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include "lib/common.h"
#include "lib/ImageDatabase.h"


using namespace cv;
using namespace std;

// Directory containing positive sample images
static string posSamplesDir = "data/pos/";
// Directory containing negative sample images
static string negSamplesDir = "data/neg/";
// Set the file to write the features to
static string featuresFile = "genfiles/features.dat";
// Set the file to write the SVM model to
static string svmModelFile = "genfiles/svmlightmodel.dat";
// Set the file to write the resulting detecting descriptor vector to
static string descriptorVectorFile = "genfiles/descriptorvector.dat";
// Set the file to write the resulting opencv hog classifier as YAML file
static string cvHOGFile = "genfiles/cvHOGClassifier.yaml";

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
