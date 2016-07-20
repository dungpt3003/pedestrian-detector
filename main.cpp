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

// HOG parameters for training that for some reason are not included in the HOG class
static const Size trainingPadding = Size(0, 0);
static const Size winStride = Size(8, 8);

/* Helper functions */

static string toLowerCase(const string& in) {
    string t;
    for (string::const_iterator i = in.begin(); i != in.end(); ++i) {
        t += tolower(*i);
    }
    return t;
}

static void storeCursor(void) {
    printf("\033[s");
}

static void resetCursor(void) {
    printf("\033[u");
}


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
