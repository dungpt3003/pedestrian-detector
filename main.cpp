#include <stdio.h>
#include <dirent.h>
#ifdef __MINGW32__
#include <sys/stat.h>
#endif
#include <ios>
#include <fstream>
#include <stdexcept>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include "lib/common.h"
#include "lib/ImageDatabase.h"
#include "thirdparty/svmlight/svmlight.h"

#define SVMLIGHT 1
#define TRAINHOG_USEDSVM SVMLIGHT
#define TRAINHOG_SVM_TO_TRAIN SVMlight

using namespace cv;
using namespace std;

// Directory containing positive sample images
static string posSamplesDir = "../pedestrian-detector/data/images/pos/";
// Directory containing negative sample images
static string negSamplesDir = "../pedestrian-detector/data/images/neg/";
// Set the file to write the features to
static string featuresFile = "../pedestrian-detector/genfiles/features.dat";
// Set the file to write the SVM model to
static string svmModelFile = "../pedestrian-detector/genfiles/svmlightmodel.dat";
// Set the file to write the resulting detecting descriptor vector to
static string descriptorVectorFile = "../pedestrian-detector/genfiles/descriptorvector.dat";
// Set the file to write the resulting opencv hog classifier as YAML file
static string cvHOGFile = "../pedestrian-detector/genfiles/cvHOGClassifier.yaml";

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


static void saveDescriptorVectorToFile(vector<float>& descriptorVector, vector<unsigned int>& _vectorIndices, string fileName) {
    printf("Saving descriptor vector to file '%s'\n", fileName.c_str());
    string separator = " "; // Use blank as default separator between single features
    fstream File;
    float percent;
    File.open(fileName.c_str(), ios::out);
    if (File.good() && File.is_open()) {
        printf("Saving %lu descriptor vector features:\t", descriptorVector.size());
        storeCursor();
        for (int feature = 0; feature < descriptorVector.size(); ++feature) {
            if ((feature % 10 == 0) || (feature == (descriptorVector.size()-1)) ) {
                percent = ((1 + feature) * 100 / descriptorVector.size());
                printf("%4u (%3.0f%%)", feature, percent);
                fflush(stdout);
                resetCursor();
            }
            File << descriptorVector.at(feature) << separator;
        }
        printf("\n");
        File << endl;
        File.flush();
        File.close();
    }
}

static void getFilesInDirectory(const string& dirName, vector<string>& fileNames, const vector<string>& validExtensions) {
    printf("Opening directory %s\n", dirName.c_str());
#ifdef __MINGW32__
	struct stat s;
#endif
    struct dirent* ep;
    size_t extensionLocation;
    DIR* dp = opendir(dirName.c_str());
    if (dp != NULL) {
        while ((ep = readdir(dp))) {
            // Ignore (sub-)directories like . , .. , .svn, etc.
#ifdef __MINGW32__
			stat(ep->d_name, &s);
			if (s.st_mode & S_IFDIR) {
				continue;
			}
#else
            if (ep->d_type & DT_DIR) {
                continue;
            }
#endif
            extensionLocation = string(ep->d_name).find_last_of("."); // Assume the last point marks beginning of extension like file.ext
            // Check if extension is matching the wanted ones
            string tempExt = toLowerCase(string(ep->d_name).substr(extensionLocation + 1));
            if (find(validExtensions.begin(), validExtensions.end(), tempExt) != validExtensions.end()) {
                printf("Found matching data file '%s'\n", ep->d_name);
                fileNames.push_back((string) dirName + ep->d_name);
            } else {
                printf("Found file does not match required file type, skipping: '%s'\n", ep->d_name);
            }
        }
        (void) closedir(dp);
    } else {
        printf("Error opening directory '%s'!\n", dirName.c_str());
    }
    return;
}

static void calculateFeaturesFromInput(const string& imageFilename, vector<float>& featureVector, HOGDescriptor& hog) {
    /** for imread flags from openCV documentation,
     * @see http://docs.opencv.org/modules/highgui/doc/reading_and_writing_images_and_video.html?highlight=imread#Mat imread(const string& filename, int flags)
     * @note If you get a compile-time error complaining about following line (esp. imread),
     * you either do not have a current openCV version (>2.0)
     * or the linking order is incorrect, try g++ -o openCVHogTrainer main.cpp `pkg-config --cflags --libs opencv`
     */
    Mat imageData = imread(imageFilename, IMREAD_GRAYSCALE);
    if (imageData.empty()) {
        featureVector.clear();
        printf("Error: HOG image '%s' is empty, features calculation skipped!\n", imageFilename.c_str());
        return;
    }
    // Check for mismatching dimensions
    if (imageData.cols != hog.winSize.width || imageData.rows != hog.winSize.height) {
        featureVector.clear();
        printf("Error: Image '%s' dimensions (%u x %u) do not match HOG window size (%u x %u)!\n", imageFilename.c_str(), imageData.cols, imageData.rows, hog.winSize.width, hog.winSize.height);
        return;
    }
    vector<Point> locations;
    hog.compute(imageData, featureVector, winStride, trainingPadding, locations);
    imageData.release(); // Release the image again after features are extracted
}

/**
 * Shows the detections in the image
 * @param found vector containing valid detection rectangles
 * @param imageData the image in which the detections are drawn
 */
static void showDetections(const vector<Rect>& found, Mat& imageData) {
    vector<Rect> found_filtered;
    size_t i, j;
    for (i = 0; i < found.size(); ++i) {
        Rect r = found[i];
        for (j = 0; j < found.size(); ++j)
            if (j != i && (r & found[j]) == r)
                break;
        if (j == found.size())
            found_filtered.push_back(r);
    }
    for (i = 0; i < found_filtered.size(); i++) {
        Rect r = found_filtered[i];
        rectangle(imageData, r.tl(), r.br(), Scalar(64, 255, 64), 3);
    }
}

/**
 * Test the trained detector against the same training set to get an approximate idea of the detector.
 * Warning: This does not allow any statement about detection quality, as the detector might be overfitting.
 * Detector quality must be determined using an independent test set.
 * @param hog
 */
static void detectTrainingSetTest(const HOGDescriptor& hog, const double hitThreshold, const vector<string>& posFileNames, const vector<string>& negFileNames) {
    unsigned int truePositives = 0;
    unsigned int trueNegatives = 0;
    unsigned int falsePositives = 0;
    unsigned int falseNegatives = 0;
    vector<Point> foundDetection;
    // Walk over positive training samples, generate images and detect
    for (vector<string>::const_iterator posTrainingIterator = posFileNames.begin(); posTrainingIterator != posFileNames.end(); ++posTrainingIterator) {
        const Mat imageData = imread(*posTrainingIterator, IMREAD_GRAYSCALE);
        hog.detect(imageData, foundDetection, hitThreshold, winStride, trainingPadding);
        if (foundDetection.size() > 0) {
            ++truePositives;
            falseNegatives += foundDetection.size() - 1;
        } else {
            ++falseNegatives;
        }
    }
    // Walk over negative training samples, generate images and detect
    for (vector<string>::const_iterator negTrainingIterator = negFileNames.begin(); negTrainingIterator != negFileNames.end(); ++negTrainingIterator) {
        const Mat imageData = imread(*negTrainingIterator, IMREAD_GRAYSCALE);
        hog.detect(imageData, foundDetection, hitThreshold, winStride, trainingPadding);
        if (foundDetection.size() > 0) {
            falsePositives += foundDetection.size();
        } else {
            ++trueNegatives;
        }
    }

    printf("Results:\n\tTrue Positives: %u\n\tTrue Negatives: %u\n\tFalse Positives: %u\n\tFalse Negatives: %u\n", truePositives, trueNegatives, falsePositives, falseNegatives);
}

/**
 * Test detection with custom HOG description vector
 * @param hog
 * @param hitThreshold threshold value for detection
 * @param imageData
 */
static void detectTest(const HOGDescriptor& hog, const double hitThreshold, Mat& imageData) {
    vector<Rect> found;
    Size padding(Size(8, 8));
    Size winStride(Size(8, 8));
    hog.detectMultiScale(imageData, found, hitThreshold, winStride, padding);
    showDetections(found, imageData);
}

int main(int argc, char** argv ){
    HOGDescriptor hog; // Use standard parameters here
    hog.winSize = Size(96, 160); // Training images size

    static vector<string> positiveTrainingImages;
    static vector<string> negativeTrainingImages;
    static vector<string> validExtensions;
    validExtensions.push_back("jpg");
    validExtensions.push_back("png");
    validExtensions.push_back("ppm");

    getFilesInDirectory(posSamplesDir, positiveTrainingImages, validExtensions);
    getFilesInDirectory(negSamplesDir, negativeTrainingImages, validExtensions);
    unsigned long overallSamples = positiveTrainingImages.size() + negativeTrainingImages.size();
    //cout << overallSamples << endl;

    // <editor-fold defaultstate="collapsed" desc="Calculate HOG features and save to file">
    // Make sure there are actually samples to train
    if (overallSamples == 0) {
        printf("No training sample files found, nothing to do!\n");
        return EXIT_SUCCESS;
    }

    /// @WARNING: This is really important, some libraries (e.g. ROS) seems to set the system locale which takes decimal commata instead of points which causes the file input parsing to fail
    setlocale(LC_ALL, "C"); // Do not use the system locale
    setlocale(LC_NUMERIC,"C");
    setlocale(LC_ALL, "POSIX");

    printf("Reading files, generating HOG features and save them to file '%s':\n", featuresFile.c_str());
    float percent;

    fstream File;
    File.open(featuresFile.c_str(), ios::out);
    if (File.good() && File.is_open()) {
        #if TRAINHOG_USEDSVM == SVMLIGHT
            File << "# Use this file to train, e.g. SVMlight by issuing $ svm_learn -i 1 -a weights.txt " << featuresFile.c_str() << endl;
        #endif
        // Iterate over sample images
        for (unsigned long currentFile = 0; currentFile < overallSamples; ++currentFile) {
            storeCursor();
            vector<float> featureVector;
            // Get positive or negative sample image file path
            const string currentImageFile = (currentFile < positiveTrainingImages.size() ? positiveTrainingImages.at(currentFile) : negativeTrainingImages.at(currentFile - positiveTrainingImages.size()));
            // Output progress
            if ( (currentFile+1) % 10 == 0 || (currentFile+1) == overallSamples ) {
                percent = ((currentFile+1) * 100 / overallSamples);
                printf("%5lu (%3.0f%%):\tFile '%s'", (currentFile+1), percent, currentImageFile.c_str());
                fflush(stdout);
                resetCursor();
            }
            // Calculate feature vector from current image file
            calculateFeaturesFromInput(currentImageFile, featureVector, hog);
            if (!featureVector.empty()) {
                /* Put positive or negative sample class to file,
                 * true=positive, false=negative,
                 * and convert positive class to +1 and negative class to -1 for SVMlight
                 */
                File << ((currentFile < positiveTrainingImages.size()) ? "+1" : "-1");
                // Save feature vector components
                for (unsigned int feature = 0; feature < featureVector.size(); ++feature) {
                    File << " " << (feature + 1) << ":" << featureVector.at(feature);
                }
                File << endl;
            }
        }
        printf("\n");
        File.flush();
        File.close();
    } else {
        printf("Error opening file '%s'!\n", featuresFile.c_str());
        return EXIT_FAILURE;
    }

    /// Read in and train the calculated feature vectors
    printf("Calling %s\n", TRAINHOG_SVM_TO_TRAIN::getInstance()->getSVMName());
    //TRAINHOG_SVM_TO_TRAIN::getInstance()->read_problem(const_cast<char*> (featuresFile.c_str()));
    //TRAINHOG_SVM_TO_TRAIN::getInstance()->train(); // Call the core libsvm training procedure
    //printf("Training done, saving model file!\n");
    //TRAINHOG_SVM_TO_TRAIN::getInstance()->saveModelToFile(svmModelFile);
}
