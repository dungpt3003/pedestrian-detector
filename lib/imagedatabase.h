#ifndef IMAGEDATABASE_H
#define IMAGEDATABASE_H

#include "common.h"
#include <iostream>
using namespace std;

// Stores the datasets used for training and testing the Support
// Vector Machine class. Contains basically a list of image filenames
// together with their true or predicted labels.
class
ImageDatabase{
private:
	std::vector<std::string> _filenames;
	std::vector<float> _labels;
	int _positivesCount;
	int _negativesCount;
	std::string _dbFilename;

public:
	// Create a new database.
	ImageDatabase():
	_negativesCount(0), _positivesCount(0){
	}

	//ImageDatabase(const char* dbFilename);
	ImageDatabase(const char* dbFilename):
	_negativesCount(0), _positivesCount(0){
		load(dbFilename);
	}

	ImageDatabase(const vector<float>& labels, const vector<string>& filenames):
	_negativesCount(0), _positivesCount(0){
		_labels = labels;
		_filenames = filenames;

		for(vector<float>::iterator i = _labels.begin(); i != _labels.end(); i++){
			if(*i > 0) _positivesCount++;
			else if(*i < 0) _negativesCount++;
		}
	}
	// Load a database from file.
	void load(const char *dbFilename){
		_dbFilename = string(dbFilename);

		_negativesCount = 0;
		_positivesCount = 0;

		ifstream f(dbFilename);
		if(!f.is_open()){
			printf("Could not open file %s for reading", dbFilename);
		}

		int nItems;
		f >> nItems;

		assert(nItems > 0);
		_labels.resize(nItems);
		_filenames.resize(nItems);

		for(int i = 0; i < nItems; i++){
			f >> _labels[i] >> _filenames[i];

			if(_labels[i] < 0) _negativesCount++;
			else if(_labels[i] > 0) _positivesCount++;
		}
	}

	// Save database to a file
	void save(const char* dbFilename){
		ofstream f(dbFilename);
		if(!f.is_open()) {
			printf("Could not open file %s for writing", dbFilename);
		}

		f << _labels.size() << "\n";
		for(int i = 0; i < _labels.size(); i++){
			f << _labels[i] << " " << _filenames[i] << "\n";
		}
	}

	// Accessors
	const int getLabel(int idx) const { return _labels[idx]; }
	const std::vector<float>& getLabels() const { return _labels; }
	const std::string getFilename(int idx) const { return _filenames[idx]; }
	const std::vector<std::string>& getFilenames() const { return _filenames; }

	// Info about the database
	int getPositivesCount() const { return _positivesCount; }
	int getNegativesCount() const { return _negativesCount; }
	int getUnlabeledCount() const { return _labels.size() - _positivesCount - _negativesCount; }
	int getSize() const { return _labels.size(); }
	std::string getDatabaseFilename() const { return _dbFilename; }
};

// Prints information about the dataset
std::ostream&
operator<<(std::ostream& s, const ImageDatabase& db){
	s << "DATABASE INFO\n"
	  << setw(20) << "Original filename:" << " " << db.getDatabaseFilename() << "\n"
	  << setw(20) << "Positives:" << setw(5) << right << db.getPositivesCount() << "\n"
	  << setw(20) << "Negaties:"   << setw(5) << right << db.getNegativesCount() << "\n"
	  << setw(20) << "Unlabeled:"  << setw(5) << right << db.getUnlabeledCount() << "\n"
	  << setw(20) << "Total:"      << setw(5) << right << db.getSize() << "\n";

	return s;
}

#endif
