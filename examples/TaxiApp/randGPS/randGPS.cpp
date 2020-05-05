/*
 * randGPS.cpp
 * 
 * A random input generator for the TaxiApp line of applications
 *
 * MERCATOR
 * Copyright (C) 2018 Washington University in St. Louis; all rights reserved.
 * 
 * Application usage:
 *
 * app numEntries maxLength
 *
 * numEntries = The number of taxi entires to be made
 * maxLength = The maximum length of each taxi entry (number of coordinate pairs), must be > 1
 */

#include <iostream>
#include <fstream>
#include <string>
#include <unordered_map>
#include <time.h>

using namespace std;

int main(int argc, char* argv[]) {
	int numEntries = atoi(argv[1]);
	srand(time(NULL));
	int maxEntries = atoi(argv[2]);
	double r = (double)rand();
	string header = "\"TRIP_ID\",\"CALL_TYPE\",\"ORIGIN_CALL\",\"ORIGIN_STAND\",\"TAXI_ID\",\"TIMESTAMP\",\"DAY_TYPE\",\"MISSING_DATA\",\"POLYLINE\"\n";
	ofstream file;
	string fileName = "rand" + to_string(numEntries);
	fileName = fileName + "_";
	fileName = fileName + to_string(maxEntries);
	fileName = fileName + ".csv";

	file.open(fileName);
	file << header;

	string tmp = "";
	int z = 0;
	for(int i = 0; i < numEntries; ++i) {
		file << "\"T" << i << "\",";
		z = rand() % 3;
		tmp = (z == 0 ? "\"A\"," : (z == 1 ? "\"B\"," : "\"C\","));
		file << tmp;
		if(z == 0) {
			tmp = "" + to_string(rand() % 97999 + 2000);
			tmp = tmp + ",NA,";
		}
		else if(z == 1) {
			tmp = "NA," + to_string(rand() % 60 + 1);
			tmp = tmp + ",";
		}
		else {
			tmp = "NA,NA,";
		}

		file << tmp;

		file << "" << to_string(rand() % 1000 + 20000000) << ",";
		file << "" << to_string(time(NULL) + (rand() % 2 == 0 ? (rand() % 2000) : (rand() % 2000 * -1))) << ",";
		file << "\"A\",";
		file << "\"False\",\"[";
		
		for(int j = rand() % (maxEntries - 1); j < maxEntries; ++j) {
			file << "[";
			r = (double)rand() / RAND_MAX;
			r += rand() % 90;
			if(rand() % 2 == 1)
				r *= -1;
			//cout << r << endl;
			file << to_string(r) << ", ";

			r = (double)rand() / RAND_MAX;
			r += rand() % 180;
			if(rand() % 2 == 1)
				r *= -1;
			//cout << r << endl;
			file << to_string(r) << (j == maxEntries - 1 ? "]]\"\n" : "], ");
		}
	}
	return 0;
}
