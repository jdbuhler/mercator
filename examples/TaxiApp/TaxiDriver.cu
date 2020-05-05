//
// AGGENUMDRIVER.CU
// Aggregation and Enumeration functionality test
//
// MERCATOR
// Copyright (C) 2018 Washington University in St. Louis; all rights reserved.
//

#include <iostream>
#include <cstdlib>

#include <algorithm>
//#include <chrono>

#include <vector>
#include <fstream>
#include <string>

#include "Taxi.cuh"

#define DEBUG_INFO 1

using namespace std;

int main(int argc, char** argv)
{
  //const unsigned int NVALUES = 1000000000; // one BEEEELLION values
  //const unsigned int NVALUES = 4000; // one BEEEELLION values

  ////const unsigned int NVALUES = 800000; // one BEEEELLION values

  ////const unsigned int NVALUES = 1000000; // CURRENT BREAKING TEST
  //const unsigned int NVALUES = 74000; // one BEEEELLION values
  //const unsigned int NVALUES = 7400; // one BEEEELLION values
  //const unsigned int NVALUES = 256; // one BEEEELLION values
  //const unsigned int NVALUES = 220000; // one BEEEELLION values
  //const unsigned int NVALUES = 10000000; // one BEEEELLION values
  //const unsigned int NVALUES = 35840; // one BEEEELLION values
  //const unsigned int NVALUES = 143360; // one BEEEELLION values
  //const unsigned int NVALUES = 107520; // one BEEEELLION values

  //const unsigned int NVALUES = 71934;
  //const unsigned int NVALUES = 71935;

  //const unsigned int NVALUES = 129024;
  //const unsigned int NVALUES = 129025;

  //const unsigned int NVALUES = 1000;
  //const unsigned int NVALUES = 100;
  //const unsigned int NVALUES = 514;
  //const unsigned int NVALUES = 20;
  //const unsigned int NVALUES = 640;


  //const unsigned int NVALUES = 999;
  //const unsigned int NVALUES = 4096;

  //const unsigned int MULTIPLIER = 1024;
  //const unsigned int MULTIPLIER = 256;
  
  ///////////////
  //EVEN TESTS 1B
  //const unsigned int NVALUES = 32000000;
  //const unsigned int NVALUES = 8000000;
  //const unsigned int NVALUES = 2000000;
  //const unsigned int NVALUES = 2500000;
  //const unsigned int NVALUES = 5000000;
  const unsigned int NVALUES = 100000000;
  
  //const unsigned int MULTIPLIER = 32;
  //const unsigned int MULTIPLIER = 128;
  //const unsigned int MULTIPLIER = 512;
  ////const unsigned int MULTIPLIER = 2048;
  //const unsigned int MULTIPLIER = 1024;


  ///////////////
  //ODD TESTS 1B
  //const unsigned int NVALUES = 16000000;
  //const unsigned int NVALUES = 5333333;
  //const unsigned int NVALUES = 1333333;
  //const unsigned int NVALUES = 1000000;
  
  //const unsigned int MULTIPLIER = 64;
  //const unsigned int MULTIPLIER = 192;
  //const unsigned int MULTIPLIER = 768;
  //const unsigned int MULTIPLIER = 1024;


  ///////////////
  //ODD TESTS2 1B
  //const unsigned int NVALUES = 7937985;
  //const unsigned int NVALUES = 1996101;
  //const unsigned int NVALUES = 999024;
  //const unsigned int NVALUES = 499756;
  
  //const unsigned int MULTIPLIER = 129;
  //const unsigned int MULTIPLIER = 513;
  //const unsigned int MULTIPLIER = 1025;
  //const unsigned int MULTIPLIER = 2049;



  //cout << "HERE1" << endl;
  srand(0);
  //cout << "HERE2" << endl;

  //string fname = "test_taxi.csv";

  string fname = argv[1];

  #if DEBUG_INFO
  cout << "Input Name: " << argv[1] << endl;
  cout << "File Name:  " << fname << endl;
  #endif

  string line;
  //vector<string> lines = vector<string>();
  vector<unsigned int> lineLens = vector<unsigned int>();
  ifstream file(fname);

  file.seekg(0, std::ios::end);
  size_t length = file.tellg();
  file.seekg(0, std::ios::beg);
  #if DEBUG_INFO
  cout << "File length: " << length << endl;
  #endif

  //unsigned int longest = 0;
  //unsigned int shortest = UINT_MAX;

  char* h_string = new char[length];
  //char h_string[totalLen];
  file.seekg(0, std::ios::beg);
  file.read(h_string, length);
  //file.close();
  #if DEBUG_INFO
  cout << "gcount: " << file.gcount() << endl;
  #endif
  //cout << "h_string:" << endl << h_string << endl;
  file.seekg(0, std::ios::beg);
  
  bool firstIter = true;
  #if DEBUG_INFO
  unsigned int len = 0;
  vector<string> lines = vector<string>();
  #endif
  if(file.is_open()) {
	while(getline(file, line)) {
		//cout << line << endl << "LENGTH: " << line.length() << endl;
		if(firstIter) {
			firstIter = false;
		}
		else {
			lineLens.push_back(line.length() + 1);
		}
		#if DEBUG_INFO
		lines.push_back(line);
		++len;
		cout << line.substr(0, 100) << endl;
		#endif
			//lines.push_back(line);
		//if(line.length() > longest)
		//	longest = line.length();
		//if(line.length() < shortest)
		//	shortest = line.length();
	}
	//size_t leny = file.tellg();
	//cout << "File Length 2: " << leny << endl;
	//file.close();
  }
  #if DEBUG_INFO
  cout << "Done Reading. . ." << endl;
  cout << "Total Lines: " << lineLens.size() << endl;
  cout << "Len: " << len << endl;
  unsigned int totalBrackets = 0;
  unsigned int totalRealBrackets = 0;
  for(unsigned int i = 0; i < lines.size(); ++i) {
	for(unsigned int j = 0; j < lines.at(i).length(); ++j) {
		if(lines.at(i).at(j) == '[') {
			totalBrackets++;
			if(lines.at(i).at(j + 1) != '[') {
				totalRealBrackets++;
			}
		}
	}
  }
  cout << "Total Brackets: " << totalBrackets << "\t\tTotal Real Brackets: " << totalRealBrackets << "\t\tPercentage of Characters: " << double(totalBrackets) / double(length) << endl;

  unsigned int currentBrackets = 0;
  vector<double> bracketRatios = vector<double>();

  for(unsigned int i = 0; i < lines.size(); ++i) {
	for(unsigned int j = 0; j < lines.at(i).length(); ++j) {
		if(lines.at(i).at(j) == '[')
			++currentBrackets;
	}
	bracketRatios.push_back(double(currentBrackets) / double(lines.at(i).length()));
	currentBrackets = 0;
  }

  double totalRatios = 0.0;
  double maxRatio = 0.0;
  double minRatio = 2.0;
  for(unsigned int i = 0; i < bracketRatios.size(); ++i) {
	totalRatios += bracketRatios.at(i);
	if(bracketRatios.at(i) < minRatio)
		minRatio = bracketRatios.at(i);
	if(bracketRatios.at(i) > maxRatio)
		maxRatio = bracketRatios.at(i);
  }

  cout << "Average Bracket Ratio per Line: " << totalRatios / bracketRatios.size() << endl;
  cout << "Max Bracket Ratio per Line: " << maxRatio << endl;
  cout << "Min Bracket Ratio per Line: " << minRatio << endl;

  #endif
  
  //cout << "Longest = " << longest << "\t\tShortest = " << shortest << endl;

  //string allLines = "";
  //for(unsigned int i = 0; i < lines.size(); ++i) {
	//allLines = allLines + lines.at(i);
  //}
  #if DEBUG_INFO
  cout << "Done combining lines. . ." << endl;
  #endif
  //unsigned int totalLen = allLines.length();
  unsigned int totalLen = length;
  #if DEBUG_INFO
  cout << "All Lines Length: " << totalLen << endl;
  #endif
/*
  //const char* h_string = allLines.c_str();
  char* h_string = new char[totalLen];
  //char h_string[totalLen];
  file.seekg(0, std::ios::beg);
  file.read(h_string, length);
  //file.close();
  cout << "gcount: " << file.gcount() << endl;
  cout << "h_string:" << endl << h_string << endl;
*/
  char* d_string;

  cudaMalloc(&d_string, totalLen * sizeof(char));
  cudaMemcpy(d_string, h_string, totalLen * sizeof(char), cudaMemcpyHostToDevice);


  //Line* inputValues = new Line[lines.size()];
  Line* inputValues = new Line[lineLens.size()];
  #if DEBUG_INFO
  cout << "Created inputValues pointer" << endl;
  #endif
  //unsigned int prevLen = 0;
  unsigned int prevLen = 109;	//Known starting position of line after header
  for(unsigned int i = 0; i < lineLens.size(); ++i) {
	inputValues[i] = Line(i + 1, d_string + prevLen, lineLens.at(i));
	#if DEBUG_INFO
	cout << "LINE " << i + 1 << ": " << lineLens.at(i) << endl;
	#endif
	prevLen += lineLens.at(i);
  }
  //for(unsigned int i = 0; i < lines.size(); ++i) {
	//inputValues[i] = Line(i, d_string + prevLen, lines.at(i).length());
	//prevLen = lines.at(i).length();
  //}
  #if DEBUG_INFO
  cout << "Done creating lines. . ." << endl;
  #endif


  //unsigned int* d_compositeStorage;
  //cudaMalloc(&d_compositeStorage, NVALUES * MULTIPLIER * sizeof(unsigned int));
  //cout << "AFTER MALLOC... ui " << sizeof(unsigned int) << " d " << sizeof(double) << " c " << sizeof(Composite) << endl;

  //double CPUverification[NVALUES];
  //double* CPUverification = new double[NVALUES];

  //unsigned int h_compositeStorage[NVALUES * MULTIPLIER];
  //unsigned int* h_compositeStorage = new unsigned int[NVALUES * MULTIPLIER];
  //Composite *inputValues = new Composite [NVALUES];
  unsigned int total = NVALUES;
  //Composite inputValues[NVALUES];
  /*
  for(unsigned int z = 0; z < NVALUES * MULTIPLIER; ++z) {
	//h_compositeStorage[z] = z % MULTIPLIER;
	h_compositeStorage[z] = z;
	if(z % MULTIPLIER == 0) {
  		//cout << "Created Composite " << z / MULTIPLIER << "..." << endl;
		unsigned int crand = rand() % MULTIPLIER;
		inputValues[z / MULTIPLIER] = Composite(d_compositeStorage + z, crand);
		total += crand;
  		cout << "Created Composite " << z / MULTIPLIER << "...\tlength = " << crand << endl;
		//inputValues[z] = Composite(z, rand() % MULTIPLIER);
  		//cout << "Created Composite " << z / MULTIPLIER << "...\t" << inputValues[z].startPointer << "\t" << inputValues[z].length << endl;
		//inputValues[z] = Composite();

		//unsigned int sum = 0;
		//for(unsigned int y = 0; y < crand; ++y) {
		//	sum += h_compositeStorage[z - crand + y];
		//}
		//CPUverification[z / MULTIPLIER] = sum / crand;
	}
	if(z % MULTIPLIER == 0 && z != 0) {
		unsigned int sum = 0;
		for(unsigned int y = 0; y < inputValues[z / MULTIPLIER].length; ++y) {
			sum += h_compositeStorage[z - inputValues[z / MULTIPLIER].length + y];
		}
		CPUverification[z / MULTIPLIER] = sum / inputValues[z / MULTIPLIER].length;
	}
  }
  */

  //std::normal_distribution<double> distribution(512, 300);

  //unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  //std::default_random_engine generator(seed);

  /*for(unsigned int z = 0; z < NVALUES * MULTIPLIER; ++z) {
	h_compositeStorage[z] = rand() % MULTIPLIER;
	//h_compositeStorage[z] = z % MULTIPLIER;
	//h_compositeStorage[z] = 1;
  }

  for(unsigned int z = 0; z < NVALUES; ++z) {
	unsigned int crand = rand() % MULTIPLIER;
	inputValues[z] = Composite(d_compositeStorage + z * MULTIPLIER, crand);
	CPUverification[z] = 0;
	//cout << distribution(generator) << endl;
	for(unsigned int y = 0; y < crand; ++y) {
		CPUverification[z] += h_compositeStorage[z * MULTIPLIER + y];
	}
	//CPUverification[z] /= crand;
  }

  //char c;
  //cin >> c;

  cout << "BEFORE COPY..." << endl;
  cudaMemcpy(d_compositeStorage, h_compositeStorage, NVALUES * MULTIPLIER * sizeof(unsigned int), cudaMemcpyHostToDevice);
  cout << "AFTER COPY..." << endl;
  */  

  //double *outputValues = new double [NVALUES];
  Position *outputValues = new Position [NVALUES];
  //cout << "HERE3" << endl;
  
  //unsigned int total = 0;
  //for (unsigned int j = 0; j < NVALUES; j++) {
    //inputValues[j] = rand();
    //inputValues[j] = j % 4;
    //total += j % 4;
  //}
  //cout << "HERE4" << endl;
  
  // begin MERCATOR usage
  
  ////cout << "BEFORE BUFFER ALLOCATION..." << endl;
  //Mercator::Buffer<Composite> inputBuffer(NVALUES);
  //Mercator::Buffer<unsigned int> outputBufferAccept(NVALUES);
  //Mercator::Buffer<double> outputBufferAccept(NVALUES);
  //cout << "HERE5" << endl;

  //Mercator::Buffer<Line> inputBuffer(lines.size());
  Mercator::Buffer<Line> inputBuffer(lineLens.size());
  Mercator::Buffer<Position> outputBufferAccept(NVALUES);
  ////cout << "AFTER BUFFER ALLOCATION..." << endl;
  //int x;
  //cin >> x; 
  Taxi efapp;
  
  //cout << "HERE6" << endl;
  ////cout << "BEFORE SET SOURCE SINK..." << endl;
  efapp.src.setSource(inputBuffer);
  efapp.snk.setSink(outputBufferAccept);
  ////cout << "AFTER SET SOURCE SINK..." << endl;
  //cout << "HERE7" << endl;
  
  // move data into the input buffer
  ////cout << "BEFORE SET INPUT BUFFER" << endl;
  //inputBuffer.set(inputValues, lines.size());
  inputBuffer.set(inputValues, lineLens.size());
  ////cout << "AFTER SET INPUT BUFFER" << endl;
  //cout << "HERE8" << endl;
  
  cout << "RUNNING APP. . . " << endl;
  ////unsigned int max = UINT_MAX * 32;
  ////cout << UINT_MAX << "\t\t" << max << endl;
  efapp.run();
  cout << "APP FINISHED. . ." << endl;
  
  // get data out of the output buffer
  unsigned int outSize = outputBufferAccept.size();
  outputBufferAccept.get(outputValues, outSize);
  
  // end MERCATOR usage
  
  cout << "# outputs = " << outSize << endl;
  cout << "# expected outputs = " << total << endl;

  //for(unsigned int j = 0; j < NVALUES*MULTIPLIER; j+=1)
	//cout << "out[" << j << "]:\t" << outputValues[j] << endl;

  /*
  std::vector<double> outs;
  std::vector<double> ins;

  for(unsigned int i = 0; i < NVALUES; ++i) {
	outs.push_back(outputValues[i]);
	ins.push_back(CPUverification[i]);
  }

  std::sort(outs.begin(), outs.end());
  std::sort(ins.begin(), ins.end());

  for(unsigned int i = 0; i < NVALUES; ++i) {
	if(outs[i] != ins[i]) {
		cout << "GPU[" << i << "] = " << outs[i] << "\t\tCPU[" << i << "] = " << ins[i] << "\t\tlength: " << inputValues[i].length << "\t\tdiff: " << outs[i] - ins[i] << endl;
	}
  }
  */

  /*
  bool invalid = false;
  bool valid[NVALUES];
  bool used[NVALUES];
  for(unsigned int i = 0; i < NVALUES; ++i) {
	valid[i] = false;
	used[i] = false;
  }

  for(unsigned int j = 0; j < NVALUES; ++j) {
	for(unsigned int k = 0; k < NVALUES; ++k) {
		if(!used[k]) {
		if(outputValues[j] == CPUverification[k]) {
			valid[k] = true;
			used[k] = true;
		}
		//if(outputValues[j] != CPUverification[k]) {
			//cout << "GPU[" << j << "] = " << outputValues[j] << "\t\tCPU[" << j << "] = " << CPUverification[j] << "\t\tlength: " << inputValues[j].length << "\t\tdiff: " << outputValues[j] - CPUverification[j] << endl;
			//invalid = true;
		//}
		//else {
		//	valid[j] = true;
		//	used[j] = true;
		//	cout << "GPU[" << j << "] = " << outputValues[j] << "\t\tCPU[" << j << "] = " << CPUverification[j] << "\t\tlength: " << inputValues[j].length << "\t\tdiff: " << outputValues[j] - CPUverification[j] << endl;
		//}
		}
  	}
  }

  for(unsigned int i = 0; i < NVALUES; ++i) {
	if(valid[i] == false)
		invalid = true;
  }

  if(invalid) {
	cout << "*** SOME OUTPUTS ARE INCORRECT ***" << endl;
  }
  else {
	cout << "*** OUTPUTS VALIDATED CORRECT ***" << endl;
  }
  */

/*
  vector<unsigned int> outPos = vector<unsigned int>();

  for(unsigned int i = 0; i < outSize; ++i) {
	if(i == 0) {
		
	}
	outPos.push_back(0);
  }

  for(unsigned int i = 0; i < outSize; ++i) {
	//if(outputValues[i].tag != 0 || i == 0)
	//cout << outputValues[i].tag << ", " << outputValues[i].longitude << ", " << outputValues[i].latitude << endl;
	outPos.at(outputValues[i].tag) += 1;
  }

  unsigned int longestPos = 0;
  unsigned int shortestPos = UINT_MAX;

  unsigned int longestIdx = 0;
  unsigned int shortestIdx = 0;

  for(unsigned int i = 0; i < outSize; ++i) {
	if(outPos.at(i) > longestPos) {
		longestPos = outPos.at(i);
		longestIdx = i;
	}
	if(outPos.at(i) < shortestPos) {
		shortestPos = outPos.at(i);
		shortestIdx = i;
	}
  }

  cout << "Shortest = " << shortestPos << "\t\tLongest = " << longestPos << endl;

*/

  #if DEBUG_INFO
  for(unsigned int i = 0; i < outSize; ++i) {
	cout << outputValues[i].tag << ", " << outputValues[i].longitude << ", " << outputValues[i].latitude << endl;
  }
  #endif

  #if DEBUG_INFO
  vector<unsigned int> pairCount = vector<unsigned int>();
  //len = 500;
  for(unsigned int i = 0; i < len; ++i) {
	pairCount.push_back(0);
  }
  for(unsigned int i = 0; i < outSize; ++i) {
	pairCount.at(outputValues[i].tag) = pairCount.at(outputValues[i].tag) + 1;
  }

  unsigned int pairCountMax = 0;
  unsigned int pairCountMin = UINT_MAX;
  unsigned int pairCountMaxTag = 0;
  unsigned int pairCountMinTag = 0;
  unsigned int pairCountTotal = 0;
  for(unsigned int i = 1; i < len; ++i) {
	if(pairCount.at(i) > pairCountMax) {
		pairCountMax = pairCount.at(i);
		pairCountMaxTag = i;
	}
	if(pairCount.at(i) < pairCountMin) {
		pairCountMin = pairCount.at(i);
		pairCountMinTag = i;
	}

	pairCountTotal += pairCount.at(i);
  }

  unsigned int maxLineLength = 0;
  unsigned int minLineLength = UINT_MAX;
  unsigned int totalLineLength = 0;
  for(unsigned int i = 0; i < lineLens.size(); ++i) {
	if(lineLens.at(i) > maxLineLength) {
		maxLineLength = lineLens.at(i);
		//pairCountMaxTag = i;
	}
	if(lineLens.at(i) < minLineLength) {
		minLineLength = lineLens.at(i);
		//pairCountMinTag = i;
	}

	totalLineLength += lineLens.at(i);
	//pairCountTotal += pairCount.at(i);
  }


  cout << "MIN PAIR: " << pairCountMin << "\t\tTAG: " << pairCountMinTag << endl;
  cout << "MAX PAIR: " << pairCountMax << "\t\tTAG: " << pairCountMaxTag << endl;
  cout << "MAX LENGTH: " << maxLineLength << "\t\tMIN LENGTH: " << minLineLength << "\t\tAVG LENGTH: " << double(totalLineLength) / double(lineLens.size()) << endl;
  cout << "TOTAL PAIR: " << pairCountTotal << "\t\tAVG PAIR: " << double(pairCountTotal) / double(len) << endl;
  #endif

  //delete [] inputValues;
  //delete [] outputValues;

  //cudaFree(&d_compositeStorage);
  
  return 0;
}
