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

#include "Stat.cuh"

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
  //const unsigned int NVALUES = 500000;

  //Half  
  //const unsigned int NVALUES = 16000000;
  //const unsigned int NVALUES = 4000000;
  //const unsigned int NVALUES = 1000000;
  //const unsigned int NVALUES = 250000;

  //const unsigned int MULTIPLIER = 32;
  //const unsigned int MULTIPLIER = 128;
  //const unsigned int MULTIPLIER = 512;
  //const unsigned int MULTIPLIER = 2048;


  ///////////////
  //ODD TESTS 1B
  //const unsigned int NVALUES = 16000000;
  //const unsigned int NVALUES = 5333333;
  //const unsigned int NVALUES = 1333333;
  //const unsigned int NVALUES = 1000000;

  //Half  
  //const unsigned int NVALUES = 8000000;
  //const unsigned int NVALUES = 2666666;
  //const unsigned int NVALUES = 666666;
  //const unsigned int NVALUES = 500000;

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
  
  //Half
  //const unsigned int NVALUES = 3968992;
  //const unsigned int NVALUES = 998050;
  //const unsigned int NVALUES = 499512;
  //const unsigned int NVALUES = 249878;

  //const unsigned int MULTIPLIER = 129;
  //const unsigned int MULTIPLIER = 513;
  //const unsigned int MULTIPLIER = 1025;
  //const unsigned int MULTIPLIER = 2049;


  const unsigned int NVALUES = atoi(argv[1]);
  const unsigned int MULTIPLIER = atoi(argv[2]);

  //cout << "HERE1" << endl;
  srand(0);
  //cout << "HERE2" << endl;

  /*
  unsigned int* d_compositeStorage;
  cudaMalloc(&d_compositeStorage, NVALUES * MULTIPLIER * sizeof(unsigned int));
  ////cout << "AFTER MALLOC... ui " << sizeof(unsigned int) << " d " << sizeof(double) << " c " << sizeof(Composite) << endl;

  //double CPUverification[NVALUES];
  double* CPUverification = new double[NVALUES];

  //unsigned int h_compositeStorage[NVALUES * MULTIPLIER];
  unsigned int* h_compositeStorage = new unsigned int[NVALUES * MULTIPLIER];
  Composite *inputValues = new Composite [NVALUES];
  unsigned int total = NVALUES;
  //Composite inputValues[NVALUES];
  */

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

  unsigned int* h_compositeStorage = new unsigned int[NVALUES * MULTIPLIER];
  unsigned int* d_compositeStorage;
  cudaMalloc(&d_compositeStorage, NVALUES * MULTIPLIER * sizeof(unsigned int));
  for(unsigned int z = 0; z < NVALUES * MULTIPLIER; ++z) {
	h_compositeStorage[z] = rand() % MULTIPLIER;
	//h_compositeStorage[z] = z % MULTIPLIER;
	//h_compositeStorage[z] = 1;
  }

  /*
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
  */

  unsigned int totalLenRemaining = NVALUES * MULTIPLIER;
  vector<unsigned int> lineLengths = vector<unsigned int>();
  while(totalLenRemaining != 0) {
	unsigned int crand = rand() % MULTIPLIER;
	if(crand > totalLenRemaining) {
		lineLengths.push_back(totalLenRemaining);
		totalLenRemaining = 0;
	}
	else {
		lineLengths.push_back(crand);
		totalLenRemaining -= crand;
	}
  }
  unsigned int totalComposites = lineLengths.size();


  //unsigned int* d_compositeStorage;
  //cudaMalloc(&d_compositeStorage, NVALUES * MULTIPLIER * sizeof(unsigned int));
  ////cout << "AFTER MALLOC... ui " << sizeof(unsigned int) << " d " << sizeof(double) << " c " << sizeof(Composite) << endl;

  //double CPUverification[NVALUES];
  double* CPUverification = new double[totalComposites];

  //unsigned int h_compositeStorage[NVALUES * MULTIPLIER];
  //unsigned int* h_compositeStorage = new unsigned int[NVALUES * MULTIPLIER];
  Composite *inputValues = new Composite [totalComposites];
  unsigned int total = NVALUES;


  unsigned int prevLinePos = 0;
  for(unsigned int z = 0; z < totalComposites; ++z) {
	//unsigned int crand = rand() % MULTIPLIER;
	inputValues[z] = Composite(d_compositeStorage + prevLinePos, lineLengths.at(z));
	CPUverification[z] = 0;
	//cout << distribution(generator) << endl;
	for(unsigned int y = 0; y < lineLengths.at(z); ++y) {
		CPUverification[z] += h_compositeStorage[prevLinePos + y];
	}
	prevLinePos += lineLengths.at(z);
	//CPUverification[z] /= crand;
  }

  //char c;
  //cin >> c;

  //cout << "BEFORE COPY..." << endl;
  cudaMemcpy(d_compositeStorage, h_compositeStorage, NVALUES * MULTIPLIER * sizeof(unsigned int), cudaMemcpyHostToDevice);
  //cout << "AFTER COPY..." << endl;
  
  //double *outputValues = new double [NVALUES];
  double *outputValues = new double [totalComposites];
  //cout << "HERE3" << endl;
  
  //unsigned int total = 0;
  //for (unsigned int j = 0; j < NVALUES; j++) {
    //inputValues[j] = rand();
    //inputValues[j] = j % 4;
    //total += j % 4;
  //}
  //cout << "HERE4" << endl;
  
  // begin MERCATOR usage
  
  //cout << "BEFORE BUFFER ALLOCATION..." << endl;
  //Mercator::Buffer<Composite> inputBuffer(NVALUES);
  Mercator::Buffer<Composite> inputBuffer(totalComposites);
  //Mercator::Buffer<unsigned int> outputBufferAccept(NVALUES);
  //Mercator::Buffer<double> outputBufferAccept(NVALUES);
  Mercator::Buffer<double> outputBufferAccept(totalComposites);
  //cout << "HERE5" << endl;

  //cout << "AFTER BUFFER ALLOCATION..." << endl;
  //int x;
  //cin >> x; 
  Stat efapp;
  
  //cout << "HERE6" << endl;
  //cout << "BEFORE SET SOURCE SINK..." << endl;
  efapp.src.setSource(inputBuffer);
  efapp.snk.setSink(outputBufferAccept);
  //cout << "AFTER SET SOURCE SINK..." << endl;
  //cout << "HERE7" << endl;
  
  // move data into the input buffer
  //cout << "BEFORE SET INPUT BUFFER" << endl;
  //inputBuffer.set(inputValues, NVALUES);
  inputBuffer.set(inputValues, totalComposites);
  //cout << "AFTER SET INPUT BUFFER" << endl;
  //cout << "HERE8" << endl;
  
  cout << "RUNNING APP. . . " << endl;
  //unsigned int max = UINT_MAX * 32;
  //cout << UINT_MAX << "\t\t" << max << endl;
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

  //UNCOMMENT FOR VERIFICATION
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

  delete [] inputValues;
  delete [] outputValues;

  //cudaFree(&d_compositeStorage);
  
  return 0;
}
