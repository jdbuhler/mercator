//
// AGGENUMDRIVER.CU
// Aggregation and Enumeration functionality test
//
// MERCATOR
// Copyright (C) 2018 Washington University in St. Louis; all rights reserved.
//

#include <iostream>
#include <cstdlib>

#include "AggEnum.cuh"

using namespace std;

int main()
{
 // const unsigned int NVALUES = 1000000000; // one BEEEELLION values
  //const unsigned int NVALUES = 1000000;	//DID MAIN TEST WITH THIS ONE <---------------------------------
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
  //const unsigned int NVALUES = 514;
  //const unsigned int NVALUES = 20;
  //const unsigned int NVALUES = 10;

  //EVEN TESTS 2B
  //const unsigned int NVALUES = 64000000;
  //const unsigned int NVALUES = 16000000;
  //const unsigned int NVALUES = 4000000;
  //const unsigned int NVALUES = 1000000;
  
  //const unsigned int MULTIPLIER = 32;
  //const unsigned int MULTIPLIER = 128;
  //const unsigned int MULTIPLIER = 512;
  //const unsigned int MULTIPLIER = 2048;
  
  //ODD TESTS 2B
  //const unsigned int NVALUES = 15875969;
  //const unsigned int NVALUES = 10666667;


  ///////////////
  //EVEN TESTS 1B
  //const unsigned int NVALUES = 32000000;
  //const unsigned int NVALUES = 8000000;
  //const unsigned int NVALUES = 2000000;
  //const unsigned int NVALUES = 500000;
  
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
  
  //const unsigned int MULTIPLIER = 64;
  //const unsigned int MULTIPLIER = 192;
  //const unsigned int MULTIPLIER = 768;
  //const unsigned int MULTIPLIER = 1024;


  ///////////////
  //ODD TESTS2 1B
  //const unsigned int NVALUES = 7937985;
  //const unsigned int NVALUES = 1996101;
  //const unsigned int NVALUES = 999024;
  const unsigned int NVALUES = 499756;
  
  //const unsigned int MULTIPLIER = 129;
  //const unsigned int MULTIPLIER = 513;
  //const unsigned int MULTIPLIER = 1025;
  const unsigned int MULTIPLIER = 2049;


  //const unsigned int NVALUES = 500000;
  //const unsigned int MULTIPLIER = 2048;


  //const unsigned int NVALUES = 32000000;
  //const unsigned int MULTIPLIER = 32;

  //cout << "HERE1" << endl;
  srand(0);
  //cout << "HERE2" << endl;
  
  unsigned int *inputValues = new unsigned int [NVALUES];
  unsigned int *outputValues = new unsigned int [NVALUES*MULTIPLIER];
  //cout << "HERE3" << endl;
  
  unsigned int total = 0;
  for (unsigned int j = 0; j < NVALUES; j++) {
    //inputValues[j] = rand();

    //ALT TEST
    inputValues[j] = MULTIPLIER;
    total += MULTIPLIER;


    //MAIN TEST
    //inputValues[j] = j % (MULTIPLIER + 1);
    //total += j % (MULTIPLIER + 1);
  }
  //cout << "HERE4" << endl;
  
  // begin MERCATOR usage
  
  Mercator::Buffer<unsigned int> inputBuffer(NVALUES);
  //Mercator::Buffer<unsigned int> outputBufferAccept(NVALUES);
  Mercator::Buffer<unsigned int> outputBufferAccept(NVALUES*MULTIPLIER);
  //cout << "HERE5" << endl;
 
  //int x;
  //cin >> x; 
  AggEnum efapp;
  
  //cout << "HERE6" << endl;
  efapp.src.setSource(inputBuffer);
  efapp.snk.setSink(outputBufferAccept);
  //cout << "HERE7" << endl;
  
  // move data into the input buffer
  inputBuffer.set(inputValues, NVALUES);
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
  
  delete [] inputValues;
  delete [] outputValues;
  
  return 0;
}
