//
// EFDRIVER.CU
// Even-valued filter test
//
// MERCATOR
// Copyright (C) 2018 Washington University in St. Louis; all rights reserved.
//

#include <iostream>
#include <cstdlib>

#include "EvenFilter.cuh"

using namespace std;

int main()
{
  const unsigned int NVALUES = 1000000000; // one BEEEELLION values
  //const unsigned int NVALUES = 10000000; // one BEEEELLION values
  //const unsigned int NVALUES = 100000; // one BEEEELLION values

  srand(0);
  
  unsigned int *inputValues = new unsigned int [NVALUES];
  unsigned int *outputValues = new unsigned int [NVALUES];
  
  for (unsigned int j = 0; j < NVALUES; j++)
    inputValues[j] = rand();
  
  // begin MERCATOR usage
  
  Mercator::Buffer<unsigned int> inputBuffer(NVALUES);
  Mercator::Buffer<unsigned int> outputBufferAccept(NVALUES);
  
  EvenFilter efapp;
  
  efapp.src.setSource(inputBuffer);
  efapp.snk.setSink(outputBufferAccept);
  
  // move data into the input buffer
  inputBuffer.set(inputValues, NVALUES);
  
  efapp.run();
  
  // get data out of the output buffer
  unsigned int outSize = outputBufferAccept.size();
  outputBufferAccept.get(outputValues, outSize);
  
  // end MERCATOR usage
  
  cout << "# outputs = " << outSize << endl;

/*
  for(unsigned int i=0;i<outSize; i++){
  cout  << outputValues[i]<< endl;
  }
*/
  delete [] inputValues;
  delete [] outputValues;
  
  return 0;
}
