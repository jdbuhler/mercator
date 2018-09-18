//
// MFDRIVER.CU
// Multiple module filter test
//
// MERCATOR
// Copyright (C) 2018 Washington University in St. Louis; all rights reserved.
//

#include <iostream>
#include <cstdlib>

#include "ModFilter.cuh"

using namespace std;

int main()
{
  const unsigned int NVALUES = 1000000000; // one BEEEELLION values
  
  srand(0);
  
  unsigned int *inputValues = new unsigned int [NVALUES];
  unsigned int *outputValues = new unsigned int [NVALUES];
  
  for (unsigned int j = 0; j < NVALUES; j++)
    inputValues[j] = rand();
  
  // begin MERCATOR usage
  
  Mercator::Buffer<unsigned int> inputBuffer(NVALUES);
  Mercator::Buffer<unsigned int> outputBufferAccept(NVALUES);
  
  ModFilter mfapp;
  
  mfapp.src.setSource(inputBuffer);
  mfapp.snk.setSink(outputBufferAccept);
  
  mfapp.f1.getParams()->modulus = 2;
  mfapp.f2.getParams()->modulus = 3;
  mfapp.f3.getParams()->modulus = 5;
  
  // move data into the input buffer
  inputBuffer.set(inputValues, NVALUES);
  
  mfapp.run();
  
  // get data out of the output buffer
  unsigned int outSize = outputBufferAccept.size();
  outputBufferAccept.get(outputValues, outSize);
  
  // end MERCATOR usage
  
  cout << "# outputs = " << outSize << endl;
  
  delete [] inputValues;
  delete [] outputValues;
  
  return 0;
}
