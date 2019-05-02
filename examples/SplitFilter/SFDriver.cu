//
// SFDRIVER.CU
// Splitting filter test
//
// MERCATOR
// Copyright (C) 2018 Washington University in St. Louis; all rights reserved.
//

#include <iostream>
#include <cstdlib>

#include "SplitFilter.cuh"

using namespace std;

int main()
{
  const unsigned int NVALUES = 1000000; // one BEEEELLION values
  //const unsigned int NVALUES = 1000000000; // one BEEEELLION values
  //const unsigned int NVALUES = 500000; // half a BEEEELLION values
  
  srand(0);
  
  unsigned int *inputValues = new unsigned int [NVALUES];
  unsigned int *outputValuesA = new unsigned int [NVALUES];
  unsigned int *outputValuesR = new unsigned int [NVALUES];
  
  for (unsigned int j = 0; j < NVALUES; j++)
    inputValues[j] = rand();
  
  // begin MERCATOR usage
  
  Mercator::Buffer<unsigned int> inputBuffer(NVALUES);
  Mercator::Buffer<unsigned int> outputBufferAccept(NVALUES);
  Mercator::Buffer<unsigned int> outputBufferReject(NVALUES);
  
  SplitFilter sfapp;
  
  sfapp.src.setSource(inputBuffer);
  sfapp.snkAccept.setSink(outputBufferAccept);
  sfapp.snkReject.setSink(outputBufferReject);
  
  // move data into the input buffer
  inputBuffer.set(inputValues, NVALUES);
  
  sfapp.run();
  
  // get data out of the output buffers
  unsigned int outSizeA = outputBufferAccept.size();
  outputBufferAccept.get(outputValuesA, outSizeA);
  
  unsigned int outSizeR = outputBufferReject.size();
  outputBufferReject.get(outputValuesR, outSizeR);
  
  // end MERCATOR usage
  
  cout << "# outputs accepted = " << outSizeA << endl;
  cout << "# outputs total    = " << outSizeA + outSizeR << endl;
  
  delete [] inputValues;
  delete [] outputValuesA;
  delete [] outputValuesR;
  
  return 0;
}
