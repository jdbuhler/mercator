//
// EFDRIVER.CU
// Even-valued filter test
//
// MERCATOR
// Copyright (C) 2018 Washington University in St. Louis; all rights reserved.
//

#include <iostream>

#include "EvenFilter.cuh"

using namespace std;

int main()
{
  const unsigned int NVALUES = 1000000000; // one BEEEELLION values
  //const unsigned int NVALUES = 100000; // one BEEEELLION values
  
  unsigned int *outputValues = new unsigned int [NVALUES];
  
  // begin MERCATOR usage
  
  Mercator::Buffer<unsigned int> outputBufferAccept(NVALUES);
  
  EvenFilter efapp;
  
  efapp.setNInputs(NVALUES);
  efapp.snk.setSink(outputBufferAccept);
  
  efapp.run();
  
  // get data out of the output buffer
  unsigned int outSize = outputBufferAccept.size();
  outputBufferAccept.get(outputValues, outSize);
  
  // end MERCATOR usage
  
  cout << "# outputs = " << outSize << endl;
  /*
  for(unsigned int i = 0; i < outSize; i++)
    cout << outputValues[i] << endl;
  */
  
  delete [] outputValues;
  
  return 0;
}
