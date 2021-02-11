//
// EFDRIVER.CU
// Even-valued filter test
//
// MERCATOR
// Copyright (C) 2018 Washington University in St. Louis; all rights reserved.
//

#include <iostream>

#include "SynthGain.cuh"

using namespace std;

int main()
{
  //const unsigned int NVALUES = 1000000000; // one BEEEELLION values
  const unsigned int NVALUES = 10000; // one BEEEELLION values
  
  unsigned int *outputValues = new unsigned int [NVALUES*16];
  
  // begin MERCATOR usage
  
  Mercator::Buffer<unsigned int> outputBufferAccept(NVALUES*16);
  
  SynthGain app;
  
  app.setNInputs(NVALUES);
  app.snk.setSink(outputBufferAccept);

  auto params = app.t1.getParams();
  params->avgGain = float(6.5);
  
  app.run();
  
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
