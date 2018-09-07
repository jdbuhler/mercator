#include <iostream>
#include <cstdlib>

#include "LoopFilter.cuh"

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
  
  LoopFilter lfapp;
  
  lfapp.src.setSource(inputBuffer);
  lfapp.snk.setSink(outputBufferAccept);
  
  lfapp.getParams()->numCycles = 3;
  
  // when allocating a parameter array that will be used
  // on the device, use cudaMallocManaged() so that no
  // explicit host->device copy is required.
  unsigned int *moduli;
  cudaMallocManaged(&moduli, 3 * sizeof(unsigned int));
  moduli[0] = 2;
  moduli[1] = 3;
  moduli[2] = 5;
  lfapp.getParams()->moduli = moduli;
  
  // move data into the input buffer
  inputBuffer.set(inputValues, NVALUES);
  
  lfapp.run();
  
  // get data out of the output buffer
  unsigned int outSize = outputBufferAccept.size();
  outputBufferAccept.get(outputValues, outSize);
  
  cudaFree(moduli);
  
  // end MERCATOR usage
  
  cout << "# outputs = " << outSize << endl;
  
  delete [] inputValues;
  delete [] outputValues;
  
  return 0;
}
