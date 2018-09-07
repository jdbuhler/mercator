#include <iostream>

#include "ModFilter.cuh"

using namespace std;

int main()
{
  const unsigned int NVALUES = 1000000000; // one BEEEELLION values
  
  unsigned int *outputValues = new unsigned int [NVALUES];
  
  // begin MERCATOR usage
  
  // Use a "Range" type to specify a range of equally-spaced
  // numbers as input to an app.  a Range takes constant space.
  
  Mercator::Range<unsigned int> range(0, NVALUES, 1);
  Mercator::Buffer<unsigned int> outputBufferAccept(NVALUES);
  
  ModFilter mfapp;
  
  mfapp.src.setSource(range);
  mfapp.snk.setSink(outputBufferAccept);
  
  mfapp.f1.getParams()->modulus = 2;
  mfapp.f2.getParams()->modulus = 3;
  mfapp.f3.getParams()->modulus = 5;
  
  mfapp.run();
  
  // get data out of the output buffer
  unsigned int outSize = outputBufferAccept.size();
  outputBufferAccept.get(outputValues, outSize);
  
  // end MERCATOR usage
  
  cout << "# outputs = " << outSize << endl;
  
  delete [] outputValues;
  
  return 0;
}
