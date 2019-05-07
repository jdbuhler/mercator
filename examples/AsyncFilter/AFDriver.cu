//
// AFDRIVER.CU
// Async Filter test
//
// MERCATOR
// Copyright (C) 2018 Washington University in St. Louis; all rights reserved.
//

#include <iostream>
#include <cstdlib>

#include "AsyncFilter.cuh"

using namespace std;

int main()
{
  const unsigned int NVALUES = 1000000000;
  
  srand(0);
  
  unsigned int *inputValues = new unsigned int [NVALUES];
  unsigned int *outputValues = new unsigned int [NVALUES];
  
  for (unsigned int j = 0; j < NVALUES; j++)
    inputValues[j] = rand();
  
  // begin MERCATOR usage
  
  Mercator::Buffer<unsigned int> buf1(NVALUES);
  Mercator::Buffer<unsigned int> buf2(NVALUES);
  
  AsyncFilter afapp;

  // move data into the input buffer
  buf1.set(inputValues, NVALUES);
  
  cout << "RUN 1\n"; // read from buf1; write to buf2
  
  afapp.src.setSource(buf1);
  afapp.snk.setSink(buf2);
  afapp.Filter.getParams()->modulus = 2;
  
  afapp.runAsync();

  cout << "RUN 2\n"; // read from buf2; write to buf1
  
  // clear out buffer 1 -- the clearing will not occur until all
  // previous calls in the current stream are complete, so this call
  // does not impact the data for the previous run.
  buf1.clearAsync();
  
  // runAsync() takes a snapshot of the app's parameters before
  // returning, so it is safe to change them even if the
  // work started by the previous call to runAsync() is still
  // in progress.
  afapp.src.setSource(buf2);
  afapp.snk.setSink(buf1);
  afapp.Filter.getParams()->modulus = 3;
  
  afapp.runAsync();
  
  cout << "RUN 3\n"; // read from buf1; write to buf2
  
  // clear out buffer 2
  buf2.clearAsync();
  
  afapp.src.setSource(buf1);
  afapp.snk.setSink(buf2);
  afapp.Filter.getParams()->modulus = 5;
  
  afapp.runAsync();
  
  cout << "JOIN\n"; // pause until all operations in stream are done

  afapp.join();
  
  // get data out of the output buffer
  unsigned int outSize = buf2.size();
  buf2.get(outputValues, outSize);
  
  // end MERCATOR usage
  
  cout << "# outputs = " << outSize << endl;
  
  delete [] inputValues;
  delete [] outputValues;
  
  return 0;
}
