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

int main(int argc, char* argv[])
{

  unsigned int NVALUES = 1000000000;
  unsigned int lowerBound = 0;
  unsigned int upperBound = 1000000;
  //const unsigned int NVALUES = 4000; // one BEEEELLION values
  //const unsigned int NVALUES = 800000; // one BEEEELLION values

  if(argc>1){
    //we are in non-default mode here
    //we can configure inputs for testing
    if(atoi(argv[1]) == 1){
      printf("Test harness args format:\n");
      printf("%s <controlFlg> <numVals> <lowerBound> <upperBound>\n", argv[0]);
      printf(" - <controlFlg> :: flags to change various things\n");
      printf(" --- <flag> :: 1 for help or 2 for args \n");
      printf(" - <numVals> :: the number of values to generate to be processed by the application\n");
      printf(" - <upperBound> :: the cycle threshold, after which data is no longer collected\n");
      return 0;
    }
    else if(atoi(argv[1]) == 2 && argc==4){
     // NVALUES = (unsigned int)atoi(argv[2]);
     // lowerBound = (unsigned int)atoi(argv[3]);
     // upperBound = (unsigned int)atoi(argv[4]);
      printf("Values ignored till further notice, running default\n");
    }
    else{
      printf("Invalid options or incorrect num of args. \"Run %s 1\" for usage info\n", argv[0]);
      return 0;
    }
  }
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
  
  delete [] inputValues;
  delete [] outputValues;
  
  return 0;
}
