//
// SGDRIVER.CU
// Synthetic gains test, setting different gain for each node.
//
// MERCATOR
// Copyright (C) 2018 Washington University in St. Louis; all rights reserved.
//

#include <iostream>
#include <iomanip>

#include "SynthGain.cuh"

#define GAIN1 0.25
#define GAIN2 11.5
#define GAIN3 0.125
#define GAIN4 0.25
#define GAIN5 0.5

using namespace std;

int main()
{
  //const unsigned int NVALUES = 1000000000; // one BEEEELLION values
  const unsigned int NVALUES = 100000000; // one BEEEELLION values
  //const unsigned int NVALUES = 10000; // one BEEEELLION values
  //const unsigned int NVALUES = 128; // one BEEEELLION values
  
  // begin MERCATOR usage
  
  SynthGain app;
  
  auto params1 = app.t1.getParams();
  params1->avgGain = float(GAIN1);
  auto params2 = app.t2.getParams();
  params2->avgGain = float(GAIN2);
  auto params3 = app.t3.getParams();
  params3->avgGain = float(GAIN3);
  auto params4 = app.t4.getParams();
  params4->avgGain = float(GAIN4);
  auto params5 = app.t5.getParams();
  params5->avgGain = float(GAIN5);

  float totalAvg = GAIN1 * GAIN2 * GAIN3 * GAIN4 * GAIN5;
  unsigned int totalAvgInt = (unsigned int)(totalAvg);
  if(totalAvg - totalAvgInt > 0.0) {
	++totalAvgInt;
  }

  Mercator::Buffer<size_t> outputBufferAccept(NVALUES*totalAvgInt);
  size_t *outputValues = new size_t [NVALUES*totalAvgInt];

  app.setNInputs(NVALUES);
  app.snk.setSink(outputBufferAccept);

  app.run();
  
  // get data out of the output buffer
  unsigned int outSize = outputBufferAccept.size();
  outputBufferAccept.get(outputValues, outSize);
  
  // end MERCATOR usage
  
  cout << "Cumulative Gain = " << totalAvg << endl;
  cout << "Total Output Space = " << NVALUES * totalAvgInt << endl;
  cout << "Total Number Outputs = " << outSize << endl;
  cout << "Expected Number Outputs = " << setprecision(20) << NVALUES * totalAvg << endl;
  /*
  for(unsigned int i = 0; i < outSize; i++)
    cout << outputValues[i] << endl;
  */
  
  delete [] outputValues;
  
  return 0;
}
