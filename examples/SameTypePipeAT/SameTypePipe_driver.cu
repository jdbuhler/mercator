/**
 * @brief Driver (test harnesses) for Mercator app
 *          PipeSameType.
 */

#include <iostream>

#include "driver_config.cuh"

#include "PipeEltT.cuh"   

#include "validate.cuh"

#if defined(MAPPING_1TO1)
#include "SameTypePipe_1to1map.cuh"
#define APP_TYPE SameTypePipe_1to1map
#elif defined(MAPPING_1TO2)
#include "SameTypePipe_1to2map.cuh"
#define APP_TYPE SameTypePipe_1to2map
#elif defined(MAPPING_1TO4)
#include "SameTypePipe_1to4map.cuh"
#define APP_TYPE SameTypePipe_1to4map
#elif defined(MAPPING_2TO1)
#include "SameTypePipe_2to1map.cuh"
#define APP_TYPE SameTypePipe_2to1map
#elif defined(MAPPING_4TO1)
#include "SameTypePipe_4to1map.cuh"
#define APP_TYPE SameTypePipe_4to1map
#else
#error "INVALID MAPPING SELECTION"
#endif

//#define PRINT_INPUT
//#define PRINT_OUTPUT

using namespace std;

const int NTRIALS = 1;

int main(int argc, char* argv[])
{
  const char topoString[] = "SameTypePipe";
  
  const char mapString[] =
#if defined(MAPPING_1TO1)
    "1-to-1"
#elif defined(MAPPING_1TO2)
    "1-to-2"
#elif defined(MAPPING_1TO4)
    "1-to-4"
#elif defined(MAPPING_2TO1)
    "2-to-1"
#elif defined(MAPPING_4TO1)
    "4-to-1"
#else
    "NONSTANDARD"
#endif
    ;
  
  // print app metadata
  cout << "APP PARAMS: TOPOLOGY: " << topoString 
       << " ELTS-TO-THREADS MAPPING: " << mapString
       << " FILTER_RATE: " << FILTER_RATE
       << " WORK_ITERS: " << WORK_ITERS
       << " INPUTS: " << NUM_INPUTS
       << endl;
  
  int inputSeed;
#ifdef USE_REPEATABLE_INPUTS
  inputSeed = 1919;
#else
  inputsSeed = time(0);
#endif
  srand(inputSeed);
  
  const unsigned int MAX_OUTPUTS = NUM_INPUTS;
  
  // host-side input and output data
  PipeEltT* inputs = new PipeEltT [NUM_INPUTS];
  PipeEltT* outputs = new PipeEltT [MAX_OUTPUTS];
  
  // create buffers
  Mercator::Buffer<PipeEltT> inBuffer(NUM_INPUTS);
  Mercator::Buffer<PipeEltT> outBufferAccept(MAX_OUTPUTS);
  
  // create app object
  APP_TYPE app;
  
  cout << "# GPU BLOCKS = " << app.getNBlocks() << endl; 
  for (int trial = 0; trial < NTRIALS; trial++)
    {
      // generate input data
      for (int i = 0; i < NUM_INPUTS; ++i)
	{
#ifdef USE_RANDOM_INPUTS
	  int nextID = rand() % NUM_INPUTS;
#else
	  int nextID = i;
#endif
	  
	  PipeEltT elt(nextID, WORK_ITERS);
	  inputs[i] = elt;
	}
      
#ifdef PRINT_INPUT
      
      // print input data
      
      for (int i=0; i < NUM_INPUTS; ++i)
	{
	  PipeEltT elt = inputs[i];

	  cout << "[" << i 
	       << "]: ID: " << elt.get_ID()
	       << " work iters: " << elt.get_workIters()
	       << endl;
	}
#endif
      
      inBuffer.set(inputs, NUM_INPUTS);
      
      app.getParams()->seed = 1919;
      
      // set up each node in pipeline
      
      int lastUpperBd; // final (lowest) filter value, for validation
      {
	int upperBd = NUM_INPUTS;
	
	auto params = app.A1node.getParams();
	params->filterRate = float(FILTER_RATE);
	params->upperBound = upperBd;
	
	upperBd -= (int)(FILTER_RATE * (float)upperBd);

	params = app.A2node.getParams();
	params->filterRate = float(FILTER_RATE);
	params->upperBound = upperBd;

	upperBd -= (int)(FILTER_RATE * (float)upperBd);

	params = app.A3node.getParams();
	params->filterRate = float(FILTER_RATE);
	params->upperBound = upperBd;
	
	upperBd -= (int)(FILTER_RATE * (float)upperBd);
	
	params = app.A4node.getParams();
	params->filterRate = float(FILTER_RATE);
	params->upperBound = upperBd;
	
	upperBd -= (int)(FILTER_RATE * (float)upperBd);
	
	params = app.A5node.getParams();
	params->filterRate = float(FILTER_RATE);
	params->upperBound = upperBd;
	
	lastUpperBd = upperBd;
      }
      
      // associate buffers with nodes
      
      app.setSource(inBuffer);
      app.sinkNodeAccept.setSink(outBufferAccept);
      
      cout << "SAME-TYPE-PIPE APP LAUNCHING.\n" ;
      // run main function
      app.run();
      
      cout << "SAME-TYPE-PIPE APP FINISHED.\n" ;
      
      /////////////////// output processing
      
      unsigned int outsize = outBufferAccept.size();
      outBufferAccept.get(outputs, outsize);
      
#ifdef PRINT_OUTPUT
      
      // print contents of output buffer
      
      cout << " Output buffer: \n" ;
      
      for (int i=0; i < outsize; ++i)
	{
	  const PipeEltT &elt = outputs[i];
	  
	  cout << "[" << i 
	       << "]: ID: " << elt.get_ID()
	       << " work iters: " << elt.get_workIters()
	       << " Float result: " << elt.get_floatResult()
	       << endl;
	}
#endif
      
      //////////////
      // validate output against pristine copy of input buffer
      // NB: requires specific knowledge of desired filtering 
      //      behavior within app
      //////////////
      
      // replication factor
      constexpr int GAMMA = 1;
      
      bool allValid = 
	validate_lowpassFilterApp_outputs(inputs,
					  NUM_INPUTS,
					  outputs,
					  outsize,
					  (1.0-FILTER_RATE) * lastUpperBd,
					  GAMMA);
      
      if (allValid)
	{
	  cout << "OUTPUT VALIDATED CORRECT, " 
	       << outBufferAccept.size() << " items." 
	       << endl;
	}
      else
	{
	  cout << "OUTPUT CONTAINS ERRORS. " 
	       << endl;
	  exit(1);
	}
      
      // clean out the output buffer for the next trial;
      // the input buffer has been emptied
      
      outBufferAccept.clear();
    }
  
  // cleanup
  
  delete [] inputs;
  delete [] outputs;
  
  return 0;
}
