#ifndef __UBERNODE_PIPE_DRIVER_CU
#define __UBERNODE_PIPE_DRIVER_CU


/**
 * @brief Driver (test harnesses) for Mercator app
 *          UberNodePipe.
 */

#include <iostream>
#include <cstdlib>

#include "support/util.cuh"

#include "driver_config.cuh"

#include "./tests/PipeEltT.cuh"   
#include "./tests/NodeDataT.cuh" 

#include "./utils.cuh"

//#include "../codegenInput/UberNodePipe.cuh"

#if MAPPING_1TO1
  #include "../codegenInput/UberNodePipe_1to1map.cuh"
  #define APP_TYPE UberNodePipe_1to1map
#elif MAPPING_1TO2
  #include "../codegenInput/UberNodePipe_1to2map.cuh"
  #define APP_TYPE UberNodePipe_1to2map
#elif MAPPING_1TO4
  #include "../codegenInput/UberNodePipe_1to4map.cuh"
  #define APP_TYPE UberNodePipe_1to4map
#elif MAPPING_2TO1
  #include "../codegenInput/UberNodePipe_2to1map.cuh"
  #define APP_TYPE UberNodePipe_2to1map
#elif MAPPING_4TO1
  #include "../codegenInput/UberNodePipe_4to1map.cuh"
  #define APP_TYPE UberNodePipe_4to1map
#else
  #error "INVALID MAPPING SELECTION"  
#endif

#define PRINT_INPUT_BUFFER_UBERNODEPIPE 0
#define PRINT_OUTPUT_BUFFERS_UBERNODEPIPE 0

void run_uberNodePipe()
{
  const int OUT_BUFFER_CAPACITY = 1 * IN_BUFFER_CAPACITY; 

  // set input info
  constexpr int NUM_INPUTS = IN_BUFFER_CAPACITY;

#if 1
  // print experiment params if desired
  // NB: all possible topos included for sanity check
  // convert topology indicators to string
#if RUN_SAMETYPEPIPE
      const char topoString[] = "SameTypePipe";
#elif RUN_DIFFTYPEPIPE
      const char topoString[] = "DiffTypePipe";
#elif RUN_UBERNODEPIPE
      const char topoString[] = "UberNodePipe";
#elif RUN_SELFLOOPPIPE
      const char topoString[] = "SelfLoop";
#else
      const char topoString[] = "NONSTANDARD";
#endif

      // convert mapping indicators to string
#if MAPPING_1TO1
      const char mapString[] = "1-to-1";
#elif MAPPING_1TO2
      const char mapString[] = "1-to-2";
#elif MAPPING_1TO4
      const char mapString[] = "1-to-4";
#elif MAPPING_2TO1
      const char mapString[] = "2-to-1";
#elif MAPPING_4TO1
      const char mapString[] = "4-to-1";
#else
      const char mapString[] = "NONSTANDARD";
#endif

      // print app metadata
      printf("APP PARAMS: TOPOLOGY: %s ELTS-TO-THREADS MAPPING: %s FILTER_RATE: %.2f WORK_ITERS: %d INPUTS: %d\n", 
          topoString, mapString, FILTER_RATE, WORK_ITERS, NUM_INPUTS); 
#endif

  int inputSeed;
#if USE_REPEATABLE_INPUTS
  inputSeed = 1919;
#else
  inputsSeed = time(0);
#endif

  // alloc input data
  PipeEltT* inBufferData;
  cudaMallocManaged(&inBufferData, IN_BUFFER_CAPACITY * sizeof(PipeEltT));
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );

#if PRINT_MEM_USAGE
  printf("Allocating global mem for input buffer, capacity: %d size (bytes): %ld\n", IN_BUFFER_CAPACITY, (long long)(IN_BUFFER_CAPACITY * sizeof(PipeEltT)));
#endif

  // create buffers
  Mercator::InputBuffer<PipeEltT>* inBuffer = new Mercator::InputBuffer<PipeEltT>(inBufferData, IN_BUFFER_CAPACITY);

  Mercator::OutputBuffer<PipeEltT>* outBufferAccept = new Mercator::OutputBuffer<PipeEltT>(OUT_BUFFER_CAPACITY);

  // fill input buffer
  srand(inputSeed);
  for(int i=0; i < IN_BUFFER_CAPACITY; ++i)
  {
#if USE_RANDOM_INPUTS
    int nextID = rand() % IN_BUFFER_CAPACITY;
#else
    int nextID = i;
#endif
    // set random TTL in range [0,9]
    int nextTTL = rand() % 10;
    int loopCount = 0;
    int numWorkIters = WORK_ITERS; 

    inBuffer->add(PipeEltT(nextID, loopCount, numWorkIters)); 
  }

#if PRINT_INPUT_BUFFER_UBERNODEPIPE
  // print input buffer contents
  printf("UberNodeLoop, InputBuffer (%p):\n", inBuffer);
  for(int i=0; i < IN_BUFFER_CAPACITY; ++i)
  {
    printf("[%d]: ID: %d loop count: %d work iters: %d Int result: %d Double result: %lf Float result: %f\n", 
        i, inBuffer->peek(i).get_ID(),
        inBuffer->peek(i).get_loopCount(),
        inBuffer->peek(i).get_workIters(),
        inBuffer->peek(i).get_intResult(),
        inBuffer->peek(i).get_doubleResult(),
        inBuffer->peek(i).get_floatResult());
  }
#endif

  // copy of input data for later validation
  PipeEltT* inBufferData_gold = new PipeEltT[IN_BUFFER_CAPACITY];
  for(int i=0; i < IN_BUFFER_CAPACITY; ++i)
  {
    inBufferData_gold[i] = inBuffer->peek(i);
  }

  // create app object
  APP_TYPE* uberNodePipe = new APP_TYPE();

  // set up each main node in pipeline
  constexpr int NUM_NODES = 5;

  NodeData5FiltT* myNodeData = new NodeData5FiltT;

  int upperBd = NUM_INPUTS;
  int lastUpperBd = upperBd;  // final (lowest) filter value; used for
  for(int i=0; i < NUM_NODES; ++i)
  {
    // debug
//    printf("Setting upper bound of node %d to %d\n", i, upperBd);
    if(i == NUM_NODES - 1)
      lastUpperBd = upperBd;
    // end debug

    myNodeData->set_filterRate(i, float(FILTER_RATE));
    myNodeData->set_upperBound(i, upperBd);
    upperBd -= (int)(FILTER_RATE * (float)upperBd);
  }

  uberNodePipe->Uber->set_nodeUserData(APP_TYPE::Uber::Node::Ubernode, myNodeData);

  // associate buffers with nodes
  uberNodePipe->sourceNode->set_inBuffer(inBuffer);
  uberNodePipe->sinkNodeAccept->set_outBuffer(outBufferAccept);

  // run main function
  uberNodePipe->run();

  std::cout << "UBERNODE-PIPE APP FINISHED.\n" ;

  /////////////////// output processing

  PipeEltT* outDataAccept = outBufferAccept->get_data();

  // print contents of output buffer
#if PRINT_OUTPUT_BUFFERS_UBERNODEPIPE
  std::cout << " Output buffer: \n" ;

  printf("Ubernode-pipe, OutBufferAccept (%p):\n", outBufferAccept);
  for(int i=0; i < outBufferAccept->size(); ++i)
    printf("[%d]: ID: %d work iters: %d Int result: %d Double result: %lf Float result: %f\n", 
        i, outDataAccept[i].get_ID(), 
        outDataAccept[i].get_workIters(),
        outDataAccept[i].get_intResult(),
        outDataAccept[i].get_doubleResult(),
        outDataAccept[i].get_floatResult());

#endif   // print contents of output buffer

  //////////////
  // validate output against pristine copy of input buffer
  // NB: requires specific knowledge of desired filtering 
  //      behavior within app
  //////////////
  constexpr int GAMMA = 1;

  bool allValid = validate_lowpassFilterApp_outputs(
    inBufferData_gold,
    IN_BUFFER_CAPACITY,
    outDataAccept,
    outBufferAccept->size(),
    (1-FILTER_RATE) * lastUpperBd,
    GAMMA);

  if(allValid)
    printf("OUTPUT VALIDATED CORRECT, %d items.\n", outBufferAccept->size());
  else
    printf("OUTPUT CONTAINS ERRORS.\n");
  /////////////////////////////////////////

  // cleanup
//  for(int i=0; i < NUM_NODES; ++i)
//  {
    cudaFree(myNodeData);
    gpuErrchk( cudaPeekAtLastError() );
//  }

  cudaFree(inBufferData);
  gpuErrchk( cudaPeekAtLastError() );
  cudaFree(uberNodePipe);
  gpuErrchk( cudaPeekAtLastError() );

  delete[] inBufferData_gold;
}

#endif
