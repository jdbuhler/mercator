#ifndef __REP4SAMEPIPE_DRIVER_CU
#define __REP4SAMEPIPE_DRIVER_CU


/**
 * @brief Driver (test harnesses) for Mercator app
 *          Rep4SamePipe.
 */

#include <iostream>
#include <cstdlib>

#include "support/util.cuh"

#include "driver_config.cuh"

#include "./tests/PipeEltT.cuh"   
#include "./tests/NodeDataT.cuh" 

#include "./utils.cuh"

// only support 1-to-1 mapping for now
#if MAPPING_1TO1
  #include "../codegenInput/Rep4SamePipe_1to1map.cuh"
  #define APP_TYPE Rep4SamePipe_1to1map
//#elif MAPPING_1TO2
//  #include "../codegenInput/SameTypePipe_1to2map.cuh"
//  #define APP_TYPE SameTypePipe_1to2map
//#elif MAPPING_1TO4
//  #include "../codegenInput/SameTypePipe_1to4map.cuh"
//  #define APP_TYPE SameTypePipe_1to4map
//#elif MAPPING_2TO1
//  #include "../codegenInput/SameTypePipe_2to1map.cuh"
//  #define APP_TYPE SameTypePipe_2to1map
//#elif MAPPING_4TO1
//  #include "../codegenInput/SameTypePipe_4to1map.cuh"
//  #define APP_TYPE SameTypePipe_4to1map
#else
  #error "INVALID MAPPING SELECTION"
#endif

#define PRINT_INPUT_BUFFER_REP4SAMEPIPE 0
#define PRINT_OUTPUT_BUFFERS_REP4SAMEPIPE 0

void run_rep4SamePipe()
{
  // replication factor from inputs to outputs-- should equal number
  //   of parallel pipes
//  constexpr int GAMMA = 2;
  constexpr int GAMMA = 1;

  const int OUT_BUFFER_CAPACITY = GAMMA * IN_BUFFER_CAPACITY; 

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
#elif RUN_REPSAMEPIPE
      const char topoString[] = "RepSamePipe";
#elif RUN_REPDIFFPIPE
      const char topoString[] = "RepDiffPipe";
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

      //debug
//      printf("Size of input item: %d\n", sizeof(PipeEltT));
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

  // copy of input data for later validation
  PipeEltT* inBufferData_gold = new PipeEltT[IN_BUFFER_CAPACITY];
  for(int i=0; i < IN_BUFFER_CAPACITY; ++i)
  {
    inBufferData_gold[i] = inBuffer->peek(i);
  }


#if PRINT_INPUT_BUFFER_REPSAMEPIPE
  // print input buffer contents
  printf("RepSamePipe, InputBuffer (%p):\n", inBuffer);
  for(int i=0; i < IN_BUFFER_CAPACITY; ++i)
  {
    printf("[%d]: ID: %d loop count: %d work loops: %d Int result: %d Double result: %lf Float result: %f\n", 
        i, inBuffer->peek(i).get_ID(),
        inBuffer->peek(i).get_loopCount(),
        inBuffer->peek(i).get_workIters(),
        inBuffer->peek(i).get_intResult(),
        inBuffer->peek(i).get_doubleResult(),
        inBuffer->peek(i).get_floatResult());
  }
#endif

  // create app object
  APP_TYPE* rep4SamePipe = new APP_TYPE();

  // set up each main node in pipeline
  constexpr int NUM_NODES = 5;

  NodeDataT* myNodeData[NUM_NODES];

  int upperBd = NUM_INPUTS;
  int lastUpperBd = upperBd;  // final (lowest) filter value; used for
                              //  validation
  for(int i=0; i < NUM_NODES; ++i)
  {
    // debug
//    printf("Setting upper bound of node %d to %d\n", i, upperBd);
    if(i == NUM_NODES - 1)
      lastUpperBd = upperBd;
    // end debug

    myNodeData[i] = new NodeDataT(float(FILTER_RATE), upperBd);
    upperBd -= (int)(FILTER_RATE * (float)upperBd);
  }

  rep4SamePipe->A->set_nodeUserData(APP_TYPE::A::Node::A11node, myNodeData[0]);
  rep4SamePipe->A->set_nodeUserData(APP_TYPE::A::Node::A12node, myNodeData[1]);
  rep4SamePipe->A->set_nodeUserData(APP_TYPE::A::Node::A13node, myNodeData[2]);
  rep4SamePipe->A->set_nodeUserData(APP_TYPE::A::Node::A14node, myNodeData[3]);
  rep4SamePipe->A->set_nodeUserData(APP_TYPE::A::Node::A15node, myNodeData[4]);
  rep4SamePipe->A->set_nodeUserData(APP_TYPE::A::Node::A21node, myNodeData[0]);
  rep4SamePipe->A->set_nodeUserData(APP_TYPE::A::Node::A22node, myNodeData[1]);
  rep4SamePipe->A->set_nodeUserData(APP_TYPE::A::Node::A23node, myNodeData[2]);
  rep4SamePipe->A->set_nodeUserData(APP_TYPE::A::Node::A24node, myNodeData[3]);
  rep4SamePipe->A->set_nodeUserData(APP_TYPE::A::Node::A25node, myNodeData[4]);
  rep4SamePipe->A->set_nodeUserData(APP_TYPE::A::Node::A31node, myNodeData[0]);
  rep4SamePipe->A->set_nodeUserData(APP_TYPE::A::Node::A32node, myNodeData[1]);
  rep4SamePipe->A->set_nodeUserData(APP_TYPE::A::Node::A33node, myNodeData[2]);
  rep4SamePipe->A->set_nodeUserData(APP_TYPE::A::Node::A34node, myNodeData[3]);
  rep4SamePipe->A->set_nodeUserData(APP_TYPE::A::Node::A35node, myNodeData[4]);
  rep4SamePipe->A->set_nodeUserData(APP_TYPE::A::Node::A41node, myNodeData[0]);
  rep4SamePipe->A->set_nodeUserData(APP_TYPE::A::Node::A42node, myNodeData[1]);
  rep4SamePipe->A->set_nodeUserData(APP_TYPE::A::Node::A43node, myNodeData[2]);
  rep4SamePipe->A->set_nodeUserData(APP_TYPE::A::Node::A44node, myNodeData[3]);
  rep4SamePipe->A->set_nodeUserData(APP_TYPE::A::Node::A45node, myNodeData[4]);

  // associate buffers with nodes
  rep4SamePipe->sourceNode->set_inBuffer(inBuffer);
  rep4SamePipe->sinkNodeAccept1->set_outBuffer(outBufferAccept);
  rep4SamePipe->sinkNodeAccept2->set_outBuffer(outBufferAccept);
  rep4SamePipe->sinkNodeAccept3->set_outBuffer(outBufferAccept);
  rep4SamePipe->sinkNodeAccept4->set_outBuffer(outBufferAccept);

  // run main function
  rep4SamePipe->run();

  std::cout << "REP-4-SAME-PIPE APP FINISHED.\n" ;

  /////////////////// output processing

  PipeEltT* outDataAccept = outBufferAccept->get_data();

  // print contents of output buffer
#if PRINT_OUTPUT_BUFFERS_REP4SAMEPIPE
  std::cout << " Output buffer: \n" ;

  printf("Rep-4-same-pipe, OutBufferAccept (%p):\n", outBufferAccept);
  for(int i=0; i < outBufferAccept->size(); ++i)
    printf("[%d]: ID: %d work loops: %d Int result: %d Double result: %lf Float result: %f\n", 
        i, outDataAccept[i].get_ID(), 
        outDataAccept[i].get_workIters(),
        outDataAccept[i].get_intResult(),
        outDataAccept[i].get_doubleResult(),
        outDataAccept[i].get_floatResult());

#endif   // print contents of output buffer

  //////////////
  // validate output against pristine copy of input buffer
  // NB: since pipeline is replicated, TWO copies of every valid output
  //     should exist in output buffer
  //////////////

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
  for(int i=0; i < NUM_NODES; ++i)
  {
    cudaFree(myNodeData[i]);
    gpuErrchk( cudaPeekAtLastError() );
  }

  cudaFree(inBufferData);
  gpuErrchk( cudaPeekAtLastError() );
  cudaFree(rep4SamePipe);
  gpuErrchk( cudaPeekAtLastError() );

  delete[] inBufferData_gold;
}

#endif
