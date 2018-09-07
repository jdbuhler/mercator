#ifndef __REP4DIFFPIPE_DRIVER_CU
#define __REP4DIFFPIPE_DRIVER_CU


/**
 * @brief Driver (test harnesses) for Mercator app
 *          Rep4SemiDiffPipe.
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
  #include "../codegenInput/Rep4DiffPipe_1to1map.cuh"
  #define APP_TYPE Rep4DiffPipe_1to1map
//#elif MAPPING_1TO2
//  #include "../codegenInput/DiffTypePipe_1to2map.cuh"
//  #define APP_TYPE DiffTypePipe_1to2map
//#elif MAPPING_1TO4
//  #include "../codegenInput/DiffTypePipe_1to4map.cuh"
//  #define APP_TYPE DiffTypePipe_1to4map
//#elif MAPPING_2TO1
//  #include "../codegenInput/DiffTypePipe_2to1map.cuh"
//  #define APP_TYPE DiffTypePipe_2to1map
//#elif MAPPING_4TO1
//  #include "../codegenInput/DiffTypePipe_4to1map.cuh"
//  #define APP_TYPE DiffTypePipe_4to1map
#else
  #error "INVALID MAPPING SELECTION"  
#endif


#define PRINT_INPUT_BUFFER_REP4DIFFPIPE 0
#define PRINT_OUTPUT_BUFFERS_REP4DIFFPIPE 0

void run_rep4DiffPipe()
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
#elif RUN_REP4SAMEPIPE
      const char topoString[] = "Rep4SamePipe";
#elif RUN_REP4DIFFPIPE
      const char topoString[] = "Rep4DiffPipe";
#elif RUN_REP4SEMIDIFFPIPE
      const char topoString[] = "Rep4SemiDiffPipe";
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
    //    inBuffer->add(PipeEltT(nextTTL, i)); 
  }

#if PRINT_INPUT_BUFFER_REP4DIFFPIPE 
  // print input buffer contents
  printf("Rep4DiffPipe, InputBuffer (%p):\n", inBuffer);
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

  // debug
  // copy of input data for later validation
  PipeEltT* inBufferData_gold = new PipeEltT[IN_BUFFER_CAPACITY];
  for(int i=0; i < IN_BUFFER_CAPACITY; ++i)
  {
    inBufferData_gold[i] = inBuffer->peek(i);
  }

  // create app object
  APP_TYPE* rep4DiffPipe = new APP_TYPE();

  // set up each main node in pipeline
  constexpr int NUM_NODES = 5;

  NodeDataT* myNodeData[NUM_NODES];

  int upperBd = NUM_INPUTS;
  int lastUpperBd = upperBd;  // final (lowest) filter value; used for
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

  rep4DiffPipe->A1->set_nodeUserData(APP_TYPE::A1::Node::A1node, myNodeData[0]);
  rep4DiffPipe->B1->set_nodeUserData(APP_TYPE::B1::Node::B1node, myNodeData[1]);
  rep4DiffPipe->C1->set_nodeUserData(APP_TYPE::C1::Node::C1node, myNodeData[2]);
  rep4DiffPipe->D1->set_nodeUserData(APP_TYPE::D1::Node::D1node, myNodeData[3]);
  rep4DiffPipe->E1->set_nodeUserData(APP_TYPE::E1::Node::E1node, myNodeData[4]);
  rep4DiffPipe->A2->set_nodeUserData(APP_TYPE::A2::Node::A2node, myNodeData[0]);
  rep4DiffPipe->B2->set_nodeUserData(APP_TYPE::B2::Node::B2node, myNodeData[1]);
  rep4DiffPipe->C2->set_nodeUserData(APP_TYPE::C2::Node::C2node, myNodeData[2]);
  rep4DiffPipe->D2->set_nodeUserData(APP_TYPE::D2::Node::D2node, myNodeData[3]);
  rep4DiffPipe->E2->set_nodeUserData(APP_TYPE::E2::Node::E2node, myNodeData[4]);
  rep4DiffPipe->A3->set_nodeUserData(APP_TYPE::A3::Node::A3node, myNodeData[0]);
  rep4DiffPipe->B3->set_nodeUserData(APP_TYPE::B3::Node::B3node, myNodeData[1]);
  rep4DiffPipe->C3->set_nodeUserData(APP_TYPE::C3::Node::C3node, myNodeData[2]);
  rep4DiffPipe->D3->set_nodeUserData(APP_TYPE::D3::Node::D3node, myNodeData[3]);
  rep4DiffPipe->E3->set_nodeUserData(APP_TYPE::E3::Node::E3node, myNodeData[4]);
  rep4DiffPipe->A4->set_nodeUserData(APP_TYPE::A4::Node::A4node, myNodeData[0]);
  rep4DiffPipe->B4->set_nodeUserData(APP_TYPE::B4::Node::B4node, myNodeData[1]);
  rep4DiffPipe->C4->set_nodeUserData(APP_TYPE::C4::Node::C4node, myNodeData[2]);
  rep4DiffPipe->D4->set_nodeUserData(APP_TYPE::D4::Node::D4node, myNodeData[3]);
  rep4DiffPipe->E4->set_nodeUserData(APP_TYPE::E4::Node::E4node, myNodeData[4]);

  // associate buffers with nodes
  rep4DiffPipe->sourceNode->set_inBuffer(inBuffer);
  rep4DiffPipe->sinkNodeAccept1->set_outBuffer(outBufferAccept);
  rep4DiffPipe->sinkNodeAccept2->set_outBuffer(outBufferAccept);
  rep4DiffPipe->sinkNodeAccept3->set_outBuffer(outBufferAccept);
  rep4DiffPipe->sinkNodeAccept4->set_outBuffer(outBufferAccept);

  // run main function
  rep4DiffPipe->run();

  std::cout << "REP-4-DIFF-PIPE APP FINISHED.\n" ;

  /////////////////// output processing

  PipeEltT* outDataAccept = outBufferAccept->get_data();

  // print contents of output buffer
#if PRINT_OUTPUT_BUFFERS_DIFFTYPEPIPE
  std::cout << " Output buffer: \n" ;

  printf("Rep-4-diff-pipe, OutBufferAccept (%p):\n", outBufferAccept);
  for(int i=0; i < outBufferAccept->size(); ++i)
    printf("[%d]: ID: %d work iters: %d Int result: %d Double result: %lf Float result: %f\n", 
        i, outDataAccept[i].get_ID(), 
        outDataAccept[i].get_workIters(),
        outDataAccept[i].get_intResult(),
        outDataAccept[i].get_doubleResult(),
        outDataAccept[i].get_floatResult());

#endif  


  //////////////
  // validate output against pristine copy of input buffer
  // NB: requires specific knowledge of desired filtering 
  //      behavior within app
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
  cudaFree(rep4DiffPipe);
  gpuErrchk( cudaPeekAtLastError() );

  delete[] inBufferData_gold;
}

#if 0
void run_diffTypePipe()
{
  // set up input buffer
  //  const int IN_BUFFER_CAPACITY = 32;
  //  const int IN_BUFFER_CAPACITY = 128;
  //  const int IN_BUFFER_CAPACITY = 256;
  //  const int IN_BUFFER_CAPACITY = 512;
  const int IN_BUFFER_CAPACITY = 1024;
  //  const int IN_BUFFER_CAPACITY = 16384;
  //  const int IN_BUFFER_CAPACITY = 32768;
  //  const int IN_BUFFER_CAPACITY = 65536;
  //  const int IN_BUFFER_CAPACITY = 1<<20;


  // output buffers 
  const int OUT_BUFFER_CAPACITY = 1 * IN_BUFFER_CAPACITY; 

  PipeEltT* inBufferData;
  cudaMallocManaged(&inBufferData, IN_BUFFER_CAPACITY * sizeof(PipeEltT));
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );

#if PRINT_MEM_USAGE
  printf("Allocating global mem for input buffer, capacity: %d size (bytes): %ld\n", IN_BUFFER_CAPACITY, (long long)(IN_BUFFER_CAPACITY * sizeof(PipeEltT)));
#endif

  Mercator::InputBuffer<PipeEltT>* inBuffer = new Mercator::InputBuffer<PipeEltT>(inBufferData, IN_BUFFER_CAPACITY);


#if 0
  Mercator::OutputBuffer<PipeEltT>* outBufferRejectA = new Mercator::OutputBuffer<PipeEltT>(OUT_BUFFER_CAPACITY);
  Mercator::OutputBuffer<PipeEltT>* outBufferRejectB = new Mercator::OutputBuffer<PipeEltT>(OUT_BUFFER_CAPACITY);
  Mercator::OutputBuffer<PipeEltT>* outBufferRejectC = new Mercator::OutputBuffer<PipeEltT>(OUT_BUFFER_CAPACITY);
#endif

  Mercator::OutputBuffer<PipeEltT>* outBufferAccept = new Mercator::OutputBuffer<PipeEltT>(OUT_BUFFER_CAPACITY);

  // fill input buffer
  const int SEED = 191919;
  srand(SEED);
  for(int i=0; i < IN_BUFFER_CAPACITY; ++i)
  {
    // set random TTL in range [0,9]
    int nextTTL = rand() % 10;
    inBuffer->add(PipeEltT(nextTTL, i)); 
  }

  // print input buffer contents
  printf("SelfLoop, InputBuffer (%p):\n", inBuffer);
  for(int i=0; i < IN_BUFFER_CAPACITY; ++i)
  {
    printf("[%d]: ID: %d TTL: %d\n", 
        i, inBuffer->peek(i).get_ID(),
        inBuffer->peek(i).get_ttL());
  }

  // create app object
  DiffTypePipe* diffTypePipe = new DiffTypePipe();

  // associate buffers with nodes
  diffTypePipe->sourceNode->set_inBuffer(inBuffer);
  diffTypePipe->sinkNodeAccept->set_outBuffer(outBufferAccept);

  // run main function
  diffTypePipe->run();

  std::cout << "DIFF-TYPE-PIPE APP FINISHED.\n" ;

  // print contents of output buffer
#if PRINT_OUTPUT_BUFFERS
  std::cout << " Output buffer: \n" ;

  PipeEltT* outDataAccept = outBufferAccept->get_data();
  printf("DiffTypePipe, OutBufferAccept (%p):\n", outBufferAccept);
  for(int i=0; i < outBufferAccept->size(); ++i)
    printf("[%d]: %d\n", i, outDataAccept[i].get_ID());

#endif   // print contents of output buffer

  // cleanup
  cudaFree(inBufferData);
  cudaFree(diffTypePipe);

}
#endif

#endif
