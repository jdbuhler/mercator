#ifndef __SYNTHAPP3_DRIVER_CU
#define __SYNTHAPP3_DRIVER_CU


/**
 * @brief Driver (test harnesses) for Mercator app
 *          SynthApp3.
 */

#include <iostream>

#include "driver_config.cuh"

#include "./tests/datatypes.h"
#include "../codegenInput/SynthApp3.cuh"

#define PRINT_OUTPUT_BUFFERS 1

void run_synthApp3()
{
  // output buffers accommodates 4 outputs/input
  const int OUT_BUFFER_CAPACITY1 = 4 * IN_BUFFER_CAPACITY; 
  const int OUT_BUFFER_CAPACITY2 = 16 * IN_BUFFER_CAPACITY; 

  int* inBufferData;
  cudaMallocManaged(&inBufferData, IN_BUFFER_CAPACITY * sizeof(int));
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );

#if PRINT_MEM_USAGE
  printf("Allocating global mem for input buffer, capacity: %d size (bytes): %ld\n", IN_BUFFER_CAPACITY, (long long)(IN_BUFFER_CAPACITY * sizeof(int)));
#endif

  Mercator::InputBuffer<int>* inBuffer = new Mercator::InputBuffer<int>(inBufferData, IN_BUFFER_CAPACITY);


  Mercator::OutputBuffer<int>* outBuffer1 = new Mercator::OutputBuffer<int>(OUT_BUFFER_CAPACITY1);
  Mercator::OutputBuffer<int>* outBuffer2 = new Mercator::OutputBuffer<int>(OUT_BUFFER_CAPACITY2);
  Mercator::OutputBuffer<int>* outBuffer3 = new Mercator::OutputBuffer<int>(OUT_BUFFER_CAPACITY2);

  // fill input buffer
  for(int i=0; i < IN_BUFFER_CAPACITY; ++i)
  {
    inBuffer->add(i); 
  }

  // create app object
  SynthApp3* synthApp3 = new SynthApp3();

  // set node-, module-, app-level user data
  MyModuleData* filter1data = new MyModuleData(2);
  synthApp3->Filter1->set_userData(filter1data); // regular module

  MyModuleData* filter2data = new MyModuleData(7);
  synthApp3->Filter2->set_userData(filter2data); // regular module

  MyNodeData* filter1nodeData = new MyNodeData(3);
  
  synthApp3->Filter1->set_nodeUserData(SynthApp3::Filter1::Node::filter1node, filter1nodeData);

  MyNodeData* filter2nodeData = new MyNodeData(5);
  
  synthApp3->Filter2->set_nodeUserData(SynthApp3::Filter2::Node::filter2node, filter2nodeData);

  // associate buffers with nodes
  synthApp3->sourceNode->set_inBuffer(inBuffer);
  synthApp3->sinkNodeReject1->set_outBuffer(outBuffer1);
  synthApp3->sinkNodeReject2->set_outBuffer(outBuffer2);
  synthApp3->sinkNodeAccept->set_outBuffer(outBuffer3);

  // run main function
  synthApp3->run();

  std::cout << "SynthApp3 finished.\n" ;

  // print contents of output buffer
#if PRINT_OUTPUT_BUFFERS
  std::cout << " Output buffers: \n" ;

  int* outData1 = outBuffer1->get_data();
  printf("SynthApp3, OutBuffer1 (%p):\n", outBuffer1);
  for(int i=0; i < outBuffer1->size(); ++i)
    printf("[%d]: %d\n", i, outData1[i]);

  int* outData2 = outBuffer2->get_data();
  printf("SynthApp3, OutBuffer2 (%p):\n", outBuffer2);
  for(int i=0; i < outBuffer2->size(); ++i)
    printf("[%d]: %d\n", i, outData2[i]);

  int* outData3 = outBuffer3->get_data();
  printf("SynthApp3, OutBuffer3 (%p):\n", outBuffer3);
  for(int i=0; i < outBuffer3->size(); ++i)
    printf("[%d]: %d\n", i, outData3[i]);
#endif   // print contents of output buffer

  // cleanup
  cudaFree(inBufferData);
  cudaFree(synthApp3);

}

#endif
