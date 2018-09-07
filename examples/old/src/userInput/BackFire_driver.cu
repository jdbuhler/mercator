#ifndef __BACKFIRE_DRIVER_CU
#define __BACKFIRE_DRIVER_CU


/**
 * @brief Driver (test harnesses) for Mercator app
 *          BackFire.
 */

#include <iostream>

#include "driver_config.cuh"

#include "../codegenInput/BackFire.cuh"

#define PRINT_OUTPUT_BUFFERS 1

void run_backFire()
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
  BackFire* backFire = new BackFire();

  // set node-, module-, app-level user data
  MyModuleData* filter1data = new MyModuleData(2);
  backFire->Filter1->set_userData(filter1data); // regular module

  MyModuleData* filter2data = new MyModuleData(7);
  backFire->Filter2->set_userData(filter2data); // regular module

  MyNodeData* filter1nodeData = new MyNodeData(3);
  backFire->Filter1->set_nodeUserData(BackFire::Filter1::Node::filter1node, filter1nodeData);

  MyNodeData* filter2nodeData = new MyNodeData(5);
  backFire->Filter2->set_nodeUserData(BackFire::Filter2::Node::filter2node, filter2nodeData);

  // associate buffers with nodes
  backFire->sourceNode->set_inBuffer(inBuffer);
  backFire->sinkNodeReject1->set_outBuffer(outBuffer1);
  backFire->sinkNodeReject2->set_outBuffer(outBuffer2);
  backFire->sinkNodeAccept->set_outBuffer(outBuffer3);

  // run main function
  backFire->run();

  std::cout << "BackFire finished.\n" ;

  // print contents of output buffer
#if PRINT_OUTPUT_BUFFERS
  std::cout << " Output buffers: \n" ;

  int* outData1 = outBuffer1->get_data();
  printf("BackFire, OutBuffer1 (%p):\n", outBuffer1);
  for(int i=0; i < outBuffer1->size(); ++i)
    printf("[%d]: %d\n", i, outData1[i]);

  int* outData2 = outBuffer2->get_data();
  printf("BackFire, OutBuffer2 (%p):\n", outBuffer2);
  for(int i=0; i < outBuffer2->size(); ++i)
    printf("[%d]: %d\n", i, outData2[i]);

  int* outData3 = outBuffer3->get_data();
  printf("BackFire, OutBuffer3 (%p):\n", outBuffer3);
  for(int i=0; i < outBuffer3->size(); ++i)
    printf("[%d]: %d\n", i, outData3[i]);
#endif   // print contents of output buffer

  // cleanup
  cudaFree(inBufferData);
  cudaFree(backFire);

}

#endif
