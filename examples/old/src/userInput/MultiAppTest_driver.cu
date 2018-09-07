#ifndef __MULTITYPE_DRIVER_CU
#define __MULTITYPE_DRIVER_CU


/**
 * @brief Driver (test harnesses) for program with
 *        multiple Mercator apps.
 */

#include <iostream>

#include "driver_config.cuh"

#include "../codegenInput/MyApp1.cuh"
#include "../codegenInput/MyApp2.cuh"

#define PRINT_OUTPUT_BUFFERS 1

void run_test_multiApp()
{
  // set up input buffer
  const int IN_BUFFER_CAPACITY = 2048;

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
  MyApp1* myApp1 = new MyApp1();

  UserDataExt* filter1data = new UserDataExt(100);
  myApp1->Filter1->set_userData(filter1data); // regular module

  UserDataExt* filter2nodeData = new UserDataExt(1);
  myApp1->filter2node->set_userData(filter2nodeData);

  // associate buffers with nodes
  myApp1->sourceNode->set_inBuffer(inBuffer);
  myApp1->sinkNodeReject1->set_outBuffer(outBuffer1);
  myApp1->sinkNodeReject2->set_outBuffer(outBuffer2);
  myApp1->sinkNodeAccept->set_outBuffer(outBuffer3);

  // run main function
  myApp1->run();

  std::cout << "App1 finished.\n" ;

  // print contents of output buffer
#if PRINT_OUTPUT_BUFFERS
  std::cout << " Output buffers: \n" ;

  int* outData1 = outBuffer1->get_data();
  printf("App1, OutBuffer1 (%p):\n", outBuffer1);
  for(int i=0; i < outBuffer1->size(); ++i)
    printf("[%d]: %d\n", i, outData1[i]);

  int* outData2 = outBuffer2->get_data();
  printf("App1, OutBuffer2 (%p):\n", outBuffer2);
  for(int i=0; i < outBuffer2->size(); ++i)
    printf("[%d]: %d\n", i, outData2[i]);

  int* outData3 = outBuffer3->get_data();
  printf("App1, OutBuffer3 (%p):\n", outBuffer3);
  for(int i=0; i < outBuffer3->size(); ++i)
    printf("[%d]: %d\n", i, outData3[i]);
#endif   // print contents of output buffer

  // cleanup
  cudaFree(myApp1);
}

#endif
