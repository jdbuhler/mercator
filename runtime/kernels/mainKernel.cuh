#ifndef __MAIN_KERNEL_CUH
#define __MAIN_KERNEL_CUH

//
// @file mainKernel.cuh
// @brief main kernel to execute a MERCATOR app
//
// MERCATOR
// Copyright (C) 2018 Washington University in St. Louis; all rights reserved.
//

#include <cassert>

#include "options.cuh"
#include "support/collective_ops.cuh"

namespace Mercator   {

  //
  // @brief Run a MERCATOR app in every block on the GPU
  //
  // @tparam DeviceApp type of device-side application
  // 
  // @param deviceAppObjs array of per-block device-side apps
  //
  template <typename DeviceAppClass>
  __global__
  void mainKernel(DeviceAppClass **deviceAppObjs)
  {
    assert(deviceAppObjs[blockIdx.x]);

#ifdef INSTRUMENT_TIME
    __syncthreads();
    unsigned long long start = clock64();
#endif
    
    deviceAppObjs[blockIdx.x]->run();
    
#ifdef INSTRUMENT_TIME
    __syncthreads();
    unsigned long long stop = clock64();    
    
    if (IS_BOSS())
      {
	printf("%u: main kernel runtime: %llu\n",
	       blockIdx.x, stop - start);
      }
#endif
    
#ifdef INSTRUMENT_TIME
    if (IS_BOSS())
      deviceAppObjs[blockIdx.x]->printTimers();
#endif

#ifdef INSTRUMENT_OCC
    if (IS_BOSS())
      deviceAppObjs[blockIdx.x]->printOccupancy();
#endif

#ifdef INSTRUMENT_SCHED_COUNTS
    if (IS_BOSS())
      deviceAppObjs[blockIdx.x]->printSchedLoopCount();
#endif

#ifdef INSTRUMENT_OUT_DIST
    if (IS_BOSS())
      deviceAppObjs[blockIdx.x]->printOutputDistribution();
#endif

#ifdef INSTRUMENT_MAXVECTORGAIN_DIST
    if (IS_BOSS())
      deviceAppObjs[blockIdx.x]->printMaxVectorGainDistribution();
#endif
  }
}    // end Mercator namespace

#endif
