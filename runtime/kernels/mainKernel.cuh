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
    
    deviceAppObjs[blockIdx.x]->run();
    
#ifdef INSTRUMENT_TIME
    if (IS_BOSS())
      deviceAppObjs[blockIdx.x]->printTimers();
#endif

#ifdef INSTRUMENT_OCC
    if (IS_BOSS())
      deviceAppObjs[blockIdx.x]->printOccupancy();
#endif

#ifdef INSTRUMENT_COUNTS
    if (IS_BOSS())
      deviceAppObjs[blockIdx.x]->printCounts();
#endif
#ifdef INSTRUMENT_SCHED_COUNTS
    if (IS_BOSS())
      deviceAppObjs[blockIdx.x]->printSchedLoopCount();
#endif
  }
}    // end Mercator namespace

#endif
