#ifndef __CLEANUP_KERNEL_CUH
#define __CLEANUP_KERNEL_CUH

//
// @file cleanupKernel.cuh
// @brief clean up after a MERCATOR app
//
// MERCATOR
// Copyright (C) 2018 Washington University in St. Louis; all rights reserved.
//

#include <cassert>

#include "device_config.cuh"

namespace Mercator {
  
  //
  // @brief clean up after a MERCATOR app by deleting the object
  // for each block.
  //
  template <class DeviceAppClass>
  __global__
  void cleanupKernel(DeviceAppClass **deviceAppObjs)
  {
    assert(IS_BOSS());
    
    assert(deviceAppObjs[blockIdx.x]);
    
    delete deviceAppObjs[blockIdx.x];
    if(threadIdx.x==0 and blockIdx.x==0)printf("app done\n");
  }
  
}   // end Mercator namespace

#endif
