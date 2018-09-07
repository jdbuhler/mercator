#ifndef __CLEANUP_KERNEL_CUH
#define __CLEANUP_KERNEL_CUH

//
// @file cleanupKernel.cuh
// @brief clean up after a MERCATOR app
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
  }
  
}   // end Mercator namespace

#endif
