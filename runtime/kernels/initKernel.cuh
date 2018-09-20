#ifndef __INIT_KERNEL_CUH
#define __INIT_KERNEL_CUH

//
// @file initKernel.cu
// @brief initialize a MERCATOR app on the device
//
// MERCATOR
// Copyright (C) 2018 Washington University in St. Louis; all rights reserved.
//

#include <cstddef>
#include <cassert>

#include "device_config.cuh"

#include "options.cuh"

namespace Mercator {
  
  //
  // @brief Do all device-side one-time setup for app.
  //
  // @tparam HostAppParams type of host application's parameters
  // @tparam DeviceAppClass type of device application
  //
  // @param sourceTailPtr shared tail pointer to manage source
  // @param hostAppParams host-side application parameter struct
  // @param deviceApps array of ptrs to hold per-block device apps
  //
  // NB: HostAppParams is HostAppClass::Params, and DeviceAppClass is
  //     HostAppClass::DevApp, but C++ appears unable to match this
  //     function template if I specify them that way. Sigh.
  //
  template<typename HostAppParams, typename DeviceAppClass>
  __global__
  void initKernel(size_t *sourceTailPtr,
		  const HostAppParams *hostParams,
		  DeviceAppClass **deviceApps)
  {    
    assert(IS_BOSS());

    deviceApps[blockIdx.x] = new DeviceAppClass(sourceTailPtr, hostParams);
    
    // make sure alloc succeeded
      // make sure alloc succeeded
      if (deviceApps[blockIdx.x] == nullptr)
	{
	  printf("ERROR: failed to allocate app object [block %d]\n",
		 blockIdx.x);
	  
	  crash();
	}
    
#ifdef INSTRUMENT_TIME
    if (IS_BOSS_BLOCK())
      DeviceAppClass::printTimersCSVHeader();
#endif

#ifdef INSTRUMENT_OCC
    if (IS_BOSS_BLOCK())
      DeviceAppClass::printOccupancyCSVHeader();
#endif

#ifdef INSTRUMENT_COUNTS
    if (IS_BOSS_BLOCK())
      DeviceAppClass::printCountsCSVHeader();
#endif
  }
}    // end Mercator namespace

#endif
