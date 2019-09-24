#ifndef __DEVICEAPP_CUH
#define __DEVICEAPP_CUH

//
// @file DeviceApp
// base class of all MERCATOR device-side apps
//
// MERCATOR
// Copyright (C) 2018 Washington University in St. Louis; all rights reserved.
//

#include <cstdio>

#include "options.cuh"

#include "ModuleTypeBase.cuh"
#include "Scheduler.cuh"

#include "instrumentation/device_timer.cuh"
#include "instrumentation/occ_counter.cuh"
#include "instrumentation/item_counter.cuh"

namespace Mercator {
 
  template <unsigned int numModules,
	    unsigned int _THREADS_PER_BLOCK,
	    unsigned int _DEVICE_STACK_SIZE,
	    unsigned int _DEVICE_HEAP_SIZE>
  class DeviceApp {
    
  public:
    
    static const unsigned int THREADS_PER_BLOCK = _THREADS_PER_BLOCK;
    static const unsigned int DEVICE_STACK_SIZE = _DEVICE_STACK_SIZE;
    static const unsigned int DEVICE_HEAP_SIZE  = _DEVICE_HEAP_SIZE;
    
    //
    // @brief constructor
    // initialize space for app's modules
    //
    __device__
    DeviceApp()
    {
      for (unsigned int j = 0; j < numModules; j++)
	modules[j] = nullptr;
    }

    //
    // @brief destructor
    // clean up app's modules
    //
    __device__
    virtual ~DeviceApp() 
    {
      for (unsigned int j = 0; j < numModules; j++)
	{
	  if (modules[j])
	    delete modules[j];
	}
    }
    
    //
    // @brief run an app in one block from the main kernel
    //   We need to initialize all modules, then actually
    //   do the run, and finally clean up all modules
    //
    __device__
    void run()
    {
      // call init hooks for each module
      for (unsigned int j = 0; j < numModules; j++)
	modules[j]->init();
      
      __syncthreads(); // make sure init is visible to all threads

      //setup shortcut addresses for fast accesses
      for (unsigned int j = 0; j < numModules; j++){
            modules[j]->addressShortCut();  
      } 
      
      
      scheduler.run(modules, modules[sourceModuleIdx]);
      
      __syncthreads(); // make sure run is complete in all threads

      // call cleanup hooks for each module
      for (unsigned int j = 0; j < numModules; j++)
	modules[j]->cleanup();    
    }
    
    /////////////////////////////////////////////////////////////
    // INSTRUMENTATION PRINTING
    /////////////////////////////////////////////////////////////
    
#ifdef INSTRUMENT_TIME
    __device__
    void printTimers() const
    {
      scheduler.printTimersCSV();
      
      for (unsigned int j = 0; j < numModules; j++)
	modules[j]->printTimersCSV(j);
    }

    __device__
    static
    void printTimersCSVHeader()
    {
      printf("blockIdx,moduleID,gather,run,scatter\n");
    }
#endif
    
#ifdef INSTRUMENT_OCC
    __device__
    void printOccupancy() const
    {
      for (unsigned int j = 0; j < numModules; j++)
	modules[j]->printOccupancyCSV(j);
    }
    
    __device__
    static
    void printOccupancyCSVHeader()
    {
      printf("blockIdx,moduleID,maxWidth,totalInputs,totalRuns,totalFullRuns\n");
    }
#endif
    
#ifdef INSTRUMENT_COUNTS
    __device__
    void printCounts() const
    {
      for (unsigned int j = 0; j < numModules; j++)
	modules[j]->printCountsCSV(j);
    }

    __device__
    static
    void printCountsCSVHeader()
    {
      printf("blockIdx,moduleID,channelId,nodeId,count\n");
    }
#endif
    
  protected:
    
    __device__
    void registerModules(ModuleTypeBase * const *imodules, 
		    unsigned int isourceModuleIdx) 
    {
      for (unsigned int j = 0; j < numModules; j++)
	modules[j] = imodules[j];
      
      sourceModuleIdx = isourceModuleIdx;
    }
    
  private:
    
    Scheduler<numModules, THREADS_PER_BLOCK> scheduler;
    
    ModuleTypeBase *modules[numModules];
    unsigned int sourceModuleIdx;
  };
}

#endif
