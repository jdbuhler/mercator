#ifndef __DEVICEAPP_CUH
#define __DEVICEAPP_CUH

//
// @file DeviceApp
// base class of all MERCATOR device-side apps
//
// MERCATOR
// Copyright (C) 2019 Washington University in St. Louis; all rights reserved.
//

#include <cstdio>

#include "options.cuh"

#include "NodeBase.cuh"
#include "Scheduler.cuh"

#include "instrumentation/device_timer.cuh"
#include "instrumentation/occ_counter.cuh"
#include "instrumentation/item_counter.cuh"

namespace Mercator {
 
  template <unsigned int numNodes,
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
    // initialize space for app's nodes
    //
    __device__
    DeviceApp()
      : scheduler(numNodes)
    {
      for (unsigned int j = 0; j < numNodes; j++)
	nodes[j] = nullptr;
    }

    //
    // @brief destructor
    // clean up app's nodes
    //
    __device__
    virtual ~DeviceApp() 
    {
      for (unsigned int j = 0; j < numNodes; j++)
	{
	  if (nodes[j])
	    delete nodes[j];
	}
    }
    
    //
    // @brief run an app in one block from the main kernel
    //   We need to initialize all nodes, then actually
    //   do the run, and finally clean up all nodes
    //
    __device__
    void run()
    {
      // call init hooks for each node
      for (unsigned int j = 0; j < numNodes; j++)
	nodes[j]->init();
      
      if (IS_BOSS())
	{
	  nodes[sourceNodeIdx]->activate();
	}
      
      __syncthreads(); // init and scheduler state visible to all threads
      
      scheduler.run();
      
      __syncthreads(); // run is complete in all threads
      
#ifndef NDEBUG
      if (IS_BOSS())
	{
	  // sanity check -- make sure no node still has pending inputs
	  bool hasPending = false;
	  for (unsigned int j = 0; j < numNodes; j++)
	    hasPending |= (nodes[j]->numPending() > 0);
	  
	  assert(!hasPending);
	}
#endif
      
      // call cleanup hooks for each node
      for (unsigned int j = 0; j < numNodes; j++)
	nodes[j]->cleanup();    
    }
    
    /////////////////////////////////////////////////////////////
    // INSTRUMENTATION PRINTING
    /////////////////////////////////////////////////////////////
    
#ifdef INSTRUMENT_TIME
    __device__
    void printTimers() const
    {
      scheduler.printTimersCSV();
      
      for (unsigned int j = 0; j < numNodes; j++)
	nodes[j]->printTimersCSV(j);
    }

    __device__
    static
    void printTimersCSVHeader()
    {
      printf("blockIdx,nodeID,input,run,output\n");
    }
#endif
    
#ifdef INSTRUMENT_OCC
    __device__
    void printOccupancy() const
    {
      for (unsigned int j = 0; j < numNodes; j++)
	nodes[j]->printOccupancyCSV(j);
    }
    
    __device__
    static
    void printOccupancyCSVHeader()
    {
      printf("blockIdx,nodeID,maxWidth,totalInputs,totalRuns,totalFullRuns\n");
    }
#endif
    
#ifdef INSTRUMENT_COUNTS
    __device__
    void printCounts() const
    {
      for (unsigned int j = 0; j < numNodes; j++)
	nodes[j]->printCountsCSV(j);
    }

    __device__
    static
    void printCountsCSVHeader()
    {
      printf("blockIdx,nodeID,channelId,nodeId,count\n");
    }
#endif
    
  protected:
    
    __device__
    void registerNodes(NodeBase * const *inodes, 
		       unsigned int isourceNodeIdx) 
    {
      for (unsigned int j = 0; j < numNodes; j++)
	nodes[j] = inodes[j];
      
      sourceNodeIdx = isourceNodeIdx;
    }

    Scheduler scheduler;
    
  private:
    
    NodeBase *nodes[numNodes];
    unsigned int sourceNodeIdx;
  };
}

#endif
