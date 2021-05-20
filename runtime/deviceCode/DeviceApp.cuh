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
#include "instrumentation/sched_counter.cuh"

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
      const unsigned int sourceNodeIdx = 0; // nodes are sorted topologically
      
      // call init hooks for each node
      for (unsigned int j = 0; j < numNodes; j++)
	nodes[j]->init();

      __syncthreads(); // writes from init() visible to all threads
      
      if (IS_BOSS())
	{
	  nodes[sourceNodeIdx]->activate();
	}
      
      scheduler.run();
      
#ifndef NDEBUG
      if (IS_BOSS())
	{
	  // sanity check -- make sure no node still has pending inputs
	  bool hasPending = false;
	  for (unsigned int j = 0; j < numNodes; j++)
	    hasPending |= nodes[j]->hasPending();
	  
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
      printf("blockIdx,nodeID,user,push,overhead\n");
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
      printf("blockIdx,nodeID,totalInputs,totalRuns,totalFullRuns\n");
    }
#endif
    
#ifdef INSTRUMENT_SCHED_COUNTS
  __device__
  void printSchedLoopCount() const
    {
      scheduler.printLoopCount();
    }
    
#endif    

#ifdef INSTRUMENT_OUT_DIST
    __device__
    void printOutputDistribution() const
    {
      for (unsigned int j = 0; j < numNodes; j++)
	nodes[j]->printOutputDistributionCSV(j);
    }
    
    __device__
    static
    void printOutputDistributionCSVHeader()
    {
      printf("blockIdx,nodeID,channelIdx,numberOutputs,totalTimes\n");
    }
#endif

#ifdef INSTRUMENT_MAXVECTORGAIN_DIST
    __device__
    void printMaxVectorGainDistribution() const
    {
      for (unsigned int j = 0; j < numNodes; j++)
	nodes[j]->printMaxVectorGainDistributionCSV(j);
    }
    
    __device__
    static
    void printMaxVectorGainDistributionCSVHeader()
    {
      printf("blockIdx,nodeID,channelIdx,maxNumberOutputs,occurances\n");
    }
#endif
  protected:
    
    __device__
    void registerNodes(NodeBase * const *inodes) 
    {
      for (unsigned int j = 0; j < numNodes; j++)
	nodes[j] = inodes[j];
    }

    Scheduler scheduler;
    
  private:
    
    NodeBase *nodes[numNodes];
  };
}

#endif
