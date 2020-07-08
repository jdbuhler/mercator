#ifndef __SCHEDULER_CUH
#define __SCHEDULER_CUH

//
// @file Scheduler.cuh
// @brief MERCATOR device-side node scheduler
//
// MERCATOR
// Copyright (C) 2019 Washington University in St. Louis; all rights reserved.
//

#include <cstdio>
#include <cassert>

#include "options.cuh"

#include "device_config.cuh"

#include "instrumentation/device_timer.cuh"
#include "instrumentation/sched_counter.cuh"

namespace Mercator  {

  class NodeBase; // fwd declaratino
  
  //
  // @class Scheduler
  // @brief Schedules all nodes in an application
  //
  
  class Scheduler {
    
  public:
    
    //
    // @brief Constructor
    // Called single-threaded from init kernel
    // 
    __device__
    Scheduler(unsigned int numNodes)
    {
      workList = new NodeBase * [numNodes];
      top = -1;
    }
    
    //
    // @brief destructor
    // called single-threaded from cleanup kernel
    //
    __device__
    ~Scheduler() 
    {
      delete [] workList;
    }
    
    //
    // @brief run the MERCATOR application to consume all input
    //

    __device__
    void run();
    
    __device__
    void addFireableNode(NodeBase *node)
    {
      assert(IS_BOSS());
      
      workList[++top] = node;
    }
    
#ifdef INSTRUMENT_TIME
    __device__
    void printTimersCSV() const
    {
      printf("%d,%d,%llu,%llu,%llu\n",
	     blockIdx.x, -1, schedulerTimer.getTotalTime(), 0, 0);
    }
#endif

#ifdef INSTRUMENT_SCHED_COUNTS
  __device__
  void printLoopCount() const
    {
      printf("%u: Sched Loop Count: %llu\n", 
	     blockIdx.x, schedCounter.getLoopCount());
    }
#endif
    
  private:
    
    NodeBase **workList;
    int top;
    
#ifdef INSTRUMENT_TIME
    DeviceTimer schedulerTimer;
#endif
    
#ifdef INSTRUMENT_SCHED_COUNTS
    SchedCounter schedCounter;
#endif
    
  };  // end Scheduler class
}   // end Mercator namespace

#endif
