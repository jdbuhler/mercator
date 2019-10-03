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

#include "Queue.cuh"

#include "NodeBase.cuh"

#include "instrumentation/device_timer.cuh"

namespace Mercator  {
  
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
      : workQueue(numNodes)
    {}
    
    //
    // @brief destructor
    // called single-threaded from cleanup kernel
    //
    __device__
    ~Scheduler() {}
    
    //
    // @brief run the MERCATOR application to consume all input
    //

    __device__
    void run()
    {
      //TIMER_START(scheduler);
      
      while (!workQueue.empty())
	{
	  __shared__ NodeBase *nextNode;
	  
	  if (IS_BOSS())
	    {
	      nextNode = workQueue.dequeue();
	    }
	  
	  __syncthreads(); // for nextNode, queue status
	  
	  nextNode->fire();
	  
	  // boss thread did any updates to fireable item queue, so 
	  // it sees all newly enqueued items w/o a syncthreads
	}
      
      //TIMER_STOP(scheduler);
    }
    
    __device__
    void addFireableNode(NodeBase *node)
    {
      assert(IS_BOSS());

      workQueue.enqueue(node);
    }
    
    
#ifdef INSTRUMENT_TIME
    __device__
    void printTimersCSV() const
    {
      printf("%d,%d,%llu,%llu,%llu\n",
	     blockIdx.x, -1, schedulerTimer.getTotalTime(), 0, 0);
    }
#endif
    
  private:
    
    Queue<NodeBase *> workQueue;
        
#ifdef INSTRUMENT_TIME
    DeviceTimer schedulerTimer;
#endif
    
  };  // end Scheduler class
}   // end Mercator namespace

#endif
