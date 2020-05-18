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
#include "instrumentation/sched_counter.cuh"

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
      : workQueue(numNodes),
	localFlushSize(numNodes + 1)
    {
	localFlush = new bool [numNodes + 1];
	//stimcheck: TODO need to get the number of enumIds set here, not numNodes
	//just using numNodes instead for now since enumIds is strictly less than
	//numNodes + 1.  The 0th position is the default flag, and is never modified.
	for(unsigned int i = 0; i < numNodes + 1; ++i) {
		localFlush[i] = false;
	}
    }
    
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
      NODE_TIMER_START(scheduler);
      
      while (true)
	{
	  __shared__ NodeBase *nextNode;
	  
          COUNT_SCHED_LOOP();
	  
	  if (IS_BOSS())
	    {
	      nextNode = (workQueue.empty() ? nullptr : workQueue.dequeue());
	    }
	  __syncthreads(); // for nextNode
	  
	  if (!nextNode) // queue is empty -- terminate
	    break;

	  __syncthreads();
	  
	  NODE_TIMER_STOP(scheduler);

	  //if(nextNode) { //stimcheck: No idea why this has to be here, but otherwise we get an illegal address access
	  	if(nextNode->getWriteThruId() > 0) {
		    assert(nextNode->getWriteThruId() < localFlushSize);
		    if(!(localFlush[nextNode->getWriteThruId()])) {
			//remove local write thru id and DO NOT fire node
			nextNode->setWriteThruId(0);
			//deactivate?
		    }
		  }
		  else {
		  	nextNode->fire();
		  }
	  //}

	  __syncthreads();
	  
	  NODE_TIMER_START(scheduler);
	}
      
      NODE_TIMER_STOP(scheduler);
    }
    
    __device__
    void addFireableNode(NodeBase *node)
    {
      assert(IS_BOSS());

      workQueue.enqueue(node);
    }

    __device__
    void setLocalFlush(unsigned int i)
    {
      assert(IS_BOSS());
      assert(i < localFlushSize);

      localFlush[i] = true;
    }

    __device__
    void removeLocalFlush(unsigned int i)
    {
      assert(IS_BOSS());
      assert(i < localFlushSize);

      localFlush[i] = false;
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
  void printLoopCount() const{
    printf("%u: Sched Loop Count: %llu\n", blockIdx.x, schedCounter.getLoopCount());
  }
#endif
    
  private:
    
    Queue<NodeBase *> workQueue;

    bool* localFlush;
    unsigned int localFlushSize;
        
#ifdef INSTRUMENT_TIME
    DeviceTimer schedulerTimer;
#endif

#ifdef INSTRUMENT_SCHED_COUNTS
    SchedCounter schedCounter;
#endif
    
  };  // end Scheduler class
}   // end Mercator namespace

#endif
