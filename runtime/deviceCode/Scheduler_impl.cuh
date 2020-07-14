#ifndef __SCHEDULER_IMPL_CUH
#define __SCHEDULER_IMPL_CUH

//
// @file Scheduler.cuh
// @brief MERCATOR device-side node scheduler
//
// MERCATOR
// Copyright (C) 2019 Washington University in St. Louis; all rights reserved.
//

#include "Scheduler.cuh"

#include "NodeBase.cuh"

namespace Mercator  {
  
  __device__
  void Scheduler::run()
  {
    NODE_TIMER_START(scheduler);
    
    while (true)
      {	
	COUNT_SCHED_LOOP();
		
	__shared__ NodeBase *nextNode;
	if (IS_BOSS())
	  {
	    nextNode = (top < 0 ? nullptr : workList[top--]);
	  }
	__syncthreads(); // for nextNode
	
	if (!nextNode) // queue is empty -- terminate
	  break;
	
	NODE_TIMER_STOP(scheduler);
	
	nextNode->fire();
	
	__syncthreads(); // protect state of workList from activations
	
	NODE_TIMER_START(scheduler);
      }
    
    NODE_TIMER_STOP(scheduler);
  }
}   // end Mercator namespace

#endif
