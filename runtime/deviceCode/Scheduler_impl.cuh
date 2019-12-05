#ifndef __SCHEDULER_IMPL_CUH
#define __SCHEDULER_IMPL_CUH

//
// @file Scheduler_impl.cuh
// @brief Implementation of scheduler run loop; separated from
// Scheduler.h to avoid circular dependency with NodeBase
//
// MERCATOR
// Copyright (C) 2019 Washington University in St. Louis; all rights reserved.
//

#include "Scheduler.cuh"

#include "NodeBase.cuh"

namespace Mercator  {
  
  //
  // @brief run the MERCATOR application to consume all input
  //
  
  __device__
  void Scheduler::run()
  {
    NODE_TIMER_START(scheduler);
    
    while (true)
      {
	NodeBase *nextNode;
	
	COUNT_SCHED_LOOP();
	
	if (IS_BOSS())
	  {
	    nextNode = ( top < 0 ? nullptr : workList[top--]);
	  }
	unsigned long long v = (unsigned long long) nextNode;
	nextNode = (NodeBase *) __shfl_sync(0xffffffff, v, 0);
	
	if (!nextNode) // queue is empty -- terminate
	  break;
	
	NODE_TIMER_STOP(scheduler);
	  
	nextNode->fire();
	
	NODE_TIMER_START(scheduler);
      }
    
    NODE_TIMER_STOP(scheduler);
  }

} // end Mercator namespace
#endif
