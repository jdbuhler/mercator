#ifndef __SCHEDULER_IMPL_CUH
#define __SCHEDULER_IMPL_CUH

//
// @file Scheduler.cuh
// @brief MERCATOR device-side node scheduler
//
// MERCATOR
// Copyright (C) 2020 Washington University in St. Louis; all rights reserved.
//

#include "Scheduler.cuh"

#include "NodeBase.cuh"

#include "timing_options.cuh"

namespace Mercator  {

  //
  // @brief run the MERCATOR application while any node is present
  // in the fireable list.
  //
  __device__
  void Scheduler::run()
  {
    TIMER_START(scheduler);
    
    while (true)
      {	
	COUNT_SCHED_LOOP();
	
	__syncthreads(); // BEGIN WRITE nextNode
	
	__shared__ NodeBase *nextNode;
	if (IS_BOSS())
	  {
	    nextNode = (top < 0 ? nullptr : workList[top--]);
	  }
	
	__syncthreads(); // END WRITE nextNode
	
	if (!nextNode) // queue is empty -- terminate
	  break;

	TIMER_STOP(scheduler);

        nextNode->fire();

        TIMER_START(scheduler);
      }
    
    TIMER_STOP(scheduler);
  }
}   // end Mercator namespace

#endif
