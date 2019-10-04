#ifndef __SCHED_COUNTER_H
#define __SCHED_COUNTER_H

//
// ITEM_COUNTER.H
// Instrumentation to count itms into/out of a module
//
// MERCATOR
// Copyright (C) 2018 Washington University in St. Louis; all rights reserved.
//

#ifdef INSTRUMENT_SCHED_COUNTS

struct SchedCounter{
  
  __device__
  SchedCounter()
  {
      loopCount= 0;
  }
  
  // multithreaded increment, using numInstances threads
  __device__
  void incr()
  {
    if(IS_BOSS()){
      loopCount+=1;
    }
  }
  __device__
  unsigned long long getLoopCount()const {
    return loopCount;
  } 
 
  unsigned long long loopCount;
};

#endif

#ifdef INSTRUMENT_SCHED_COUNTS
#define COUNT_SCHED_LOOP()         { schedCounter.incr(); }
#else
#define COUNT_SCHED_LOOP()         { }
#endif

#endif
