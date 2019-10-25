#ifndef __DEVICE_TIMER_H
#define __DEVICE_TIMER_H

//
// DEVICE_TIMER.CUH
// Device-side timers based on cycle counter
//
// MERCATOR
// Copyright (C) 2018 Washington University in St. Louis; all rights reserved.
//

#include "options.cuh"

#ifdef INSTRUMENT_TIME

#include "device_config.cuh" // for IS_BOSS

class DeviceTimer {

public:  

  typedef unsigned long long DevClockT;
  
  __device__
  DeviceTimer() 
  {
    if (IS_BOSS()){
      totalTime = 0;
      //next_timer= rand select threead
    } 
  }

  __device__
  DevClockT getTotalTime() const 
  { return totalTime; }

  __device__
  void start() 
  { 
    __syncthreads();
    if (IS_BOSS())
      lastStart = clock64(); 
    /*
    remove sync thread
    if(threadIdx.x==next_timer){
      lastStart = clock64(); 
      next_timer= rand select threead
    }
    */
  }
  
  __device__
  void stop()  
  { 
    __syncthreads();
    if (IS_BOSS())
    {
      DevClockT now = clock64();
      totalTime += timeDiff(lastStart, now);
    }
    /*
    remove sync thread
    if(threadIdx.x==next_timer){
      DevClockT now = clock64();
      totalTime += timeDiff(lastStart, now);
      next_timer= rand select threead
    }
    */
  }
  
private:

  DevClockT totalTime;  
  
  DevClockT lastStart;

  __device__
  DevClockT timeDiff(DevClockT start, DevClockT end)
  {
    DevClockT diff = end - start;
 
    if (end < start)
      {
        diff += DevClockT(-1);
        diff++;
      }

    return diff;
  }
};

#endif

//
// macros so that we can do timings only if needed,
// without a lot of extra lines of code.  We assume
// that a timer object tm##Timer is in scope.
//

#ifdef INSTRUMENT_TIME
#define NODE_TIMER_START(tm) { (tm ## Timer).start(); } 
#else
#define NODE_TIMER_START(tm) {}
#endif

#ifdef INSTRUMENT_TIME
#define NODE_TIMER_STOP(tm) { (tm ## Timer).stop(); } 
#else
#define NODE_TIMER_STOP(tm) {}
#endif

#endif
