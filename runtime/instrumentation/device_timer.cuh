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

//#ifdef INSTRUMENT_TIME //or INSTRUMENT_FG_TIME
#if defined(INSTRUMENT_TIME) || defined(INSTRUMENT_FG_TIME)
#include "device_config.cuh" // for IS_BOSS

class DeviceTimer {

public:  

  typedef unsigned long long DevClockT;
  
  __device__
  DeviceTimer() 
  {
    if (IS_BOSS())
      totalTime = 0; 
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
  }
  
  __device__
  void fine_start() 
  { 
    __syncthreads();
    if (IS_BOSS())
      fineStart = clock64(); 
  }
  
  __device__
  void fine_stop()  
  { 
    __syncthreads();
    if (IS_BOSS())
      {
	DevClockT now = clock64();
	// some array location = timeDiff(fineStart, now);
      }
  }
private:

  DevClockT totalTime;  
  DevClockT fineStart;
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

// coarse grained timer, times total amount of time spent in a node for life of kernel
#ifdef INSTRUMENT_TIME
#define TIMER_START(tm) { (tm ## Timer).start(); } 
#else
#define TIMER_START(tm) {}
#endif

#ifdef INSTRUMENT_TIME
#define TIMER_STOP(tm) { (tm ## Timer).stop(); } 
#else
#define TIMER_STOP(tm) {}
#endif

// fine grained timers, gets time spent in a single firing of single vector width of single node of single block
#ifdef INSTRUMENT_FG_TIME
#define FINE_TIMER_START(tm) { (tm ## Timer).fine_start(); } 
#else
#define FINE_TIMER_START(tm) {}
#endif

#ifdef INSTRUMENT_FG_TIME
#define FINE_TIMER_STOP(tm) { (tm ## Timer).fine_stop(); } 
#else
#define FINE_TIMER_STOP(tm) {}
#endif


#endif
