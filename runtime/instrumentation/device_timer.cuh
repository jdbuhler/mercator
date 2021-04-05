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

#ifdef RAND_TIMING_THREAD
  #include <curand_kernel.h>
  #define NUM_THREADS 128
#endif

class DeviceTimer {

public:  

  typedef unsigned long long DevClockT;
  
  __device__
  DeviceTimer() 
  {
    if (IS_BOSS()){
      totalTime = 0;
      #ifdef RAND_TIMING_THREAD
      curand_init(clock64(), 0, 1, &s);
      next_timer  = ceilf(curand_uniform(&s) * NUM_THREADS);
      #endif
    } 
  }

  __device__
  DevClockT getTotalTime() const 
  { return totalTime; }

  __device__
  void start() 
  { 
    #ifndef RAND_TIMING_THREAD
    __syncthreads();
    if (IS_BOSS())
      lastStart = clock64(); 
    #else
    if(threadIdx.x==next_timer){
      lastStart = clock64(); 
      next_timer  = ceilf(curand_uniform(&s) * NUM_THREADS);
    }
    #endif
  }
  
  __device__
  void stop()  
  { 
    #ifndef RAND_TIMING_THREAD
    __syncthreads();
    if (IS_BOSS())
    {
      DevClockT now = clock64();
      totalTime += timeDiff(lastStart, now);
    }
    #else
    if(threadIdx.x==next_timer){
      DevClockT now = clock64();
      totalTime += timeDiff(lastStart, now);
      next_timer  = ceilf(curand_uniform(&s) * NUM_THREADS);
    }
    #endif
  }
  
private:

  DevClockT totalTime;  
  
  DevClockT lastStart;

#ifdef RAND_TIMING_THREAD
  curandState s;
  unsigned int next_timer;
#endif

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
#define XNODE_TIMER_START(node,tm) { (node->tm ## Timer).start(); } 
#else
#define XNODE_TIMER_START(node,tm) {}
#endif

#ifdef INSTRUMENT_TIME
#define XNODE_TIMER_STOP(node,tm) { (node->tm ## Timer).stop(); } 
#else
#define XNODE_TIMER_STOP(node,tm) {}
#endif

#endif
