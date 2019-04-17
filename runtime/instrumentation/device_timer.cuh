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
      nextStamp = 0;
      totalStampsTaken=0;
      for (int i=0; i<INSTRUMENT_FG_TIME;i++){
        fineArr[i]=DevClockT(0);
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
  DevClockT getTimeArrayElm(unsigned int i) const
  { return fineArr[i]; }

  __device__
  int getTotalStampsTaken() const{
    return totalStampsTaken;
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
  
    #ifdef INSTRUMENT_FG_TIME
      int max=INSTRUMENT_FG_TIME;
    #else
      int max=0;
    #endif
    __syncthreads();
    if (IS_BOSS())
      {
      totalStampsTaken++;
	DevClockT now = clock64();
        if (nextStamp>=max){
          nextStamp=0;
        }
	fineArr[nextStamp] = timeDiff(fineStart, now);
        nextStamp++;
      }
  }
private:

  DevClockT totalTime;  
  DevClockT lastStart;
  DevClockT fineStart;
  int totalStampsTaken;
  int nextStamp;
  #ifdef INSTRUMENT_FG_TIME //no need to waste space if we arnt gonna use it
  DevClockT fineArr[INSTRUMENT_FG_TIME]; 
  #else
  DevClockT fineArr[1]; 
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
