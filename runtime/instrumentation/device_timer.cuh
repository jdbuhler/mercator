#ifndef __DEVICE_TIMER_H
#define __DEVICE_TIMER_H

#include "options.cuh"

#ifdef INSTRUMENT_TIME

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
#define TIMER_START(tm) { (tm ## Timer).start(); } 
#else
#define TIMER_START(tm) {}
#endif

#ifdef INSTRUMENT_TIME
#define TIMER_STOP(tm) { (tm ## Timer).stop(); } 
#else
#define TIMER_STOP(tm) {}
#endif

#endif
