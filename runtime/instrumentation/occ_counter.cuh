#ifndef __OCC_COUNTER_CUH
#define __OCC_COUNTER_CUH

#include "options.cuh"

#ifdef INSTRUMENT_OCC

/**
 * @class OccCounter
 *
 * @brief Structure to hold occupancy data for module firings.
 */
struct OccCounter {
  
  unsigned long long totalInputs;
  unsigned long long totalRuns;
  unsigned long long totalFullRuns;

  unsigned long long sizePerRun;
  
  __device__
  OccCounter()
    : totalInputs(0), 
      totalRuns(0), 
      totalFullRuns(0),
      sizePerRun(0)
  {}
  
  __device__
  void setMaxRunSize(unsigned long long isizePerRun)
  { sizePerRun = isizePerRun; }
  
  __device__
  void add_firing(unsigned long long nElements)
  {
    if (IS_BOSS())
      {
	totalInputs += nElements;
	
	totalRuns += (nElements + sizePerRun - 1)/ sizePerRun;
	
	totalFullRuns += nElements / sizePerRun;
      }
  }
     
};

#endif

#ifdef INSTRUMENT_OCC
#define OCC_COUNT(n) { occCounter.add_firing(n); }
#else
#define OCC_COUNT(n) {}
#endif

#endif
