#ifndef __OCC_COUNTER_CUH
#define __OCC_COUNTER_CUH

//
// OCC_COUNTER.H
// Occupancy tracking statistics
//
// MERCATOR
// Copyright (C) 2018 Washington University in St. Louis; all rights reserved.
//

#include "options.cuh"

#ifdef INSTRUMENT_OCC

/**
 * @class OccCounter
 *
 * @brief Structure to hold occupancy data for node firings
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
  void add_run(unsigned long long nElements)
  {
    if (IS_BOSS())
      {
	totalInputs += nElements;
	
	totalRuns++;
	
	totalFullRuns += (nElements == sizePerRun);
      }
  }
     
};

#endif

#ifdef INSTRUMENT_OCC
#define NODE_OCC_COUNT(n) { occCounter.add_run(n); }
#else
#define NODE_OCC_COUNT(n) {}
#endif

#endif
