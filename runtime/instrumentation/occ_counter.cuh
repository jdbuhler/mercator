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
  
  __device__
  OccCounter()
    : totalInputs(0), 
      totalRuns(0), 
      totalFullRuns(0)
  {}
  
  __device__
  void add_run(unsigned int nElements, unsigned int vectorWidth)
  {
    if (IS_BOSS())
      {
	totalInputs += nElements;
	
	totalRuns++;
	
	totalFullRuns += (nElements == vectorWidth);
      }
  }
     
};

#endif

#ifdef INSTRUMENT_OCC
#define NODE_OCC_COUNT(n, w) { node->occCounter.add_run(n, w); }
#else
#define NODE_OCC_COUNT(n, w) {}
#endif

#endif
