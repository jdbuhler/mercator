#ifndef __FINE_SCHEDULE_COUNTER_CUH
#define __FINE_SCHEDULE_COUNTER_CUH

//
// OCC_COUNTER.H
// Occupancy tracking statistics
//
// MERCATOR
// Copyright (C) 2018 Washington University in St. Louis; all rights reserved.
//

#include "options.cuh"

#ifdef INSTRUMENT_FINE_SCHEDULER_CALLS

/**
 * @class OccCounter
 *
 * @brief Structure to hold occupancy data for node firings
 */
struct FineScheduleCounter {
  
  unsigned long long totalSchedulerCalls;
  unsigned long long totalDSActivations;
  unsigned long long totalFullConsumptions;
  unsigned long long totalDSandFull;
  
  __device__
  FineScheduleCounter()
    : totalSchedulerCalls(0),
      totalDSActivations(0),
      totalFullConsumptions(0),	
      totalDSandFull(0)	
  {}
  
  __device__
  void add_schedule()
  {
    if (IS_BOSS())
      {
	      ++totalSchedulerCalls;
      }
  }

  __device__
  void add_dsact(bool b)
  {
    if (IS_BOSS())
      {
	      if(b)
	      ++totalDSActivations;
      }
  }

  __device__
  void add_fullcon(bool b)
  {
    if (IS_BOSS())
      {
	      if(b)
	      ++totalFullConsumptions;
      }
  }

  __device__
  void add_dsact_fullcon(bool b)
  {
    if (IS_BOSS())
      {
	      if(b)
	      ++totalDSandFull;
      }
  }
     
};

#endif

#ifdef INSTRUMENT_FINE_SCHEDULER_CALLS
#define FINE_SCHEDULE_ADD() { this->schCounter.add_schedule(); }
#define FINE_SCHEDULE_ADD_DSACTIVE(b) { this->schCounter.add_dsact(b); }
#define FINE_SCHEDULE_ADD_FULLCONSUMPTION(b) { this->schCounter.add_fullcon(b); }
#define FINE_SCHEDULE_ADD_DSANDFULL(b) { this->schCounter.add_dsact_fullcon(b); }
#else
#define FINE_SCHEDULE_ADD() {}
#define FINE_SCHEDULE_ADD_DSACTIVE(b) {}
#define FINE_SCHEDULE_ADD_FULLCONSUMPTION(b) {}
#define FINE_SCHEDULE_ADD_DSANDFULL(b) {}
#endif

#endif
