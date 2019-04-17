#ifndef __OPTIONS_CUH
#define __OPTIONS_CUH

// @file options.cuh
// @brief Define execution options for MERCATOR
//
// MERCATOR
// Copyright (C) 2018 Washington University in St. Louis; all rights reserved.
//

//
// Enable device instrumenation
//

// instrument code timing for performance model parameters?
//#define INSTRUMENT_TIME

// instrument code for fine grained timing of nodes
// this is actually overloaded to also represent the size of the 
// storage array to make per node 
#define INSTRUMENT_FG_TIME 500

// instrument code for performance model parameters?
//#define INSTRUMENT_COUNTS

// instrument code for occupancy of each module firing?
// NB: For unknown reasons, in some cases running with INSTRUMENT_TIME
//       set to 1 causes interference with this option.
//#define INSTRUMENT_OCC

// collect data from tail of app's execution?
// 1 = collect data from all firings
// 0 = collect data from all firings until tail is reached
#define INSTRUMENT_TAIL


//
// Enable host instrumenation
//

// time from host side only using CUDA timer API
#define INSTRUMENT_TIME_HOST

// profile code for time using nvprof?
//#define PROFILE_TIME

// print info about mem allocs?
//#define PRINT_MEM_USAGE


//
// Misc runtime options
//

// try to avoid calling run() on a module when there are not enough
// inputs to fill all threads
#define PREFER_FULL_ENSEMBLES true

// Scheduling algorithms
#define SCHEDULER_MAXOCC    // choose queue with max fireable occupancy
//#define SCHEDULER_LOTTERY // choose queue probabilistically

#endif
