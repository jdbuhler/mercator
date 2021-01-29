#ifndef __OPTIONS_CUH
#define __OPTIONS_CUH

// @file options.cuh
// @brief Define execution options for MERCATOR
//
// MERCATOR
// Copyright (C) 2020 Washington University in St. Louis; all rights reserved.
//

//
// Enable device instrumenation
//

// instrument # of trips through scheduling loop?
//#define INSTRUMENT_SCHED_COUNTS

// instrument code timing for performance model parameters?
//#define INSTRUMENT_TIME

// instrument code for input counts and occupancy of each node firing?
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

// enable debug printing
//#define PRINTDBG




//
// Misc runtime options
//
//how many blocks to run with
//#define USE_ONE_BLOCKS
//#define USE_X_BLOCKS 8
//#define USE_SM_BLOCKS
#define USE_MAX_BLOCKS

//use random thread to record timings 
//#define RAND_TIMING_THREAD

#endif
