#ifndef __TIMING_OPTIONS_CUH
#define __TIMING_OPTIONS_CUH

//
// TIMING_OPTIONS.CUH
//
// These timer calls count time only when a node is
// not in flushing mode, so always consuming full ensembles.
//
// MERCATOR
// Copyright (C) 2019 Washington University in St. Louis; all rights reserved.
//

#include "options.cuh"

#ifndef INSTRUMENT_TAIL
#define MOD_TIMER_START(tm)			\
  { if (!this->isFlushing) { TIMER_START(tm); } }
#else
#define MOD_TIMER_START(tm) \
  { TIMER_START(tm); }
#endif

#ifndef INSTRUMENT_TAIL
#define MOD_TIMER_STOP(tm)			\
  { if (!this->isFlushing) { TIMER_STOP(tm); } }
#else
#define MOD_TIMER_STOP(tm) \
  { TIMER_STOP(tm); }
#endif

#ifndef INSTRUMENT_TAIL
#define MOD_OCC_COUNT(n)                        \
  { if (!this->isFlushing) { OCC_COUNT(n); } }
#else
#define MOD_OCC_COUNT(n)                        \
  { OCC_COUNT(n); }
#endif

#endif
