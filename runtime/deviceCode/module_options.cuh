#ifndef __MODULE_OPTIONS_CUH
#define __MODULE_OPTIONS_CUH

//
// MODULE_OPTIONS.CUH
//
// These timer calls count time only when a module is
// not in the tail of its execution
//
// MERCATOR
// Copyright (C) 2018 Washington University in St. Louis; all rights reserved.
//

#include "options.cuh"

#ifndef INSTRUMENT_TAIL
#define MOD_TIMER_START(tm)			\
  { if (!this->isInTail()) { TIMER_START(tm); } }
#else
#define MOD_TIMER_START(tm) \
  { TIMER_START(tm); }
#endif

#ifndef INSTRUMENT_TAIL
#define MOD_TIMER_STOP(tm)			\
  { if (!this->isInTail()) { TIMER_STOP(tm); } }
#else
#define MOD_TIMER_STOP(tm) \
  { TIMER_STOP(tm); }
#endif

#ifndef INSTRUMENT_TAIL
#define MOD_FINE_TIMER_START(tm)			\
  { if (!this->isInTail()) { FINE_TIMER_START(tm); } }
#else
#define MOD_FINE_TIMER_START(tm) \
  { FINE_TIMER_START(tm); }
#endif

#ifndef INSTRUMENT_TAIL
#define MOD_FINE_TIMER_STOP(tm)			\
  { if (!this->isInTail()) { FINE_TIMER_STOP(tm); } }
#else
#define MOD_FINE_TIMER_STOP(tm) \
  { FINE_TIMER_STOP(tm); }
#endif

#ifndef INSTRUMENT_TAIL
#define MOD_OCC_COUNT(n)                        \
  { if (!this->isInTail()) { OCC_COUNT(n); } }
#else
#define MOD_OCC_COUNT(n)                        \
  { OCC_COUNT(n); }
#endif

#endif
