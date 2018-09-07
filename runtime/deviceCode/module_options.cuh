#ifndef __MODULE_OPTIONS_CUH
#define __MODULE_OPTIONS_CUH

#include "options.cuh"

//
// these timer calls count time only when a module is
// not in the tail of its execution
//

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
#define MOD_OCC_COUNT(n)                        \
  { if (!this->isInTail()) { OCC_COUNT(n); } }
#else
#define MOD_OCC_COUNT(n)                        \
  { OCC_COUNT(n); }
#endif

#endif
