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
#define NODE_TIMER_START(node,tm)			\
  { if (!(node)->isFlushing()) { XNODE_TIMER_START(node,tm); } }
#define TIMER_START(tm)			\
  { if (!this->isFlushing()) { XNODE_TIMER_START(this,tm); } }
#else
#define NODE_TIMER_START(node,tm)		\
  { XNODE_TIMER_START(node,tm); }
#define TIMER_START(tm)		\
  { XNODE_TIMER_START(this,tm); }
#endif

#ifndef INSTRUMENT_TAIL
#define NODE_TIMER_STOP(node,tm)			\
  { if (!(node)->isFlushing()) { XNODE_TIMER_STOP(node,tm); } }
#define TIMER_STOP(tm)			\
  { if (!this->isFlushing()) { XNODE_TIMER_STOP(this,tm); } }
#else
#define NODE_TIMER_STOP(node,tm)		\
  { XNODE_TIMER_STOP(node,tm); }
#define TIMER_STOP(tm)		\
  { XNODE_TIMER_STOP(this,tm); }
#endif



#ifndef INSTRUMENT_TAIL
#define OCC_COUNT(n, w)				\
  { if (!node->isFlushing()) { NODE_OCC_COUNT(n, w); } }
#else
#define OCC_COUNT(n, w)				\
  { NODE_OCC_COUNT(n, w); }
#endif

#endif
