#ifndef __ITEM_COUNTER_H
#define __ITEM_COUNTER_H

//
// ITEM_COUNTER.H
// Instrumentation to count itms into/out of a node
//
// MERCATOR
// Copyright (C) 2019 Washington University in St. Louis; all rights reserved.
//

#ifdef INSTRUMENT_COUNTS

struct ItemCounter {
  
  __device__
  ItemCounter()
    : count(0)
  {}
  
  // multithreaded increment, using numInstances threads
  __device__
  void incr(unsigned long long c)
  { count += c; }
  
  unsigned long long count;
};

#endif

#ifdef INSTRUMENT_COUNTS
#define COUNT_ITEMS(n)    { itemCounter.incr(n); }
#else
#define COUNT_ITEMS(n)    { }
#endif

#endif
