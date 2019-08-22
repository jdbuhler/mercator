#ifndef __ITEM_COUNTER_H
#define __ITEM_COUNTER_H

//
// ITEM_COUNTER.H
// Instrumentation to count itms into/out of a module
//
// MERCATOR
// Copyright (C) 2018 Washington University in St. Louis; all rights reserved.
//

#ifdef INSTRUMENT_COUNTS

template <int NUM_INSTS>
struct ItemCounter {
  
  __device__
  ItemCounter()
  {
    for (int j = 0; j < NUM_INSTS; j++)
      counts[j] = 0;
  }
  
  // multithreaded increment, using numInstances threads
  __device__
  void incr(unsigned long long count)
  {
    incrByInst(threadIdx.x, count);
  }
  
  __device__
  void incrByInst(int instIdx, unsigned long long count)
  {
    assert(instIdx < NUM_INSTS);
    counts[instIdx] += count;
  }
  
  // multithreaded increment of a single value.
  // We use atomic to let all threads that want
  // to do the increment do it concurrently.
  __device__
  void incrSingle(int i, bool doIncr)
  {
    if (doIncr)
      atomicAdd(&(counts[i]), 1ULL);
  }
  
  unsigned long long counts[NUM_INSTS];
};

#endif

#ifdef INSTRUMENT_COUNTS
#define COUNT_ITEMS(n)         { itemCounter.incr(n); }
#define COUNT_ITEMS_INST(i, c) { itemCounter.incrByInst(i,c); }
#define COUNT_SINGLE(i,b) { itemCounter.incrSingle(i,b); }
#else
#define COUNT_ITEMS(n)    { }
#define COUNT_ITEMS_INST(n, i) { }
#define COUNT_SINGLE(i,b) { }
#endif

#endif
