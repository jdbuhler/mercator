#ifndef __OUT_DIST_COUNTER_CUH
#define __OUT_DIST_COUNTER_CUH

//
// OUT_DIST_COUNTER.H
// Output distribution tracking statistics
//
// MERCATOR
// Copyright (C) 2018 Washington University in St. Louis; all rights reserved.
//

#include "options.cuh"

#ifdef INSTRUMENT_OUT_DIST

/**
 * @class OutDistCounter
 *
 * @brief Structure to hold output distribution data for node firings
 */
struct OutDistCounter {
  
  unsigned long long distribution[OUT_DIST_MAX];
  unsigned long long tempOut;
  
  __device__
  OutDistCounter():
	  tempOut(0)
  {}

  __device__
  void add_output(unsigned int nElements)
  {
    if (IS_BOSS())
      {
        tempOut += nElements;
      }
  }
  
  __device__
  void reset()
  {
    if (IS_BOSS())
      {
        tempOut = 0;
      }
  }

  __device__
  void finalize()
  {
    if (IS_BOSS())
      {
        distribution[tempOut] += 1;
      }
  }


};

#endif

#ifdef INSTRUMENT_OUT_DIST
#define CHANNEL_OUT_COUNT(n) { channel->outDistCounter.add_output(n); }
#else
#define CHANNEL_OUT_COUNT(n) {}
#endif

#ifdef INSTRUMENT_OUT_DIST
#define CHANNEL_OUT_RESET() { using Channel = Channel<void*>; for(unsigned int tempc = 0; tempc < numChannels; ++tempc) { Channel *channel = static_cast<Channel*>(node->getChannel(tempc)); channel->outDistCounter.reset(); } }
#else
#define CHANNEL_OUT_RESET() {}
#endif

#ifdef INSTRUMENT_OUT_DIST
#define CHANNEL_OUT_FINALIZE() { using Channel = Channel<void*>; for(unsigned int tempc = 0; tempc < numChannels; ++tempc) { Channel *channel = static_cast<Channel*>(node->getChannel(tempc)); channel->outDistCounter.finalize(); } }
#else
#define CHANNEL_OUT_FINALIZE() {}
#endif

#endif
