#ifndef __MAXVECTORGAIN_DIST_COUNTER_CUH
#define __MAXVECTORGAIN_DIST_COUNTER_CUH

//
// MAXVECTORGAIN_DIST_COUNTER.H
// Maximum gain distribution of single thread in every vector tracking statistics
//
// MERCATOR
// Copyright (C) 2018 Washington University in St. Louis; all rights reserved.
//

#include "options.cuh"

#ifdef INSTRUMENT_MAXVECTORGAIN_DIST

/**
 * @class MaxVectorGainCounter
 *
 * @brief Structure to hold distribution of maximum single gain over all threads in a vector for node firings
 */
struct MaxVectorGainCounter {
  
  //TODO: MAKE THE COUNTER WORK FOR BUFFERED NODES
  unsigned long long distribution[MAXVECTORGAIN_DIST_MAX];
  unsigned long long tempOut;
  
  __device__
  MaxVectorGainCounter():
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

#ifdef INSTRUMENT_MAXVECTORGAIN_DIST
#define CHANNEL_MAXVECTORGAIN_COUNT(n) { channel->maxVectorGainDistCounter.add_output(n); }
#else
#define CHANNEL_MAXVECTORGAIN_COUNT(n) {}
#endif

#ifdef INSTRUMENT_MAXVECTORGAIN_DIST
#define CHANNEL_MAXVECTORGAIN_RESET() { using Channel = Channel<void*>; for(unsigned int tempc = 0; tempc < numChannels; ++tempc) { Channel *channel = static_cast<Channel*>(node->getChannel(tempc)); channel->maxVectorGainDistCounter.reset(); } }
#else
#define CHANNEL_MAXVECTORGAIN_RESET() {}
#endif

#ifdef INSTRUMENT_MAXVECTORGAIN_DIST
#define CHANNEL_MAXVECTORGAIN_FINALIZE() { using Channel = Channel<void*>; for(unsigned int tempc = 0; tempc < numChannels; ++tempc) { Channel *channel = static_cast<Channel*>(node->getChannel(tempc)); channel->maxVectorGainDistCounter.finalize(); } }
#else
#define CHANNEL_MAXVECTORGAIN_FINALIZE() {}
#endif

#endif
