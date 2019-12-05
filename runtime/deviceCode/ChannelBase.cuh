#ifndef __CHANNEL_BASE_CUH
#define __CHANNEL_BASE_CUH

//
// @file ChannelBase.cuh
// @brief Base class of MERCATOR channel object
//
// MERCATOR
// Copyright (C) 2019 Washington University in St. Louis; all rights reserved.
//

#include "QueueBase.cuh"

#include "options.cuh"

#include "instrumentation/item_counter.cuh"

namespace Mercator  {

  //
  // @class ChannelBase
  // @brief Base class for an output channel of a node.  This is
  // a pure interface class, so that we can mix channels with different
  // properties in a node's channels[] array.
  //
  class ChannelBase {
  public:
    
    __device__
    ChannelBase(unsigned int ioutputsPerInput)
      : outputsPerInput(ioutputsPerInput),
	dsQueue(nullptr)
    {}
    
    __device__
    virtual
    ~ChannelBase() {}

    //
    // @brief Set the downstream target of the edge for
    // this channel.
    //
    // @param idsQueue downstream edge's queue
    //
    __device__
    void setDSQueue(const QueueBase *idsQueue)
    {
      dsQueue = idsQueue;
    }


    //
    // @brief get the number of inputs whose output could
    // be safely written to this channel's downstream queue.
    //
    __device__
      unsigned int dsCapacity() const
    {
      return dsQueue->getFreeSpace() / outputsPerInput;
    }
    
    //
    // @brief determine whether the downstream queue for this channel
    // has enough space to hold the max possible # of outputs that
    // could be produced by 'size' inputs.
    //
    __device__
      bool checkDSFull(unsigned int size) const
    {
      return (dsQueue->getFreeSpace() < size * outputsPerInput);
    }
    
#ifdef INSTRUMENT_COUNTS
    // counts outputs on channel
    ItemCounter itemCounter;
#endif
    
  private:
    
    const unsigned int outputsPerInput;  // max # outputs per input to node
    const QueueBase *dsQueue;            // ptr to ds queue
	
  };
}   // end Mercator namespace

#endif
