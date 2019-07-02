#ifndef __CHANNEL_BASE_CUH
#define __CHANNEL_BASE_CUH

//
// @file ChannelBase.cuh
// @brief Base class of MERCATOR channel object
//
// MERCATOR
// Copyright (C) 2018 Washington University in St. Louis; all rights reserved.
//

#include "options.cuh"

#include "Signal.cuh"

#include "instrumentation/item_counter.cuh"

namespace Mercator  {

  //
  // @class ChannelBase
  // @brief Base class for an output channel of a module.  This is
  // a pure interface class, so that we can mix channels with different
  // properties in a module's channels[] array.
  //
  template <class Props>
  class ModuleType<Props>::ChannelBase {
  public:
    
    __device__
    ChannelBase() {}
    
    __device__
    virtual
    ~ChannelBase() {}
    
    //
    //  @brief get the number of inputs that can be safely
    //  consumed by the specified instance of this channel's
    //  module without overrunning the available downstream
    //  queue space.  Virtual because it requires access to
    //  the channel's queue, which does not have an untyped base.
    //
    __device__
    virtual
    unsigned int dsCapacity(unsigned int) const = 0;
    
    //  stimcheck: Signal version of downstream capacity
    //
    //  @brief get the number of signals that can be safely
    //  consumed by the specified instance of this channel's
    //  module without overrunning the available downstream
    //  queue space.  Virtual because it requires access to
    //  the channel's queue, which does not have an untyped base.
    //
    __device__
    virtual
    unsigned int dsSignalCapacity(unsigned int) const = 0;


    __device__
    virtual
    bool dsSignalQueueHasPending(unsigned int instIdx) const = 0;

    __device__
    virtual
    unsigned int dsPendingOccupancy(unsigned int instIdx) const = 0;

    __device__
    virtual
    void resetNumProduced(unsigned int instIdx) = 0;

    __device__
    virtual
    unsigned int getNumItemsProduced(unsigned int instIdx) const = 0;

    //
    // @brief After a call to run(), scatter its outputs
    //  to the appropriate queues.
    //  NB: must be called with all threads
    //
    __device__
    virtual
    void scatterToQueues(InstTagT, bool, bool) = 0;
    
    // stimcheck: Signal version of scattering to queues
    // Used when sending all buffered signals downstream
    //
    // @brief After a call to run(), scatter its outputs
    //  to the appropriate signal queues.
    //  NB: must be called with all threads
    //
    __device__
    virtual
    void scatterToSignalQueues(InstTagT, bool, bool) = 0;

    // 
    // @brief Check if this channel is an aggregate channel
    //
    __device__
    virtual
    bool isAggregate() const = 0;

    // 
    // @brief Check if this channel is an aggregate channel
    //
    __device__
    virtual
    void setAggregate() = 0;

#ifdef INSTRUMENT_COUNTS
    // counts outputs on channel, accessed by ModuleType
    ItemCounter<numInstances> itemCounter;
#endif

  };
}   // end Mercator namespace

#endif
