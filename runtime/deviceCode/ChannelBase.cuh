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
    

    __device__
    virtual 
    unsigned int getGain()const=0;


     //
    //  @brief get the downstream module of the
    //  the specified instance of this channel's
    //  module . Virtual because it requires access to
    //  the channel's queue, which does not have an untyped base.
    //

    __device__
    virtual 
    ModuleTypeBase* getDSModule(unsigned int instIdx) const = 0;

    __device__
    virtual
    InstTagT getDSInstance(unsigned int instIdx) const=0;
    //
    //  @brief get the number of inputs that can be safely be
    //  consumed by the specified instance of this channel's
    //  module without overrunning the available downstream
    //  queue space.  Virtual because it requires access to
    //  the channel's queue, which does not have an untyped base.
    //
    __device__
    virtual
    unsigned int dsCapacity(unsigned int) const = 0;
    
    //
    // @brief After a call to run(), scatter its outputs
    //  to the appropriate queues.
    //  NB: must be called with all threads
    //
    __device__
    virtual
    bool  compressCopyToDSQueue(InstTagT, bool, bool) = 0;

    __device__
    virtual
    void scatterToQueues(InstTagT, bool, bool) = 0;
#ifdef INSTRUMENT_COUNTS
    // counts outputs on channel, accessed by ModuleType
    ItemCounter<numInstances> itemCounter;
#endif

  };
}   // end Mercator namespace

#endif
