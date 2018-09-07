#ifndef __CHANNEL_BASE_CUH
#define __CHANNEL_BASE_CUH
//
// @file ChannelBase.cuh
// @brief Base class of MERCATOR channel object
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
    
    //
    //  @brief get the number of inputs that can be safely be
    //  consumed by the specified instance of this channel's
    //  module without overrunning the available downstream
    //  queue space.  Virtual because it requires access to
    //  the channel's queue, which does not have an untyped base.
    //
    __device__
    virtual
    unsigned int dsCapacity(unsigned int instIdx) const = 0;
    
    //
    // @brief After we finish a call to run() during firing, prepare
    // to receive outptus from the next run.
    //
    __device__
    virtual
    void finishRun() = 0;

    //
    // @brief When all runs in a firing are complete, remove items
    //  from the output buffer and place them in the appropriate queues
    //
    __device__
    virtual
    void scatterToQueues() = 0;
    
#ifdef INSTRUMENT_COUNTS
    // counts outputs on channel, accessed by ModuleType
    ItemCounter<numInstances> itemCounter;
#endif

  };
}   // end Mercator namespace

#endif
