#ifndef __CHANNEL_BASE_CUH
#define __CHANNEL_BASE_CUH

//
// @file ChannelBase.cuh
// @brief Base class of MERCATOR channel object
//
// MERCATOR
// Copyright (C) 2019 Washington University in St. Louis; all rights reserved.
//

#include "options.cuh"

#include "instrumentation/item_counter.cuh"

namespace Mercator  {

  //
  // @class ChannelBase
  // @brief Base class for an output channel of a node.  This is
  // a pure interface class, so that we can mix channels with different
  // properties in a node's channels[] array.
  //
  template <class Props>
  class Node<Props>::ChannelBase {
  public:
    
    __device__
    ChannelBase() {}
    
    __device__
    virtual
    ~ChannelBase() {}
    
    //
    // @brief get the downstream node associated with this channel.
    // Virtual because it requires access to the channel's queue,
    // which does not have an untyped base.
    //

    __device__
    virtual 
    NodeBase* getDSNode() const = 0;
    
    //
    // @brief get # of inputs whose outputs can safely be written
    // to channel's downstream queue
    //
    
    __device__
    virtual
    unsigned int dsCapacity() const = 0;

    //
    // @brief get # of signals whose outputs can safely be written
    // to channel's downstream signal queue
    //
    
    __device__
    virtual
    unsigned int dsSignalCapacity() const = 0;
        

    //
    // @brief After a call to run(), move node's output
    // from its temporary buffer to its downstream queue.
    // NB: must be called with all threads
    //
    __device__
    virtual
    bool moveOutputToDSQueue(unsigned int wtid) = 0;
    
    __device__
    virtual
    void pushSignal(const Signal &s) = 0;
    
    //
    //
    //
    __device__
    virtual
    bool isAggregate() const = 0;
    
    //
    //
    //
    __device__
    virtual
    void setAggregate() = 0;
    
#ifdef INSTRUMENT_COUNTS
    // counts outputs on channel
    ItemCounter itemCounter;
#endif

  };
}   // end Mercator namespace

#endif
