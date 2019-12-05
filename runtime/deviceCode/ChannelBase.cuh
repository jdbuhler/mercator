#ifndef __CHANNEL_BASE_CUH
#define __CHANNEL_BASE_CUH

//
// @file ChannelBase.cuh
// @brief Base class of MERCATOR channel object
//
// MERCATOR
// Copyright (C) 2019 Washington University in St. Louis; all rights reserved.
//

#include "NodeBase.cuh"

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
    ChannelBase() {}
    
    __device__
    virtual
    ~ChannelBase() {}
    
    //
    // @brief get # of inputs whose outputs can safely be written
    // to channel's downstream queue
    //
    
    __device__
    virtual
    unsigned int dsCapacity() const = 0;

    //
    // @brief check whether the downstream queue has enough space to
    // hold all outputs produced by 'size' inputs.
    //
    
    __device__
    virtual
    bool checkDSFull(unsigned int size) const = 0;
    
#ifdef INSTRUMENT_COUNTS
    // counts outputs on channel
    ItemCounter itemCounter;
#endif

  };
}   // end Mercator namespace

#endif
