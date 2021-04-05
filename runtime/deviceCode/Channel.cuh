#ifndef __CHANNEL_CUH
#define __CHANNEL_CUH

//
// @file Channel.cuh
// @brief MERCATOR channel object
//
// MERCATOR
// Copyright (C) 2021 Washington University in St. Louis; all rights reserved.
//

#include "ChannelBase.cuh"

#include "options.cuh"

namespace Mercator  {
    
  //
  // @class Channel
  // @brief Holds all data associated with an output stream from a node.
  //
  // @tparam T type of object written to channel
  //
  template <typename T>
  class Channel : public ChannelBase {
    
  public:
    
    ///////////////////////////////////////////////////////
    // INIT/CLEANUP KERNEL FUNCIIONS
    ///////////////////////////////////////////////////////
    
    //
    // @brief Constructor (called single-threaded)
    //
    // @param minFreeSpace minimum space required for ds queue to be non-full
    //
    __device__
    Channel(unsigned int minFreeSpace, bool isAgg,
	    NodeBase *dsNode,
	    QueueBase *dsQueue, 
	    Queue<Signal> *dsSignalQueue)
      : ChannelBase(minFreeSpace, isAgg, dsNode, dsQueue, dsSignalQueue)
    {}
    
    ///////////////////////////////////////////////////////
    
    //
    // @brief Write items directly to the downstream queue.
    //
    // May be called MULTI-THREADED
    //
    // @param base base pointer to writable space in queue
    // @param offset offset at which to write item
    // @param item item to be written
    //
    __device__
    void dsWrite(size_t base,
		 unsigned int offset,
		 const T &item) const
    {
      static_cast<Queue<T>*>(dsQueue)->put(base, offset, item);
    }
  }; // end Channel class
}  // end Mercator namespace

#endif
