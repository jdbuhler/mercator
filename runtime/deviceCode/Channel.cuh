#ifndef __CHANNEL_CUH
#define __CHANNEL_CUH

//
// @file Channel.cuh
// @brief MERCATOR channel object
//
// MERCATOR
// Copyright (C) 2019 Washington University in St. Louis; all rights reserved.
//

#include <cassert>

#include <cooperative_groups.h>

#include "ChannelBase.cuh"
#include "NodeBase.cuh"
#include "Queue.cuh"

#include "options.cuh"

namespace Mercator  {
  
  using namespace cooperative_groups;
  
  //
  // @class Channel
  // @brief Holds all data associated with an output stream from a node.
  //
  template <typename T>
  class Channel final 
    : public ChannelBase {

#ifdef INSTRUMENT_COUNTS    
    using ChannelBase::itemCounter;
#endif
    
  public:
    
    //
    // @brief Constructor (called single-threaded)
    //
    // @param ioutputsPerInput Outputs per input for this channel
    //
    __device__
      Channel(unsigned int ioutputsPerInput)
      : outputsPerInput(ioutputsPerInput)
	{
	  dsQueue = nullptr;
	}
    
    //
    // @brief Destructor.
    //
    __device__
      virtual
      ~Channel()
    {}
    
    
    //
    // @brief Set the downstream target of the edge for
    // this channel.
    //
    // @param idsQueue downstream edge's queue
    //
    __device__
      void setDSQueue(Queue<T> *idsQueue)
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
     
    
    __device__
      void push(const T &item)
    {
      coalesced_group g = coalesced_threads();
      
      unsigned int dsBase;
      if (g.thread_rank() == 0)
	{
	  COUNT_ITEMS(g.size());  // instrumentation
	  dsBase = dsReserve(g.size());
	}
      
      dsWrite(item, g.shfl(dsBase, 0), g.thread_rank());
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
    
    //
    // @brief prepare for a direct write to the downstream queue(s)
    // by reserving space for the items to write.
    //
    // @param number of slots to reserve for next write9
    // @return starting index of reserved segment.
    //
    __device__
      unsigned int dsReserve(unsigned int nToWrite) const
    {
      return dsQueue->reserve(nToWrite);
    }
    
    
    //
    // @brief Write items directly to the downstream queue.
    //
    // @param item item to be written
    // @param base base pointer to writable block in queue
    // @param offset offset at which to write item
    //
    __device__
      void dsWrite(const T &item, 
		   unsigned int base,
		   unsigned int offset) const
    {
      dsQueue->putElt(base, offset, item);
    }
    
  private:
    
    const unsigned int outputsPerInput;  // max # outputs per input to node
    
    //
    // target (edge) for scattering items from output buffer
    //

    Queue<T> *dsQueue;
    
  }; // end Channel class
}  // end Mercator namespace

#endif
