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
#include <cstdio>

#include <cooperative_groups.h>

#include "ChannelBase.cuh"

#include "options.cuh"

#include "Queue.cuh"

#include "support/collective_ops.cuh"

namespace Mercator  {
  
  using namespace cooperative_groups;
  
  //
  // @class Channel
  // @brief Holds all data associated with an output stream from a node.
  //
  template <typename Props>
  template <typename T>
  class Node<Props>::Channel final 
    : public Node<Props>::ChannelBase {

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
      : outputsPerInput(ioutputsPerInput),
      numSlotsPerGroup(numEltsPerGroup * outputsPerInput)
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
    // @brief return the node at the other end of this channel's queue
    //
    __device__
      NodeBase* getDSNode() const
    {
      return dsNode;
    }
    
    
    //
    // @brief Set the downstream target of the edge for
    // this channel.
    //
    // @param idsNode downstream node
    //
    __device__
      void setDSEdge(NodeBase *idsNode, Queue<T> *idsQueue,
		     unsigned int ireservedSlots)
    {
      dsNode = idsNode;
      dsQueue = idsQueue;
      reservedQueueEntries = ireservedSlots;
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
	  dsBase = directReserve(g.size());
	}
      
      directWrite(item, g.shfl(dsBase, 0), g.thread_rank());
    }
    
    __device__
      bool checkDSFull() const
    {
            // If we've managed to fill the downstream queue, activate its
      // target node. Let our caller know if we activated the ds node.
      //
      if (dsQueue->getFreeSpace() < maxRunSize * outputsPerInput)
	{
	  if (IS_BOSS())
	    dsNode->activate();
	  
	  return true;
	}
      else
	return false;
    }
    

#if 0    
    //
    // @brief move items in each (live) thread to the output buffer
    // 
    // @param item item to be pushed
    // @param isWriter true iff thread is the writer for its group
    //
    __device__
      void push(const T &item, bool isWriter)
    {
      if (isWriter)
	{
	  int groupId = threadIdx.x / threadGroupSize;
	  
	  assert(nextSlot[groupId] < numSlotsPerGroup);
	  
	  unsigned int slotIdx =
	    groupId * numSlotsPerGroup + nextSlot[groupId];
	  
	  data[slotIdx] = item;
	  
	  nextSlot[groupId]++;
	}
    }
#endif

    //
    // @brief prepare for a direct write to the downstream queue(s)
    // by reserving space for the items to write.
    //
    // @param number of slots to reserve for next write9
    // @return starting index of reserved segment.
    //
    __device__
      unsigned int directReserve(unsigned int nToWrite) const
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
      void directWrite(const T &item, 
		       unsigned int base,
		       unsigned int offset) const
    {
      dsQueue->putElt(base, offset, item);
    }
    
  private:
    
    const unsigned int outputsPerInput;  // max # outputs per input to node
    const unsigned int numSlotsPerGroup; // # buffer slots/group in one run
    
    //
    // target (edge) for scattering items from output buffer
    //

    Queue<T> *dsQueue;
    NodeBase *dsNode;
    unsigned int reservedQueueEntries; // NB: will be used for cycles
  }; // end Channel class
}  // end Mercator namespace

#endif
