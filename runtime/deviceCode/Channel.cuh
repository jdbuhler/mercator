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

#include "ChannelBase.cuh"

#include "Signal.cuh"

#include "options.cuh"

#include "Queue.cuh"

#include "support/collective_ops.cuh"

namespace Mercator  {
    
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
	  // allocate enough total buffer capacity to hold outputs 
	  // for one run() call
	  data = new T [numThreadGroups * numSlotsPerGroup];
	  
	  // verify that alloc succeeded
	  if (data == nullptr)
	    {
	      printf("ERROR: failed to allocate channel buffer [block %d]\n",
		     blockIdx.x);
	      
	      crash();
	    }
	  
	  dsQueue = nullptr;
	  dsSignalQueue = nullptr;
	  
	  for (unsigned int j = 0; j < numThreadGroups; j++)
	    nextSlot[j] = 0;
	}
    
    //
    // @brief Destructor.
    //
    __device__
      virtual
      ~Channel()
    {
      delete [] data;
    }
    
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
      void setDSEdge(NodeBase *idsNode, 
		     Queue<T> *idsQueue,
		     Queue<Signal> *idsSignalQueue,
		     unsigned int ireservedSlots)

    {
      dsNode = idsNode;
      dsQueue = idsQueue;
      dsSignalQueue = idsSignalQueue;
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
     
    //
    // @brief get the number of signals whose output could
    // be safely written to this channel's downstream signal queue.
    //
    __device__
      unsigned int dsSignalCapacity() const
    {
      return dsSignalQueue->getFreeSpace();
    }
    
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
    
    
    //
    // @brief After a call to run(), collect and write any data generated
    // to the downstream queue. NB: must be called with all threads
    //
    // RETURNS: true iff downstream node is active after copy
    //
    __device__
      bool moveOutputToDSQueue(unsigned int wtid)
    {
      int tid = threadIdx.x;
      
      // FIXME: can we ever call this function when the dsQueue is
      // null  (i.e. the channel is not connected to anything)? If
      // so, short-circuit here
      assert(dsQueue != nullptr);
      
      BlockScan<unsigned int, Props::THREADS_PER_BLOCK> scanner;
      unsigned int count = (tid < numThreadGroups ? nextSlot[tid] : 0);
      unsigned int agg;
      
      unsigned int sum = scanner.exclusiveSum(count, agg);
      
      __shared__ unsigned int dsBase;
      if ( IS_BOSS() )
	{
	  COUNT_ITEMS(agg);  // instrumentation
	  dsBase = directReserve(agg);
	    
	  // track produced items for cedit calculation
	  numItemsProduced += agg;
	}
      __syncthreads(); // all threads must see updates to dsBase
      
      // for each thread group, copy all generated outputs downstream
      if (tid < numThreadGroups)
	{
	  for (unsigned int j = 0; j < count; j++)
	    {
	      unsigned int srcOffset = tid * outputsPerInput + j;
	      unsigned int dstOffset = sum + j;
	      const T &myData = data[srcOffset];
	      directWrite(myData, dsBase, dstOffset);
	    }
	  
	  // clear nextSlot for this thread group
	  nextSlot[tid] = 0;
	}
      
      // If we've managed to fill the downstream queue, activate its
      // target node. Let our caller know if we activated the ds node.
      //
      if (dsQueue->getFreeSpace() < maxRunSize * outputsPerInput ||
	  dsSignalQueue->getFreeSpace() < MAX_SIGNALS_PER_RUN)
	{
	  if (IS_BOSS()) 
	    {
	      dsNode->activate();
	      
	      //stimcheck: In addition to activating, pass the writeThruId
	      dsNode->setWriteThruId(wtid);
	    }
	  
	  return true;
	}
      else
	{
	  return false;
	}
    }
    
    __device__
    bool checkDSFull()
    {
      return (dsQueue->getFreeSpace() < maxRunSize * outputsPerInput ||
	      dsSignalQueue->getFreeSpace() < MAX_SIGNALS_PER_RUN);
    }
    
    //
    // @brief prepare for a direct write to the downstream queue(s)
    // by reserving space for the items to write.
    //
    // @param number of slots to reserve for next write
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


    //
    // @brief push a signal to a specified channel, and reset the number
    // of items produced on that channel. This function is SINGLE THREADED.
    //
    // @param s the signal being sent downstream
    // @param channel the channel on which the signal is being sent
    //
    __device__
    void pushSignal(const Signal& s)
    {
      assert(dsSignalQueue->getFreeSpace() > 0);
      
      unsigned int credit = 
	(dsSignalQueue->empty()
	 ? dsSignalQueue->getOccupancy()
	 : numItemsProduced);
      
      Signal &sNew = dsSignalQueue->enqueue(s);
      sNew.setCredit(credit);
      
      numItemsProduced = 0;
    }
    
    //
    //
    //
    __device__
    bool isAggregate() const
    {
      return (propFlags & 0x01);
    }
    
    //
    //
    //
    __device__
    void setAggregate()
    {
      propFlags |= 0x01;
    }
    
  private:
    
    const unsigned int outputsPerInput;  // max # outputs per input to node
    const unsigned int numSlotsPerGroup; // # buffer slots/group in one run

    //
    // output buffer
    //
    
    T* data;                              // buffered output
    
    //
    // tracking data for usage of output buffer slots
    //
    
    // next buffer slot avail for thread to push output
    unsigned char nextSlot[numThreadGroups];
    
    
    // number of items produced since last signal
    unsigned int numItemsProduced;
    
    //
    // target (edge) for scattering items from output buffer
    //

    Queue<T> *dsQueue;
    Queue<Signal> *dsSignalQueue;
    NodeBase *dsNode;
    
    unsigned int reservedQueueEntries; // NB: will be used for cycles

    unsigned int propFlags;	//Signal propagation flags for this channel
  }; // end Channel class
}  // end Mercator namespace

#endif
