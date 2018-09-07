#ifndef __CHANNEL_CUH
#define __CHANNEL_CUH
//
// @file Channel.cuh
// @brief MERCATOR channel object
//

#include <cassert>
#include <cstdio>

#include "ChannelBase.cuh"

#include "options.cuh"

#include "mapqueues/scatter.cuh"

#include "Queue.cuh"

namespace Mercator  {
    
  //
  // @class Channel
  // @brief Holds all data associated with an output stream for a module.
  // Contains parallel arrays for data items and instance tags of items.
  //
  template <typename Props>
  template <typename T>
  class ModuleType<Props>::Channel final 
    : public ModuleType<Props>::ChannelBase {

#ifdef INSTRUMENT_COUNTS    
    using ChannelBase::itemCounter;
#endif
    
  public:
    
    //
    // @brief Constructor (called single-threaded)
    //
    // @param ioutputsPerInput Outputs per input for this channel
    // @param maxRunsPerFiring how many times might we need to push on
    //        each thread (ie,how many calls to run) in one firing?
    // @param reservedQueueEntries # queue entries reserved in 
    //          each ds instance
    //
    __device__
    Channel(unsigned int ioutputsPerInput,
	    unsigned int maxRunsPerFiring,
	    const unsigned int *ireservedQueueEntries)
      : outputsPerInput(ioutputsPerInput),
        numSlotsPerGroup(numEltsPerGroup * outputsPerInput),
        slotsUsedPerRun(numThreadGroups * numSlotsPerGroup)
    {
      // allocate enough total buffer capacity to hold outputs for all
      // run() calls in one firing.
      unsigned int capacity = slotsUsedPerRun * maxRunsPerFiring;
      
      data = new T        [capacity];
      tags = new InstTagT [capacity];
      
      for(unsigned int i = 0; i < capacity; ++i)
	tags[i] = NULLTAG;
      
      for (unsigned int j = 0; j < numInstances; j++)
	{
	  dsQueues[j]      = nullptr;
	  dsInstances[j]   = 0;
	  reservedQueueEntries[j] = ireservedQueueEntries[j];
	}
      
      for (unsigned int j = 0; j < numThreadGroups; j++)
	nextSlot[j] = 0;
      
      slotsUsed = 0;
    }
    
    //
    // @brief Destructor.
    //
    __device__
    virtual
    ~Channel()
    {
      delete [] data;
      delete [] tags;
    }
    
    
    //
    // @brief Set the downstream target of the edge for
    // instance usInstIdx of this channel to (dsModule, dsInstIdx)
    //
    // @param usInstIdx instance index of upstream edge endpoint
    // @param dsQueue queue of downstream edge endpoint
    // @param dsInstIdx instance of downstream edge endpoint
    //
    __device__
    void setDSEdge(unsigned int usInstIdx, 
		   Queue<T> *dsQueue, unsigned int dsInstIdx)
    {
      dsQueues[usInstIdx]    = dsQueue;
      dsInstances[usInstIdx] = dsInstIdx; 
    }
    
    
    //
    // @brief get the number of inputs that can be safely be
    // consumed by the specified instance of this channel's
    // module without overrunning the available downstream
    // queue space.
    //
    // @param index of instance to check
    //
    __device__
    unsigned int dsCapacity(unsigned int instIdx) const
    { 
      const Queue<T> *dsQueue = dsQueues[instIdx];
      int dsInstance          = dsInstances[instIdx];
      
      if (dsQueue == nullptr) // no outgoign edge
	return UINT_MAX;
      
      // find ds queue's budget for this node
      unsigned int dsBudget = 
	dsQueue->getCapacity(dsInstance) -
	dsQueue->getOccupancy(dsInstance);
      
      // adjust budget based on reserved slots in ds queue (will be
      // nonzero for predecessors of head nodes of back edges, which
      // must reserve space for tail of back edge to fire).  Make
      // sure result is non-negative.
      dsBudget = max(0, dsBudget - reservedQueueEntries[instIdx]);
      
      // capacity = floor(budget / outputs-per-input on channel)
      return dsBudget / outputsPerInput;
    }
    
    
    //
    // @brief move items in each (live) thread to the output buffer
    // 
    // @param item item to be pushed
    // @param tag instance tag of item to be pushed
    //
    __device__
    void push(const T &item, InstTagT tag)
    {
      int groupId = threadIdx.x / threadGroupSize;
      
      assert(nextSlot[groupId] < numSlotsPerGroup);
      
      unsigned int slotIdx =
	slotsUsed + groupId * numSlotsPerGroup + nextSlot[groupId];
      
      data[slotIdx] = item;
      tags[slotIdx] = tag;
      
      nextSlot[groupId]++;
    }
    

    //
    // @brief After a call to run, move to next chunk of output buffer
    //  and reset next slot counters.
    //
    __device__
    void finishRun()
    {
      if (IS_BOSS())
	slotsUsed += slotsUsedPerRun;
      
      if (threadIdx.x < numThreadGroups)
	nextSlot[threadIdx.x] = 0;
    }
    
    
    //
    // @brief Remove items from channel and place in appropriate queues.
    // Calls directReserve/Write internally
    //
    __device__
    void scatterToQueues()
    { 
      int tid = threadIdx.x;
      
      // main loop: stride through channel in block-sized chunks
      for (unsigned int base = 0; 
	   base < slotsUsed;
	   base += Props::THREADS_PER_BLOCK)
	{
	  unsigned int myIdx = base + tid;
	  
	  // get instance tag of my slot
	  InstTagT itemTag = 
	    (myIdx < slotsUsed
	     ? tags[myIdx]
	     : NULLTAG);
	  
	  // nullify used positions in Channel for next round
	  if (itemTag != NULLTAG)
	    tags[myIdx] = NULLTAG;
	  
	  // parallel sort vars
	  unsigned short idx;
	  unsigned short offset;
	  __shared__ unsigned short aggs[numInstances];
	  
	  // determine for each item with a valid tag where it goes in
	  // its target queue.  While we're at it, calculate the
	  // total number of items going onto each instance's queue.
	  using Scatter = QueueScatter<numInstances, Props::THREADS_PER_BLOCK>;
	  
	  offset = Scatter::WarpSortAndComputeOffsets(itemTag, idx, aggs);
	  
	  __shared__ unsigned int dsBase[numInstances];
	  
	  if (tid < numInstances) 
	    {
	      // reserve space in all downstream queues
	      
	      COUNT_ITEMS(aggs[tid]);  // instrumentation
	      
	      dsBase[tid] = directReserve(tid, aggs[tid]);
	    }
	  
	  // make sure all threads see queue state after reservation
	  __syncthreads();  
	  
	  if (itemTag < numInstances)  // if new item valid
	    {
	      // write data to all downstream queues
	      
	      const T &myData = data[base + idx];
	      
	      directWrite(itemTag, myData, dsBase[itemTag], offset);
	    }
	}
      
      slotsUsed = 0; // reset for next firing
    }  
    
    
    //
    // @brief prepare for a direct write to the downstream queue(s)
    // by reserving space for the items to write.
    //
    // @param instance of queue that we reserve in
    // @param number of slots to reserve for next write9
    // @return starting index of reserved segment.
    //
    __device__
    unsigned int directReserve(unsigned int instIdx, 
			       unsigned int nToWrite) const
    {
      Queue<T> *dsQueue   = dsQueues[instIdx];
      InstTagT dsInst     = dsInstances[instIdx];
      
      return (dsQueue ? dsQueue->reserve(dsInst, nToWrite) : 0);
    }
    
    
    //
    // @brief Write items directly to the downstream queue.
    //
    // @param instIdx of queue that we reserve in
    // @param item item to be written
    // @param base base pointer to writable block in queue
    // @param offset offset at which to write item
    //
    __device__
    void directWrite(unsigned int instIdx, const T &item, 
		     unsigned int base,
		     unsigned int offset) const
    {
      Queue<T> *dsQueue   = dsQueues[instIdx];
      InstTagT dsInst     = dsInstances[instIdx];
      
      if (dsQueue)
	dsQueue->putElt(dsInst, base, offset, item);
    }
    
  private:

    const unsigned int outputsPerInput;  // max # outputs per input to module
    const unsigned int numSlotsPerGroup; // # buffer slots/group in one run
    const unsigned int slotsUsedPerRun;  // max # outputs from one run() call
    
    //
    // output buffer
    //
    
    T*               data;     // buffered output
    InstTagT *       tags;     // node tag associated with each output
    
    //
    // tracking data for usage of output buffer slots
    //
    
    // next buffer slot avail for thread to push output
    unsigned int nextSlot[numThreadGroups];
    
    unsigned int slotsUsed; // how many slots consumed in this firing?
    
    //
    // targets (edges) for scattering items from output buffer
    //
    
    Queue<T> *dsQueues[numInstances];
    InstTagT dsInstances[numInstances];
    
    // reserved ds queue entries on outgoing edge corresponding to
    // each instance of this channel.
    unsigned int reservedQueueEntries[numInstances];

  }; // end Channel class
}  // end Mercator namespace

#endif
