#ifndef __CHANNEL_CUH
#define __CHANNEL_CUH

//
// @file Channel.cuh
// @brief MERCATOR channel object
//
// MERCATOR
// Copyright (C) 2018 Washington University in St. Louis; all rights reserved.
//

#include <cassert>
#include <cstdio>

#include "ChannelBase.cuh"

#include "options.cuh"

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
    // @param reservedQueueEntries # queue entries reserved in 
    //          each ds instance
    //
    __device__
    Channel(unsigned int ioutputsPerInput,
	    const unsigned int *ireservedQueueEntries)
      : outputsPerInput(ioutputsPerInput),
        numSlotsPerGroup(numEltsPerGroup * outputsPerInput)
    {
      // allocate enough total buffer capacity to hold outputs 
      // for one run() call
      data = new T [numThreadGroups * numSlotsPerGroup];
      signalData = new Signal [numThreadGroups * numSlotsPerGroup];
      
      // verify that alloc succeeded
      if (data == nullptr)
	{
	  printf("ERROR: failed to allocate channel buffer [block %d]\n",
		 blockIdx.x);
	  
	  crash();
	}
      
      // verify that alloc succeeded
      if (signalData == nullptr)
	{
	  printf("ERROR: failed to allocate channel buffer (Signal) [block %d]\n",
		 blockIdx.x);
	  
	  crash();
	}

      for (unsigned int j = 0; j < numInstances; j++)
	{
	  dsQueues[j]      = nullptr;
	  dsSignalQueues[j]      = nullptr;
	  dsInstances[j]   = 0;
	  dsSignalInstances[j]   = 0;
	  reservedQueueEntries[j] = ireservedQueueEntries[j];
	  reservedSignalQueueEntries[j] = ireservedQueueEntries[j];
	}
      
      for (unsigned int j = 0; j < numThreadGroups; j++) {
	nextSlot[j] = 0;
	nextSignalSlot[j] = 0;
      }
    }
    
    //
    // @brief Destructor.
    //
    __device__
    virtual
    ~Channel()
    {
      delete [] data;
      delete [] signalData;
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
    
    // stimcheck: Signal equivalent of setDSEdge
    //
    // @brief Set the downstream (Signal) target of the edge for
    // instance usInstIdx of this channel to (dsModule, dsInstIdx)
    //
    // @param usInstIdx instance index of upstream edge endpoint
    // @param dsSignalQueue queue of downstream edge endpoint
    // @param dsInstIdx instance of downstream edge endpoint
    //
    __device__
    void setDSSignalEdge(unsigned int usInstIdx, 
		   Queue<Signal> *dsSignalQueue, unsigned int dsInstIdx)
    {
      dsSignalQueues[usInstIdx]    = dsSignalQueue;
      dsSignalInstances[usInstIdx] = dsInstIdx; 
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
      
      if (dsQueue == nullptr) // no outgoing edge
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
    
    //  stimcheck: Signal version of downstream capacity
    //
    //  @brief get the number of signals that can be safely
    //  consumed by the specified instance of this channel's
    //  module without overrunning the available downstream
    //  queue space.  Virtual because it requires access to
    //  the channel's queue, which does not have an untyped base.
    //
    // @param index of instance to check
    //
    __device__
    unsigned int dsSignalCapacity(unsigned int instIdx) const
    { 
      const Queue<Signal> *dsSignalQueue = dsSignalQueues[instIdx];
      int dsInstance          = dsInstances[instIdx];
      
      if (dsSignalQueue == nullptr) // no outgoing edge
	return UINT_MAX;
      
      // find ds queue's budget for this node
      unsigned int dsBudget = 
	dsSignalQueue->getCapacity(dsInstance) -
	dsSignalQueue->getOccupancy(dsInstance);
      
      // adjust budget based on reserved slots in ds queue (will be
      // nonzero for predecessors of head nodes of back edges, which
      // must reserve space for tail of back edge to fire).  Make
      // sure result is non-negative.
      dsBudget = max(0, dsBudget - reservedSignalQueueEntries[instIdx]);
      
      // capacity = floor(budget / outputs-per-input on channel)

      //stimcheck: Currently using the data outputPerInput var as a soft
      //upper bound on the number of outputs possible per signal handling.
      //Signals will be taken care of sequentially and scattered to their
      //respective queue after being "spent" (no credits remaining, and
      //downstream space is available).
      return dsBudget / outputsPerInput;
    }

    __device__
    bool dsSignalQueueHasPending(unsigned int instIdx) const {
	const Queue<Signal> *dsSignalQueue = dsSignalQueues[instIdx];
	int dsInstance = dsInstances[instIdx];
	if(dsSignalQueue == nullptr)
		return false;
	if(dsSignalQueue->getOccupancy(dsInstance) > 0)
		return true;
	return false;
    }


    __device__
    unsigned int dsPendingOccupancy(unsigned int instIdx) const {
	const Queue<T> *dsQueue = dsQueues[instIdx];
	int dsInstance = dsInstances[instIdx];
	unsigned int occ = 0;
	if(dsQueue == nullptr)
		return 0;
	else
		occ = dsQueue->getOccupancy(dsInstance);
	return occ;
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
    
    // stimcheck: Push for Signals
    //
    // @brief move items in each (live) thread to the output buffer
    // 
    // @param item item to be pushed
    // @param isWriter true iff thread is the writer for its group
    //
    __device__
      void pushSignal(const Signal &item, bool isWriter)
    {
      if (isWriter)
	{
	  int groupId = threadIdx.x / threadGroupSize;
	  
	  assert(nextSignalSlot[groupId] < numSlotsPerGroup);
	  
	  unsigned int slotIdx =
	    groupId * numSlotsPerGroup + nextSignalSlot[groupId];
	  
	  printf("PRE groupId: %d, slotIdx: %d, numSlotsPerGroup: %d, nextSignalSlot[groupId]: %d\n", groupId, slotIdx, numSlotsPerGroup, nextSignalSlot[groupId]);
	  signalData[slotIdx] = item;
	  
	  nextSignalSlot[groupId]++;

	  printf("groupId: %d, slotIdx: %d, numSlotsPerGroup: %d, nextSignalSlot[groupId]: %d\n", groupId, slotIdx, numSlotsPerGroup, nextSignalSlot[groupId]);
	}
    }

    __device__
    void resetNumProduced(unsigned int instIdx) {
	numItemsProduced[instIdx] = 0;
    }

    __device__
    unsigned int getNumItemsProduced(unsigned int instIdx) const {
	unsigned int total = numItemsProduced[instIdx];
	return total;
    }

    //
    // @brief After a call to run(), scatter its outputs
    //  to the appropriate queues.
    //  NB: must be called with all threads
    //
    // @param instIdx instance corresponding to current thread
    // @param isHead is this the first thread for its instance?
    // @param isWriter true iff thread is the writer for its group
    //
    __device__
      unsigned int scatterToQueues(InstTagT instIdx, bool isHead, bool isWriter)
    {
      unsigned int instTotal = 0;
      int tid = threadIdx.x;
      int groupId = tid / threadGroupSize;
  
      //
      // Compute a segmented exclusive sum of the number of outputs to
      // be written back to queues by each thread group.  Only the
      // writer threads contribute to the sums.
      //
      
      BlockSegScan<unsigned int, Props::THREADS_PER_BLOCK> scanner;
      
      unsigned int count = (isWriter ? nextSlot[groupId] : 0);
      unsigned int sum = scanner.exclusiveSumSeg(count, isHead);
      instTotal = sum;

      numItemsProduced[instIdx] += sum;
      
      //
      // Find the first and last thread for each instance.  Inputs
      // to one node are assigned to a contiguous set of threads.
      //
      
      BlockDiscontinuity<InstTagT, Props::THREADS_PER_BLOCK> disc;
      
      bool isTail = disc.flagTails(instIdx, NULLTAG);
        
      //
      // The last thread with a given instance can compute the total
      // number of outputs written for that instance.  That total
      // is used to reserve space in the instance's downstream queue. 
      //
      __shared__ unsigned int dsBase[numInstances];
      if (isTail && instIdx < numInstances)
	{
	  instTotal = sum + count; // exclusive -> inclusive sum
	  	  
	  COUNT_ITEMS(instTotal);  // instrumentation
	      
	  dsBase[instIdx] = directReserve(instIdx, instTotal);
	}
      
      __syncthreads(); // all threads must see updates to dsBase[]
      
      //stimcheck: Add to the current counter of number of items produced
      numItemsProduced[instIdx] += instTotal;

      __syncthreads(); // all threads must see the updates to the numItemsProduced

      //printf("NUMPRODUCED = %d\n", numItemsProduced[instIdx]);

      //
      // Finally, writer threads move the data to its queue.  We
      // take some loss of occupancy by looping over the outputs, 
      // but it saves us from having to tag each output with its
      // instance number (which would be needed if we tried to do
      // the writes using contiguous threads.)
      //
      
      if (isWriter)
	{
	  for (unsigned int j = 0; j < count; j++)
	    {
	      // where is the item in the ouput buffer?
	      unsigned int srcOffset = tid * outputsPerInput + j;
	      
	      // where is the item going in the ds queue?
	      unsigned int dstOffset = sum + j;
	      
	      if (instIdx < numInstances) // is this thread active?
		{
		  const T &myData = data[srcOffset];
		  
		  directWrite(instIdx, myData, dsBase[instIdx], dstOffset);
		}
	    }
	}
      
      // finally, reset the output counters per thread group
      if (tid < numThreadGroups)
	nextSlot[tid] = 0;

	//printf("DATA QUEUE OCCUPANCY: %d", dsQueue->getOccupancy(dsInstance));

      return instTotal;
    }
    
    //
    // @brief After a call to run(), scatter its outputs
    //  to the appropriate queues.
    //  NB: must be called with all threads
    //
    // @param instIdx instance corresponding to current thread
    // @param isHead is this the first thread for its instance?
    // @param isWriter true iff thread is the writer for its group
    //
    __device__
      void scatterToSignalQueues(InstTagT instIdx, bool isHead, bool isWriter)
    {
      int tid = threadIdx.x;
      int groupId = tid / threadGroupSize;
  
      //
      // Compute a segmented exclusive sum of the number of outputs to
      // be written back to queues by each thread group.  Only the
      // writer threads contribute to the sums.
      //
      
      BlockSegScan<unsigned int, Props::THREADS_PER_BLOCK> scanner;
      
      unsigned int count = (isWriter ? nextSignalSlot[groupId] : 0);
      unsigned int sum = scanner.exclusiveSumSeg(count, isHead);
      
      //
      // Find the first and last thread for each instance.  Inputs
      // to one node are assigned to a contiguous set of threads.
      //
      
      BlockDiscontinuity<InstTagT, Props::THREADS_PER_BLOCK> disc;
      
      bool isTail = disc.flagTails(instIdx, NULLTAG);
        
      //
      // The last thread with a given instance can compute the total
      // number of outputs written for that instance.  That total
      // is used to reserve space in the instance's downstream queue. 
      //
      __shared__ unsigned int dsBase[numInstances];
      if (isTail && instIdx < numInstances)
	{
	  unsigned int instTotal = sum + count; // exclusive -> inclusive sum
	  	  
	  COUNT_ITEMS(instTotal);  // instrumentation
	      
	  dsBase[instIdx] = directSignalReserve(instIdx, instTotal);
	}
      
      __syncthreads(); // all threads must see updates to dsBase[]
      
      //
      // Finally, writer threads move the data (Signal) to its queue.  We
      // take some loss of occupancy by looping over the outputs, 
      // but it saves us from having to tag each output with its
      // instance number (which would be needed if we tried to do
      // the writes using contiguous threads.)
      //
      
      if (isWriter)
	{
	  for (unsigned int j = 0; j < count; j++)
	    {
	      // where is the item in the ouput buffer?
	      unsigned int srcOffset = tid * outputsPerInput + j;
	      
	      // where is the item going in the ds queue?
	      unsigned int dstOffset = sum + j;
	      
	      if (instIdx < numInstances) // is this thread active?
		{
		  const Signal &myData = signalData[srcOffset];
		  
		  directSignalWrite(instIdx, myData, dsBase[instIdx], dstOffset);
		}
	    }
	}
      
      // finally, reset the output counters per thread group
      if (tid < numThreadGroups)
	nextSignalSlot[tid] = 0;
    }

    //
    // @brief prepare for a direct write to the downstream queue(s)
    // by reserving space for the items to write.
    //
    // @param instance of queue that we reserve in
    // @param number of slots to reserve for next write
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

    // stimcheck: Direct reserve of space for Signal queues
    //
    // @brief prepare for a direct write to the downstream signal queue(s)
    // by reserving space for the items to write.
    //
    // @param instance of queue that we reserve in
    // @param number of slots to reserve for next write
    // @return starting index of reserved segment.
    //
    __device__
    unsigned int directSignalReserve(unsigned int instIdx, 
			       unsigned int nToWrite) const
    {
      Queue<Signal> *dsSignalQueue   = dsSignalQueues[instIdx];
      InstTagT dsSignalInst     = dsSignalInstances[instIdx];
      
      return (dsSignalQueue ? dsSignalQueue->reserve(dsSignalInst, nToWrite) : 0);
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
    
    // stimcheck: Write Signals directly to the downstream queue
    //
    // @brief Write items directly to the downstream queue.
    //
    // @param instIdx of queue that we reserve in
    // @param item item to be written
    // @param base base pointer to writable block in queue
    // @param offset offset at which to write item
    //
    __device__
    void directSignalWrite(unsigned int instIdx, const Signal &item, 
		     unsigned int base,
		     unsigned int offset) const
    {
      Queue<Signal> *dsSignalQueue   = dsSignalQueues[instIdx];
      InstTagT dsInst     = dsSignalInstances[instIdx];
      
      if (dsSignalQueue)
	dsSignalQueue->putElt(dsInst, base, offset, item);
      //if(dsSignalQueue->getOccupancy(dsInst) > 1)
	//printf("DS SIGNAL QUEUE OVERSIZED: %d, dsInst: %d\n", dsSignalQueue->getOccupancy(dsInst), dsInst);
    }

  private:
    
    const unsigned int outputsPerInput;  // max # outputs per input to module
    const unsigned int numSlotsPerGroup; // # buffer slots/group in one run
    
    //
    // output buffer
    //
    
    T* data;                              // buffered output
    Signal* signalData;                   // buffered signal output
    
    //
    // tracking data for usage of output buffer slots
    //
    
    // next buffer slot avail for thread to push output
    unsigned int nextSlot[numThreadGroups];
    unsigned int nextSignalSlot[numThreadGroups];
    
    //
    // targets (edges) for scattering items from output buffer
    //
    
    Queue<T> *dsQueues[numInstances];
    InstTagT dsInstances[numInstances];
    
    Queue<Signal> *dsSignalQueues[numInstances];
    InstTagT dsSignalInstances[numInstances];

    // reserved ds queue entries on outgoing edge corresponding to
    // each instance of this channel.
    unsigned int reservedQueueEntries[numInstances];
    unsigned int reservedSignalQueueEntries[numInstances];


    //stimcheck: Counter for number of items produced between signals.
    // Used for setting right amount of credit for each signal.
    unsigned int numItemsProduced[numInstances];

  }; // end Channel class
}  // end Mercator namespace

#endif
