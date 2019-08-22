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
      
      // verify that alloc succeeded
      if (data == nullptr)
	{
	  printf("ERROR: failed to allocate channel buffer [block %d]\n",
		 blockIdx.x);
	  
	  crash();
	}
      
      for (unsigned int j = 0; j < numInstances; j++)
	{
	  dsQueues[j]      = nullptr;
	  dsInstances[j]   = 0;
	  reservedQueueEntries[j] = ireservedQueueEntries[j];
	}
      
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
    //  @brief return the moduleid of the downsteam module along this channel
    //
    //

    __device__
    ModuleTypeBase* getDSModule(unsigned int instIdx) const{
      return dsQueues[instIdx]->getAssocatedModule();
    }
  
    __device__
    InstTagT getDSInstance(unsigned int instIdx) const{
      return dsInstances[instIdx];
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
    // @brief After a call to run(), scatter its outputs
    //  to the appropriate queues.
    //  NB: must be called with all threads
    //
    // @param instIdx instance corresponding to current thread
    // @param isHead is this the first thread for its instance?
    // @param isWriter true iff thread is the writer for its group
    //
    #ifdef SCHEDULER_MINSWITCHES
    //TODO:: simplfy this a lot cause we arnt pulling from multiple queue
    __device__
      bool  scatterToQueues(InstTagT instIdx, bool isHead, bool isWriter)
    {
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
	  	  
	  COUNT_ITEMS_INST(instIdx, instTotal);  // instrumentation
          #ifdef PRINTDBG
            printf("\t\tChannel total dumping:%u\n", instTotal); 
          #endif
	  dsBase[instIdx] = directReserve(instIdx, instTotal);
	}
      
      __syncthreads(); // all threads must see updates to dsBase[]
      
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
    
      __syncthreads();

      //TODO:: check this more agressivly 
      //check if there is enough space for a full warp to be fired again
      unsigned int dsInstId = dsInstances[instIdx];
      unsigned int queueCap = dsQueues[instIdx]->getCapacity(dsInstId);
      unsigned int queueOcc = dsQueues[instIdx]->getOccupancy(dsInstId);
      unsigned int dsQueue_rem = queueCap - queueOcc;


      //return if i am allowed to fire again
      if(dsQueue_rem >= (maxRunSize*outputsPerInput)){//it is safe to fire again 
        //printf("there is %u slots in ds queue\n", dsQueue_rem);
        return true;
      }
      
      //not enough space, so lets go ahead and activate the ds node
      //only one thread activates the ds modules
  
      ModuleTypeBase* dsModule = dsQueues[instIdx]->getAssocatedModule();
      dsModule->activate(dsInstId);
      return false;
    }
    #else

    __device__
      void scatterToQueues(InstTagT instIdx, bool isHead, bool isWriter)
    {
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
	      
	  dsBase[instIdx] = directReserve(instIdx, instTotal);
	}
      
      __syncthreads(); // all threads must see updates to dsBase[]
      
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
    }
#endif
    
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
  
    
    __device__ 
    unsigned int getGain()const
    {return outputsPerInput;}
      
  private:
    
    const unsigned int outputsPerInput;  // max # outputs per input to module
    const unsigned int numSlotsPerGroup; // # buffer slots/group in one run
    
    //
    // output buffer
    //
    
    T* data;                              // buffered output
    
    //
    // tracking data for usage of output buffer slots
    //
    
    // next buffer slot avail for thread to push output
    unsigned int nextSlot[numThreadGroups];
    
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
