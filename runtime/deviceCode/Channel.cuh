#ifndef __CHANNEL_CUH
#define __CHANNEL_CUH

//
// @file Channel.cuh
// @brief MERCATOR channel object
//
// MERCATOR
// Copyright (C) 2020 Washington University in St. Louis; all rights reserved.
//

#include <cassert>
#include <cstdio>

#include "ChannelBase.cuh"

#include "options.cuh"

#include "support/collective_ops.cuh"

namespace Mercator  {
    
  //
  // @class Channel
  // @brief Holds all data associated with an output stream from a node.
  //
  template <typename T,
	    unsigned int THREADS_PER_BLOCK>
  class Channel : public ChannelBase {
    
  public:
    
    //
    // @brief Constructor (called single-threaded)
    //
    // @param ioutputsPerInput Outputs per input for this channel
    //
    __device__
    Channel(unsigned int outputsPerInput, 
	    unsigned int inumThreadGroups,
	    unsigned int ithreadGroupSize,
	    unsigned int numEltsPerGroup,
	    bool isAgg)
      : ChannelBase(outputsPerInput, isAgg),
	numThreadGroups(inumThreadGroups),
	threadGroupSize(ithreadGroupSize),
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
      
      nextSlot = new unsigned char [numThreadGroups];
      for (unsigned int j = 0; j < numThreadGroups; j++)
	nextSlot[j] = 0;
    }
    
    __device__
    virtual
    ~Channel()
    { 
      delete [] nextSlot; 
      delete [] data; 
    }
    
    __device__
    void push(const T &item)
    {
      int groupId = threadIdx.x / threadGroupSize;
      
      assert(nextSlot[groupId] < numSlotsPerGroup);
      
      unsigned int slotIdx =
	groupId * numSlotsPerGroup + nextSlot[groupId];
      
      data[slotIdx] = item;
      
      nextSlot[groupId]++;
    }
    
    __device__
    void completePush()
    {
      int tid = threadIdx.x;
      
      BlockScan<unsigned int, THREADS_PER_BLOCK> scanner;
      unsigned int count = (tid < numThreadGroups ? nextSlot[tid] : 0);
      unsigned int totalToWrite;
      
      unsigned int dsOffset = scanner.exclusiveSum(count, totalToWrite);
      
      __shared__ unsigned int dsBase;
      if ( IS_BOSS() )
	{
	  dsBase = dsReserve(totalToWrite);
	  
	  // track produced items for credit calculation
	  numItemsWritten += totalToWrite;
	}
      __syncthreads(); // all threads must see updates to dsBase
      
      // for each thread group, copy all generated outputs downstream
      if (tid < numThreadGroups)
        {
          for (unsigned int j = 0; j < count; j++)
            {
              unsigned int srcOffset = tid * outputsPerInput + j;
              unsigned int dstIdx = dsOffset + j;
              const T &myData = data[srcOffset];
	      dsWrite(dsBase, dstIdx, myData);
	    }

          // clear nextSlot for this thread group
          nextSlot[tid] = 0;
        }
    }
    
    
    //
    // @brief move items in each of first totalToWrite threads to the
    // output buffer
    // 
    // @param item item to be pushed
    // @param totalToWrite total number of items to be written
    //
    __device__
    void pushCount(const T &item, unsigned int totalToWrite)
    {
      int tid = threadIdx.x;
      
      __shared__ unsigned int dsBase;
      if ( IS_BOSS() )
	{
	  dsBase = dsReserve(totalToWrite);
	  
	  // track produced items for credit calculation
	  numItemsWritten += totalToWrite;
	}
      __syncthreads(); // all threads must see updates to dsBase
      
      if (tid < totalToWrite)
	dsWrite(dsBase, tid, item);
    }
    
  private:    
    
    const unsigned int numThreadGroups;
    const unsigned int threadGroupSize;
    const unsigned int numSlotsPerGroup;
    
    T *data;
    unsigned char *nextSlot;
    
    //
    // @brief prepare for a direct write to the downstream queue(s)
    // by reserving space for the items to write.
    //
    // @param number of slots to reserve for next write
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
    void dsWrite(unsigned int base,
		 unsigned int offset,
		 const T &item) const
    {
      static_cast<Queue<T>*>(dsQueue)->putElt(base, offset, item);
    }
    
  }; // end Channel class
}  // end Mercator namespace

#endif
