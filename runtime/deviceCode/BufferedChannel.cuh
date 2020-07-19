#ifndef __BUFFEREDCHANNEL_CUH
#define __BUFFEREDCHANNEL_CUH

//
// @file BufferedChannel.cuh
// @brief MERCATOR channel object with a buffer to allow push() with 
// a subset of threads
//
// MERCATOR
// Copyright (C) 2020 Washington University in St. Louis; all rights reserved.
//

#include <cassert>
#include <cstdio>

#include "BufferedChannelBase.cuh"

#include "options.cuh"

#include "support/collective_ops.cuh"

namespace Mercator  {
    
  //
  // @class BufferedChannel
  // @brief Holds all data associated with an output stream from a node.
  //
  template <typename T,
	    unsigned int THREADS_PER_BLOCK>
  class BufferedChannel : public BufferedChannelBase {
    
    using ChannelBase::dsReserve;
    using ChannelBase::numItemsWritten;
    
  public:

    ///////////////////////////////////////////////////////
    // INIT/CLEANUP KERNEL FUNCIIONS
    ///////////////////////////////////////////////////////
    
    //
    // @brief Constructor (called single-threaded)
    //
    // @param ioutputsPerInput Outputs per input for this channel
    //
    __device__
    BufferedChannel(unsigned int outputsPerInput, bool isAgg,
		    unsigned int numThreadGroups,
		    unsigned int threadGroupSize,
		    unsigned int numEltsPerGroup)
      : BufferedChannelBase(outputsPerInput, isAgg, 
			    numThreadGroups, threadGroupSize, numEltsPerGroup),
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
    }
    
    __device__
    virtual
    ~BufferedChannel()
    { 
      delete [] data; 
    }
    
    /////////////////////////////////////////////////////////////
    
    //
    // @brief push item to per-thread output buffer
    // May be called MULTI-THREADED, but does not synchronize
    //
    // @param item item to write to buffer
    //
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
    
    
    //
    // @brief move any pushed data from output buffer to downstream queue
    // MUST BE CALLED WITH ALL THREADS
    //
    __device__
    void completePush()
    {
      int tid = threadIdx.x;
      
      BlockScan<unsigned int, THREADS_PER_BLOCK> scanner;
      unsigned int count = (tid < numThreadGroups ? nextSlot[tid] : 0);
      unsigned int totalToWrite;
      
      unsigned int dsOffset = scanner.exclusiveSum(count, totalToWrite);
      
      // BEGIN WRITE dsBase, ds queue, numItemsWritten, nextSlot
      __syncthreads(); 

      // clear nextSlot for this thread group, since we're done with it
      if (tid < numThreadGroups)
	nextSlot[tid] = 0;
      
      __shared__ unsigned int dsBase;
      if (IS_BOSS())
	{
	  dsBase = dsReserve(totalToWrite);
	  
	  // track produced items for credit calculation
	  numItemsWritten += totalToWrite;
	}
      
      // END WRITE dsBase, ds queue numItemsWritten, nextSlot
      __syncthreads(); 
      
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
	}
    }
  
  private:    
    
    const unsigned int numSlotsPerGroup;
    
    T *data;
    
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
    void dsWrite(unsigned int base,
		 unsigned int offset,
		 const T &item) const
    {
      static_cast<Queue<T>*>(dsQueue)->putElt(base, offset, item);
    }
    
  }; // end BufferedChannel class
}  // end Mercator namespace

#endif
