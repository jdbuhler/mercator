#ifndef __CHANNELBUFFER_CUH
#define __CHANNELBUFFER_CUH

//
// @file ChannelBuffer.cuh
// @brief MERCATOR channel buffer object to allow push() with 
// a subset of threads
//
// MERCATOR
// Copyright (C) 2020 Washington University in St. Louis; all rights reserved.
//

#include <cstdio>
#include <cassert>

#include "ChannelBufferBase.cuh"
#include "Channel.cuh"

#include "options.cuh"

#include "support/collective_ops.cuh"

namespace Mercator  {
    
  //
  // @class ChannelBuffer
  // @brief Buffers data sent to an output channel during a single run
  //
  template <typename T,
	    unsigned int THREADS_PER_BLOCK>
  class ChannelBuffer : public ChannelBufferBase {
    
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
    ChannelBuffer(unsigned int ioutputsPerInput,
		  unsigned int inumThreadGroups,
		  unsigned int ithreadGroupSize,
		  unsigned int numEltsPerGroup)
      : ChannelBufferBase(inumThreadGroups),
	outputsPerInput(ioutputsPerInput),
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
    }
    
    __device__
    ~ChannelBuffer()
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
    void store(const T &item)
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
    void finishWrite(ChannelBase *ichannel)
    {
      int tid = threadIdx.x;
      
      using Channel = Channel<T>;
      Channel *channel = static_cast<Channel*>(ichannel);
      
      BlockScan<unsigned int, THREADS_PER_BLOCK> scanner;
      unsigned int count = (tid < numThreadGroups ? nextSlot[tid] : 0);
      unsigned int totalToWrite;
      
      unsigned int dsOffset = scanner.exclusiveSum(count, totalToWrite);
      
      // BEGIN WRITE dsBase, ds queue, nextSlot
      __syncthreads(); 

      // clear nextSlot for this thread group, since we're done with it
      if (tid < numThreadGroups)
	nextSlot[tid] = 0;
      
      __shared__ size_t dsBase;
      if (IS_BOSS())
	dsBase = channel->dsReserve(totalToWrite);
      
      // END WRITE dsBase, ds queue, nextSlot
      __syncthreads(); 
      
      // for each thread group, copy all generated outputs downstream
      if (tid < numThreadGroups)
        {
          for (unsigned int j = 0; j < count; j++)
            {
              unsigned int srcOffset = tid * outputsPerInput + j;
              unsigned int dstIdx = dsOffset + j;
              const T &myData = data[srcOffset];
	      channel->dsWrite(dsBase, dstIdx, myData);
	    }
	}
    }
    
  private:    
    
    const unsigned int outputsPerInput;
    const unsigned int numThreadGroups;
    const unsigned int threadGroupSize;
    
    const unsigned int numSlotsPerGroup;
    
    T *data;
    
  }; // end ChannelBuffer class
}  // end Mercator namespace

#endif
