#ifndef __BUFFEREDCHANNEL_CUH
#define __BUFFEREDCHANNEL_CUH

//
// @file Channel.cuh
// @brief MERCATOR channel object
//
// MERCATOR
// Copyright (C) 2020 Washington University in St. Louis; all rights reserved.
//

#include <cassert>
#include <cstdio>

#include "Channel.cuh"

#include "options.cuh"

#include "support/collective_ops.cuh"

namespace Mercator  {
    
  //
  // @class BufferedChannel
  // @brief Holds all data associated with an output stream from a node.
  //
  template <typename T,
	    unsigned int THREADS_PER_BLOCK>
  class BufferedChannel : public Channel<T> {
    
    using BaseType = Channel<T>;
    
    using BaseType::dsReserve;
    using BaseType::dsWrite;
    using BaseType::numItemsWritten;
    
  public:
    
    //
    // @brief Constructor (called single-threaded)
    //
    // @param ioutputsPerInput Outputs per input for this channel
    //
    __device__
    BufferedChannel(unsigned int ioutputsPerInput, bool isAgg,
		    unsigned int inumThreadGroups,
		    unsigned int ithreadGroupSize,
		    unsigned int numEltsPerGroup)
      : BaseType(inumThreadGroups * numEltsPerGroup * ioutputsPerInput, isAgg),
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
      
      nextSlot = new unsigned char [numThreadGroups];
      for (unsigned int j = 0; j < numThreadGroups; j++)
	nextSlot[j] = 0;
    }
    
    __device__
    virtual
    ~BufferedChannel()
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
      
      __syncthreads(); // protect use of dsBase from any later write
    }
  
  private:    
    
    const unsigned int outputsPerInput;
    const unsigned int numThreadGroups;
    const unsigned int threadGroupSize;
    const unsigned int numSlotsPerGroup;
    
    T *data;
    unsigned char *nextSlot;
  }; // end BufferedChannel class
}  // end Mercator namespace

#endif
