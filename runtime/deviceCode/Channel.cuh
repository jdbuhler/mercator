#ifndef __CHANNEL_CUH
#define __CHANNEL_CUH

//
// @file Channel.cuh
// @brief MERCATOR channel object
//
// MERCATOR
// Copyright (C) 2020 Washington University in St. Louis; all rights reserved.
//

#include "ChannelBase.cuh"

#include "options.cuh"

namespace Mercator  {
    
  //
  // @class Channel
  // @brief Holds all data associated with an output stream from a node.
  //
  template <typename T>
  class Channel : public ChannelBase {
    
  public:

    ///////////////////////////////////////////////////////
    // INIT/CLEANUP KERNEL FUNCIIONS
    ///////////////////////////////////////////////////////
    
    //
    // @brief Constructor (called single-threaded)
    //
    // @param minFreeSpace minimum space required for ds queue to be non-full
    //
    __device__
    Channel(unsigned int minFreeSpace, bool isAgg)
      : ChannelBase(minFreeSpace, isAgg)
    {}
    
    ///////////////////////////////////////////////////////
    
    //
    // @brief move items in each of first totalToWrite threads to the
    // output buffer. MUST BE CALLED WITH ALL THREADS.
    // 
    // @param item item to be pushed
    // @param totalToWrite total number of items to be written
    //
    __device__
    void pushCount(const T &item, unsigned int totalToWrite)
    {
      int tid = threadIdx.x;
      
      __syncthreads(); // BEGIN WRITE dsBase, ds queue
      
      __shared__ unsigned int dsBase;
      if (IS_BOSS())
	{
	  dsBase = dsReserve(totalToWrite);
	  
	  // track produced items for credit calculation
	  numItemsWritten += totalToWrite;
	}
      
      __syncthreads(); // END WRITE dsBase, ds queue
      
      if (tid < totalToWrite)
	dsWrite(dsBase, tid, item);
    }
    
  private:
        
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
    
  }; // end Channel class
}  // end Mercator namespace

#endif
