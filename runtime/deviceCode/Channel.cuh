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
    
    //
    // @brief Constructor (called single-threaded)
    //
    // @param minFreeSpace minimum space required for ds queue to be non-full
    //
    __device__
    Channel(unsigned int minFreeSpace, bool isAgg)
      : ChannelBase(minFreeSpace, isAgg)
    {}
    
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
      
      __syncthreads(); // protect use of dsBase from any later write
    }
    
  protected:
    
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
