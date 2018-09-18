#ifndef __GATHER_CUH
#define __GATHER_CUH

//
// GATHER.CUH
// Routines to assign indices for gathering inputs from queues
//
// MERCATOR
// Copyright (C) 2018 Washington University in St. Louis; all rights reserved.
//

#include <cstdint>

#include <cub/cub.cuh>

namespace Mercator {

  // class QueueGather
  // Compute the information needed to collect data from a number of different
  // queues for provision to a module's code.  See the documentation of
  // the two public functions below for details.
  //
  // Template parameters:
  //   NQUEUES -- number of queues from which to collect; must be at
  //     most ARCH_WARP_SIZE
  //   ARCH_WARP_SIZE -- number of threads per warp
  //
  // These functions must be called with an integral number of active
  // warps.
  //
  template <uint8_t NQUEUES,
	    unsigned int ARCH_WARP_SIZE = 32>
  class QueueGather {
  private:
  
    // mask for all threads in a warp
    static const unsigned int WARP_MASK = ((1ULL << ARCH_WARP_SIZE) - 1ULL);
  
    //
    // warpLowerBound()
    // For each input thread, compute the index of the least entry in Ai
    // (a warp-width array stored in per-thread registers) that is >= the
    // thread ID, modified by an optional offset.  Every active warp is
    // assumed to have its own copy of Ai.
    //
    __device__ static unsigned int
    warpLowerBound(unsigned int Ai, unsigned int idx)
    {
      unsigned int currPos = ARCH_WARP_SIZE/2; // middle of warp

      for (int stride = currPos/2; stride > 0; stride >>= 1)
	{
	  currPos += (idx >= __shfl_sync(WARP_MASK, Ai, currPos) 
		      ? stride : -stride);
	}
    
      return currPos - (idx < __shfl_sync(WARP_MASK, Ai, currPos));
    }

  public:
  
    //
    // loadExclSums()
    // Load the exclusive progressive sum of A, which has NQUEUES <= WARP_SIZE
    // entries, into registers in each warp.  For threads >= NQUEUES in each
    // warp, set the value to UINT_MAX to act as boundary conditions for binary
    // search.  Note that all warps get the same values, so that we can do the 
    // lower bound computations needed for gathering in each warp independently.
    // 
    // Also return the sum of A's values in agg, which is per-thread.
    //
    __device__ static unsigned int 
    loadExclSums(unsigned int A0, unsigned int &agg)
    {
      static_assert(NQUEUES <= ARCH_WARP_SIZE, 
		    "too many queues for loadExclSums");
    
      // put the values of A0 into a shared buffer so we can 
      // bounce them // out to all warps
      __shared__ unsigned int A[NQUEUES];
    
      if (threadIdx.x < NQUEUES)
	A[threadIdx.x] = A0;
    
      __syncthreads();
    
      unsigned int laneId = cub::LaneId(); // == threadIdx % WARP_SIZE
    
      unsigned int Ai = (laneId < NQUEUES ? A[laneId] : 0);
    
      if (NQUEUES == 1)
	{
	  agg = A[0];
	}	
      else
	{
	  using Scan = WarpScan<unsigned int, ARCH_WARP_SIZE>;
	  
	  Ai = Scan::exclusiveSum(Ai, agg);
	}
    
      return (laneId < NQUEUES ? Ai : UINT_MAX);
    }
  
    //
    // BlockComputeQueues()
    // INPUTS: 
    //   - in each warp, an array A[] of ARCH_WARP_SIZE nondecreasing values
    //     (thread i of each warp holds A[i]); least value is assumed = 0
    //   - an index value idx (in each thread)
    // RETURNS:
    //   For each idx,
    //    + the least i s.t. A[i] <= idx , in "queue"
    //    + the difference between idx and A[queue], in "offset"
    //
    // EXAMPLE:
    // A[] = [0  4  6  7  11, UINT_MAX, ...]
    // idx = 3  --> queue = 0, offset = 3
    // idx = 8  --> queue = 3, offset = 1
    // idx = 12 --> queue = 4, offset = 1
    //
    // INTENDED USAGE: A[] is the progressive exclusive sum of the
    // numbers of elements in each of up to ARCH_WARP_SIZE
    // queues. Suppose we number elements globally, with the elts in
    // each queue numbered consecutively, starting from 0.  Given idx,
    // "queue" is the number of the queue in which the idxth element
    // occurs, and "offset" is the difference between this element's
    // global number and the number of the first element in its queue.
    //
    __device__ static void
    BlockComputeQueues(unsigned int Ai, 
		       unsigned int idx,
		       uint8_t &queue,
		       unsigned int &offset)
    {
      if (NQUEUES == 1)
	{
	  queue = 0;
	
	  offset = idx;
	}
      else
	{
	  // compute the queue from which the current elt should be read
	  queue = warpLowerBound(Ai, idx);
	
	  // compute the offset in the queue from which elt should be read
	  offset = idx - __shfl_sync(WARP_MASK, Ai, queue);
	}
    }
  };
}

#endif
