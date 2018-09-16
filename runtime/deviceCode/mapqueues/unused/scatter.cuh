#ifndef __SCATTER_CUH
#define __SCATTER_CUH

#include "warpsort.cuh"
#include "subwarpscan.cuh"

#include "support/collective_ops.cuh"

namespace Mercator {
  
  //
  // class QueueScatter
  // Compute the information needed to distribute data tagged with queue numbers
  // from a common output buffer to individual queues.  See the documentation
  // of the WarpSortAndComputeOffsets() function below for details.
  //
  // Template parameters:
  //   NQUEUES -- number of queues; only input keys < NQUEUES are considered
  //   valid queue numbers and assigned corresponding offsets.
  //
  //   ACTIVE_THREADS -- number of active threads executing the scatter
  //   function; must be a multiple of ARCH_WARP_SIZE
  //
  //   ARCH_WARP_SIZE -- size of a warp for this GPU architecture
  //
  template <uint8_t NQUEUES,
	    unsigned int ACTIVE_THREADS,
	    unsigned int ARCH_WARP_SIZE = 32>
  class QueueScatter {
  
  private:

    // mask for all threads in a warp
    static const unsigned int WARP_MASK = ((1ULL << ARCH_WARP_SIZE) - 1ULL);
    
    // NP2_UINT8(x)
    // Compute the greatest power of 2 >= the input x, which must be a
    // postive integer <= 256.
    __device__
    static constexpr uint8_t LOG2_2(uint8_t s)
    { return((s & 0x02) ? 1 : 0); }
  
    __device__
    static constexpr uint8_t LOG2_4(uint8_t s)
    { return ((s & 0x0c) ? (2 + LOG2_2(s >> 2)) : LOG2_2(s)); }
  
    __device__
    static constexpr uint8_t LOG2_UINT8(uint8_t s)
    { return ((s & 0xf0) ? (4 + LOG2_4(s >> 4)) : LOG2_4(s)); }
  
    __device__
    static constexpr uint8_t NP2_UINT8(uint8_t i)
    { return (i == 1 ? 1 : (1U << (LOG2_UINT8(i - 1) + 1))); }
  
    // If no one key can exceed 256 copies over all active threads, we can 
    // use uint8s to count it; otherwise, use uint16s.
    typedef typename
    cub::If<ACTIVE_THREADS <= 256, uint8_t, uint16_t>::Type CountType;  

    enum { 
      NWARPS = ACTIVE_THREADS / ARCH_WARP_SIZE,
    
      // width of count scan over all warps -- pad out to next power of 2 
      SCANWIDTH = NP2_UINT8(NWARPS)
    };
  
    // PAD()
    // Compute padded equivalent of unpadded index value for counts[].
    // to minimize shared memory bank conflicts
    __device__
    static constexpr unsigned int PAD(unsigned int v)
    {
      // Add 32 bits (= 4 bytes) of padding, once every 64 32-bit words
      typedef enum {
	PAD_IVAL  = 64 * 4 / sizeof(CountType),
	PAD_SIZE  = 4 / sizeof(CountType)
      } hukairs;
    
      return (v + v / PAD_IVAL * PAD_SIZE);
    }
  
    // warpIsHead()
    // Given values v per thread, return 1 for each index i such that
    // i is the head of a run of identical values, or 0 otherwise.
    // Runs independently in each warp.
    __device__
    static bool
    warpIsHead(unsigned int v)
    {
      return ((cub::LaneId() == 0) | 
	      (v != __shfl_up_sync(WARP_MASK, v, 1)));
    }
  
    // warpIsTail()
    // Given values v per thread, return 1 for each index i such that
    // i is the tail of a run of identical values, or 0 otherwise.
    // Runs independently in each warp.
    __device__
    static bool
    warpIsTail(unsigned int v)
    {
      return ((cub::LaneId() == ARCH_WARP_SIZE - 1) | 
	      (v != __shfl_down_sync(WARP_MASK, v, 1)));
    }
  
    // warpHeadOffset()
    // Given values v per thread, return for each i the offset of i from
    // the head of its segment.
    // Runs independently in each warp.
    __device__
    static uint8_t
    warpHeadOffset(unsigned int v)
    {
      unsigned int headWord = __ballot_sync(WARP_MASK, warpIsHead(v));
    
      uint16_t myHead = 
	(ARCH_WARP_SIZE - 1) -
	__clz(headWord & ((1U << (cub::LaneId() + 1)) - 1));
    
      return cub::LaneId() - myHead;
    }
  
  public:
  
    // 
    // WarpSortAndComputeOffsetsMany
    // INPUT: one key per thread. Assumes that key <= ARCH_WARP_SIZE.
    // 
    // RESULTS:
    //  * sort keys within each warp
    //  * permute thread IDs to match sorting order of keys; return in idx
    //  * return offset of each sorted key among all keys in the input with
    //    the same value.
    //  * if input 'aggs' is non-zero, it is assumed to point to an 
    //    array of size NQUEUES, which is filled in with the total number
    //    of times each key appears in the input.
    //
    // Example (with three warps, assuming ARCH_WARP_SIZE = 4)
    //   input  keys =   1 3 3 2  0 2 1 3  1 2 0 0
    //
    //   output keys    =   1 2 3 3  0 1 2 3  0 0 1 2
    //   output idxs    =   0 3 1 2  4 6 5 7  10 11 8 9
    //   output offsets =   0 0 0 1  0 1 1 2  1 2 2 2
    //   aggs           = [3 3 3 3]
    //  
    //   Hence, if the output indices are distributed to queues by key,
    //   we get 
    //     0: [4 10 11] 1: [0 6 8] 2: [3 5 9] 3: [1 2 7] and all
    //   the values can be written in parallel using the supplied offsets,
    //   and the queues' next-free pointers can be incremented by the
    //   values in aggs.
    //
    // If any input value is >= NQUEUES, that value will be sorted correctly
    // within the warp but will *not* affect the offsets computed for
    // keys < NQUEUES.
    //
    __device__
    static uint16_t 
    WarpSortAndComputeOffsetsMany(uint8_t &key, uint16_t &idx,
				  uint16_t *aggs = 0)
    {
      static_assert(NQUEUES <= ARCH_WARP_SIZE, "too many keys for Scatter");
    
      const unsigned int tid = threadIdx.x;
    
      const unsigned int COUNT_SIZE = PAD(NWARPS * NQUEUES);
      __shared__ CountType counts[COUNT_SIZE];
    
      // Clear out the counts array in preparation for the next step
      if (tid < NWARPS * NQUEUES)
	counts[PAD(tid)] = 0;
    
      __syncthreads();
    
      // pack together sort key, which is the tuple [actual key, lane
      // id] (so that sort is stable within each warp), and the original
      // array index, which references any other data associated with
      // the key, into a pair.  Sorting is only by key bits.
    
      const unsigned int WARP_IDXWIDTH = LOG2_UINT8(ARCH_WARP_SIZE);
      const unsigned int KEYBITS = 2 * WARP_IDXWIDTH;
    
      // make sure we have enough room to store the sort key plus
      // a thread index for the largest possible blocksize (1024)
      static_assert(sizeof(unsigned int) * 8 - KEYBITS >= 10,
		    "Cannot support warp-size/block-size combo in WarpSort");
    
      unsigned int pair = 
	((unsigned int) key) << WARP_IDXWIDTH 
	| cub::LaneId()
	| tid << KEYBITS;
    
      const unsigned int KEYMASK = (1U << KEYBITS) - 1;
    
      // sort the key/idx pairs within each warp by key,
      // then unpack the key and index from each pair 
      pair = WarpSort<unsigned int, KEYMASK>::sort(pair);
      key = (pair & KEYMASK) >> WARP_IDXWIDTH;
      idx = (pair >> KEYBITS);
    
      // compute offset of each key relative to the beginning of
      // the sorted run of the same key within its warp
      uint16_t warpOffset = warpHeadOffset(key);
    
      const unsigned int warpId = tid / ARCH_WARP_SIZE;
    
      // Gather per-warp counts for each key into shared memory,
      // with the counts for a key in all warps stored contiguously.
      if (warpIsTail(key) & (key < NQUEUES)) 
	{
	  // record segment size for this warp
	  counts[PAD(key * NWARPS + warpId)] = warpOffset + 1;
	}
    
      __syncthreads();
    
      // Perform subwarp exclusive sums across per-warp counts
      // for each key
      // NB: may require more than the number of active threads
      for (unsigned int j = 0; j < SCANWIDTH * NQUEUES; j += ACTIVE_THREADS)
	{
	  if (j + tid < SCANWIDTH * NQUEUES)
	    {
	      typedef SubwarpScan<unsigned int, SCANWIDTH> SWScan;
	    
	      unsigned int k = (j + tid) / SCANWIDTH;
	      unsigned int w = (j + tid) % SCANWIDTH;
	    
	      unsigned int v = (w < NWARPS 
				? counts[PAD(k * NWARPS + w)] 
				: 0);
	    
	      unsigned int agg;
	      SWScan::ExclusiveSum(v, v, agg);
	    
	      // gather aggregates across scan groups
	      if (aggs && w == 0)
		aggs[k] = agg;
	    
	      if (w < NWARPS) counts[PAD(k * NWARPS + w)] = v;
	    }
	}
    
      __syncthreads();
    
      // add total contribution of previous warps to warp-local offset
      return (key < NQUEUES 
	      ? warpOffset + counts[PAD(key * NWARPS + warpId)]
	      : 0);
    }

    // 
    // WarpComputeOffsetsOne()
    // Special case of WarpSortAndComputeOffsets for NQUEUES == 1.
    // Does not actually partition valid and invalid values in each
    // warp (i.e. does not sort), but returns correct offsets for
    // all valid values.
    //
    __device__
    static uint16_t
    WarpComputeOffsetsOne(uint8_t &key, uint16_t &idx,
			  uint16_t *aggs = 0)
    {
      const unsigned int tid = threadIdx.x;    
    
      uint16_t agg;
    
      using Scan = BlockScan<uint16_t, ACTIVE_THREADS>;
      uint16_t offset = Scan::exclusiveSum(key < NQUEUES, agg);
    
      if (aggs)
	{
	  if (tid == 0)
	    aggs[0] = agg;
	  __syncthreads(); // make sure all threads can see aggregate sum
	}
    
      idx = tid; // no rearrangement of keys
    
      return offset;
    }

    __device__
    static uint16_t   
    WarpSortAndComputeOffsets(uint8_t &key, uint16_t &idx,
			      uint16_t *aggs = 0)
    {
      if (NQUEUES == 1)
	{
	  return WarpComputeOffsetsOne(key, idx, aggs);
	}
      else
	{
	  return WarpSortAndComputeOffsetsMany(key, idx, aggs);
	}
    }
  };
}

#endif
