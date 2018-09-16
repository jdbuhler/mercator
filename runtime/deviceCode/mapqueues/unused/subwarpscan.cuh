#ifndef __SUBWARPSCAN_CUH
#define __SUBWARPSCAN_CUH

#include <cub/cub.cuh>

namespace Mercator {
  
  //
  // SubwarpScan()
  //
  // Perform scan operations on sub-ranges of a warp.  Operations are
  // performed on each SCAN_WIDTH consecutive threads, which must be
  // a power of 2 <= the architecture's warp size.
  //
  // T -- type of value to be scanned (must support +)
  //
  template <typename T, uint8_t SCAN_WIDTH>

  class SubwarpScan {
  public:
  
    // mask for all threads in a warp
    static const unsigned int ARCH_WARP_SIZE = 32;
    static const unsigned int WARP_MASK = ((1ULL << ARCH_WARP_SIZE) - 1ULL);
  
    // Compute an exclusive sum across threads of the values in each
    // SCAN_WIDTH consecutive threads, returning the results in sum.
    __device__
    static void ExclusiveSum(T v, T &sum) 
    {
      if (SCAN_WIDTH == 1)
	sum = 0;
      else
	{
	  using Scan = WarpScan<unsigned int, SCAN_WIDTH>;
	  sum = Scan::exclusiveSum(v);
	}
    }
  
    // Compute an exclusive sum across threads of the values in each
    // SCAN_WIDTH consecutive threads, returning the results in sum.
    //
    // All the threads in each scanned group receive the sum of 
    // of the group's values in agg.  
    __device__
    static void ExclusiveSum(T v, T &sum, T &agg)
    {
      ExclusiveSum(v, sum);
    
      unsigned int lane = cub::LaneId();
    
      // the last thread in each scanned group computes the group's aggregate
      if (lane % SCAN_WIDTH == SCAN_WIDTH - 1)
	agg = sum + v;
    
      // broadcast the group aggregate to all its threads
      agg = __shfl_sync(WARP_MASK, 
			agg, 
			(lane / SCAN_WIDTH) * SCAN_WIDTH + SCAN_WIDTH - 1);
    }
  };
}

#endif
