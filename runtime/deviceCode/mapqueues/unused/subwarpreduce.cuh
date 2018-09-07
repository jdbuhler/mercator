#ifndef __SUBWARPREDUCE_CUH
#define __SUBWARPREDUCE_CUH

#include <cub/cub.cuh>

namespace Mercator {
  
  //
  // SubwarpReduce()
  //
  // Perform reduce operations on sub-ranges of a warp.  Operations are
  // performed on each RED_WIDTH consecutive threads, which must be
  // a power of 2<= the architecture's warp size.  The type to be reduced
  // must support a commutative, associative reduction operation.
  //
  // T                -- type of value to be reduced
  // RED_WIDTH        -- width of intervals to reduce
  //
  template <typename T, 
	    uint8_t RED_WIDTH>
  class SubwarpReduce {
  public:

    // Reduce() 
    // Compute a reduction across threads of the values in each
    // RED_WIDTH threads.  If fewer than RED_WIDTH threads of the warp
    // have valid values, nValid specifies the number of valid values
    // (starting with lane 0).
    //
    // RETURNS: result of reduction for the valid values in each window
    // of size RED_WIDTH threads with index 0 mod RED_WIDTH.
    //
    template <typename ReductionOp>
    __device__ static T 
    Reduce(T v, ReductionOp op, uint8_t nValid = RED_WIDTH)
    {
      if (RED_WIDTH == 1)
	return v;
      else
	{
	  typedef cub::WarpReduce<T, RED_WIDTH> WarpReduce;
	
	  // We have to declare temporary storage to prevent CUB from
	  // barfing, but on architectures supporting shuffle, WarpReduce
	  // does not use it.  Hence, we allocate just one, rather
	  // than one per active warp, relying on the fact that it
	  // occupies 0 bytes and is not written.
	  __shared__ typename WarpReduce::TempStorage temp_storage;
	
	  return WarpReduce(temp_storage).Reduce(v, op, nValid);
	}
    }
  };
}

#endif
