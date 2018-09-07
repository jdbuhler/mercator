#ifndef __REDUCEBYKEY_CUH
#define __REDUCEBYKEY_CUH

#include <cub/cub.cuh>

#include "warpsort.cuh"
#include "subwarpreduce.cuh"

//
// class ReduceByKey
// Reduces a collection of values, each tagged with a key <= NQUEUES,
// so that all values with the same queue are reduced together.  The
// value type T must support a commutative, associative reduction operator.
//
// Template parameters:
//   T              -- type of value to be reduced
//   NQUEUES        -- number of distinct keys; must be <= ARCH_WARP_SIZE.
//   ACTIVE_THREADS -- number of active threads; must be multiple of
//                        ARCH_WARP_SIZE
//   ARCH_WARP_SIZE -- size of a warp on the current GPU architecture
//
template <typename T,
	  uint8_t NQUEUES,
	  unsigned int ACTIVE_THREADS,
	  unsigned int ARCH_WARP_SIZE = 32>
class ReduceByKey {

  // NP2_UINT8(x)
  // Compute the greatest power of 2 >= the input x, which must be a
  // postive integer <= 256.
  static constexpr uint8_t LOG2_2(uint8_t s)
  { return((s & 0x02) ? 1 : 0); }
  
  static constexpr uint8_t LOG2_4(uint8_t s)
  { return ((s & 0x0c) ? (2 + LOG2_2(s >> 2)) : LOG2_2(s)); }
  
  static constexpr uint8_t LOG2_UINT8(uint8_t s)
  { return ((s & 0xf0) ? (4 + LOG2_4(s >> 4)) : LOG2_4(s)); }
  
  static constexpr uint8_t NP2_UINT8(uint8_t i)
  { return (i == 1 ? 1 : (1U << (LOG2_UINT8(i - 1) + 1))); }
  
  enum { 
    NWARPS    = ACTIVE_THREADS / ARCH_WARP_SIZE,
    RED_WIDTH = NP2_UINT8(NWARPS) 
  };
  
  // PAD()
  // Compute padded equivalent of unpadded index value for warpSums[].
  // to minimize shared memory bank conflicts
  __device__
  static constexpr unsigned int PAD(unsigned int v)
  {
    // Add 32 bits (= 4 bytes) of padding, once every 64 32-bit words
    // for values of type T at most 4 bytes.  For larger sizes,
    // add one padding value about every 64 bytes and hope for the best
    // (and that we don't exhaust shared memory!)
    typedef enum {
      PAD_IVAL  = 64 * 4 / sizeof(T),
      PAD_SIZE0 = 4 / sizeof(T),
      PAD_SIZE  = (PAD_SIZE0 == 0 ? 1 : PAD_SIZE0)
    } hukairs;
    
    return (v + v / PAD_IVAL * PAD_SIZE);
  }
  
  __device__
  static bool
  warpIsHead(unsigned int v)
  {
    return ((cub::LaneId() == 0) | (v != __shfl_up(v, 1)));
  }
  
public:
  
  //
  // ReduceByKey()
  //
  // Reduce an array of values, specified one per thread, that are tagged
  // with keys between 0 and NQUEUES - 1.  The reduction of all values
  // with tag j is reduced into the accumulator acc[j].
  //
  // T must support a commutative, associative reduction operator op,
  // which supports the following two methods:
  //  T zero() -- return the additive identity for the reduction 
  //    (e.g. 0 for + on integers)
  //  T operator()(const T &a, const T &b) -- apply the reduction
  //    operator to inputs a and b.
  //
  // Shared memory usage: NQUEUES * NWARPS * sizeof(T), plus padding
  //
  template <typename ReductionOp>
  __device__ static void
  Reduce(T v, uint8_t key, T acc[NQUEUES], ReductionOp op)
  {
    static_assert(NQUEUES <= ARCH_WARP_SIZE, "too many queues in ReduceByKey");
    
    const unsigned int tid = threadIdx.x;

    __shared__ T warpSums[PAD(NWARPS * NQUEUES)];
    
    // Clear out the counts array in preparation for the next step
    if (tid < NQUEUES * NWARPS)
      warpSums[PAD(tid)] = op.zero();
    
    __syncthreads();
    
    // sort by key, tracking original index of each
    // key in the warp.
    
    unsigned int pair = ((unsigned int) (cub::LaneId() << 8) | key);
    
    pair = WarpSort<unsigned int>::sort(pair);
    key          = (pair & 0x00FF);
    uint8_t lane = (pair >> 8);
    
    // reduce values with the same key tag within each warp
    // NB: rely on use of shuffle() in WarpReduce and consequent
    // zero size of temp_storage, which is never written, to
    // allocate just one copy instead of one per warp.
    typedef cub::WarpReduce<T> WarpReduce;
    __shared__ typename WarpReduce::TempStorage temp_storage;

    v = cub::ShuffleIndex(v, lane); // permute values to match order of keys    
    bool isHead = warpIsHead(key);
    T sum = WarpReduce(temp_storage).HeadSegmentedReduce(v, isHead, op);
    
    const unsigned int warpId = tid / ARCH_WARP_SIZE;
    
    if (isHead & key < NQUEUES)
      warpSums[PAD(key * NWARPS + warpId)] = sum;
    
    __syncthreads();
    
    // reduce values with the same key tag across warps
    for (unsigned int j = 0; j < RED_WIDTH * NQUEUES; j += ACTIVE_THREADS)
      {
	if (j + tid < RED_WIDTH * NQUEUES)
	  {
	    typedef SubwarpReduce<T, RED_WIDTH> Reducer;
	    
	    unsigned int k = (j + tid) / RED_WIDTH;
	    unsigned int w = (j + tid) % RED_WIDTH;
	    
	    if (w < NWARPS)
	      {
		T v = warpSums[PAD(k * NWARPS + w)];
		T currSum = Reducer::Reduce(v, op, NWARPS);
		
		if (w == 0)
		  acc[k] = op(acc[k], currSum);
	      }
	  }
      }
  }
};

#endif
