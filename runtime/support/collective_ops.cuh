#ifndef __COLLECTIVE_OPS_CUH
#define __COLLECTIVE_OPS_CUH

//
// @file collective_ops.cuh
// @brief block-wide and warp-side collective operations needed by MERCATOR
//

// Currently, we implement these operators in terms of the CUB library

#include <cub/cub.cuh>

namespace Mercator  {
  
  //
  // @brief Block-wide scan opeations
  // @tparam T data type for sum
  // @tparam TPB threads per block
  //
  template <typename T, unsigned int TPB>
  class BlockScan {
    
    using Scanner =
      cub::BlockScan<T, TPB,
		     cub::BLOCK_SCAN_WARP_SCANS>;
    
  public:
    
    //
    // @brief exclusive sum
    // @param v input values per thread
    // @param agg sum of all inputs (returned in every thread)
    //
    // @return exclusive sums in each thread
    //
    __device__
    static T exclusiveSum(const T &v, T &agg)
    {
      __shared__ typename Scanner::TempStorage CUB_tmp;
      T sv;
      
      Scanner(CUB_tmp).ExclusiveSum(v, sv, agg);
      
      return sv; 
    }
  };

  //
  // @brief warp-wide scan opeations
  // @tparam T data type for sum
  // @tparam TPB threads per warp
  //
  template <typename T, unsigned int TPW>
  class WarpScan {
    
    using Scanner = cub::WarpScan<T, TPW>;
    
  public:
    
    //
    // @brief exclusive sum
    // @param v input values per thread
    //
    // @return exclusive sums in each thread of warp
    //
    __device__
    static T exclusiveSum(const T &v)
    {
      // In theory, we need to allocate one of these per warp.  But
      // with shuffles, it is never used.  Make sure it has
      // essentially zero size (actually size 1, perhaps because CUB
      // takes its address?)
      
      __shared__ typename Scanner::TempStorage CUB_tmp;
      
      static_assert(sizeof(CUB_tmp) <= 1,
		     "warp scan requires temporary storage!");
      
      T sv;
      
      Scanner(CUB_tmp).ExclusiveSum(v, sv);
      
      return sv; 
    }
 
   //
    // @brief exclusive sum
    // @param v input values per thread
    //
    // @return exclusive sums in each thread of warp
    //
    __device__
    static T exclusiveSum(const T &v, T &agg)
    {
      // In theory, we need to allocate one of these per warp.  But
      // with shuffles, it is never used.  Make sure it has
      // essentially zero size (actually size 1, perhaps because CUB
      // takes its address?)

      __shared__ typename Scanner::TempStorage CUB_tmp;
      
      static_assert(sizeof(CUB_tmp) <= 1,
		    "warp scan requires temporary storage!");
      
      T sv;
      
      Scanner(CUB_tmp).ExclusiveSum(v, sv, agg);
      
      return sv; 
    }
  };
  

  //
  // @brief block-wide sum reductions
  // @tparam T type over which to reduce
  // @tparam TPB threads per block
  //
  template <typename T, unsigned int TPB>
  class BlockReduce {
    
    using Reducer =
      cub::BlockReduce<T, TPB>;
    
  public:
    
    //
    // @brief compute a sum reduction over the first nThreads threads
    //   of the block
    // @param v values to sum
    // @param nThreads # of threads to sum over (defaults to all)
    //
    // @returns sum (in thread 0 ONLY)
    //
    __device__
    static T sum(const T &v, unsigned int nThreads = TPB)
    {
      __shared__ typename Reducer::TempStorage CUB_tmp;
      
      return Reducer(CUB_tmp).Sum(v, nThreads);
    }
  };

  //
  // @brief block-wide sum reductions
  // @tparam T type over which to reduce
  // @tparam TPW threads per warp
  //
  template <typename T, unsigned int TPW>
  class WarpReduce {
    
    using Reducer =
      cub::WarpReduce<T, TPW>;
    
  public:
    
    //
    // @brief compute a sum reduction over the first nThreads threads
    //   of the block
    // @param v values to sum
    // @param nThreads # of threads to sum over (defaults to all)
    //
    // @returns sum (in thread 0 ONLY)
    //
    __device__
    static T sum(const T &v, unsigned int nThreads = TPW)
    {
      // In theory, we need to allocate one of these per warp.  But
      // with shuffles, it is never used.  Make sure it has
      // essentially zero size (actually size 1, perhaps because CUB
      // takes its address?)

      __shared__ typename Reducer::TempStorage CUB_tmp;
      
      static_assert(sizeof(CUB_tmp) <= 1,
		    "warp reduce requires temporary storage!");
       
      return Reducer(CUB_tmp).Sum(v, nThreads);
    }
  };
  
  
  //
  // @brief compute the argmax of a set of key-value pairs
  // @tparam K key type
  // @tparam V value type
  // @tparam TPB threads per block
  //
  template <typename K, typename V, unsigned int TPB>
  class BlockArgMax {
    
    using Reducer =
      cub::BlockReduce<cub::KeyValuePair<K, V>, TPB>;
    
  public:
    
    //
    // @brief compute argmax, the key associated with the largest value
    // @param key key element of each thread's pair
    // @param value value element of each thread's pair
    // @return key associated with largest value (in 0th thread ONLY)
    //
    __device__
    static K argmax(const K &key, const V &value)
    {
      cub::KeyValuePair<K, V> myPair;
      myPair.key   = key;
      myPair.value = value;
      
      __shared__ typename Reducer::TempStorage CUB_tmp;
      
      auto resultPair =
	Reducer(CUB_tmp).Reduce(myPair, cub::ArgMax());
      
      return resultPair.key;
    }

    //
    // @brief compute argmax, the key associated with the largest value,
    //    in a way that supports ops over arrays greater than block size
    //
    // @param key key element of each thread's pair
    // @param value value element of each thread's pair
    // @param maxValue output parameter for largest value (set in 0th 
    //          thread ONLY)
    // @param nThreads number of threads with valid inputs
    // @return key associated with largest value (in 0th thread ONLY)
    //
    __device__
    static K argmax(const K &key, const V &value, V &maxValue,
		    unsigned int nThreads = TPB)
    {
      cub::KeyValuePair<K, V> myPair;
      myPair.key   = key;
      myPair.value = value;
      
      __shared__ typename Reducer::TempStorage CUB_tmp;
      
      auto resultPair =
	Reducer(CUB_tmp).Reduce(myPair, cub::ArgMax(), nThreads);
      
      if (threadIdx.x == 0)
	maxValue = resultPair.value;
      
      return resultPair.key;
    }

  };
  
  
  //
  // @brief compute discontinuities in a block of values
  // @tparam T data type for sum
  // @tparam TPB threads per block
  //
  template <typename T, unsigned int TPB>
  class BlockDiscontinuity {
    
    using Discontinuity =
      cub::BlockDiscontinuity<T, TPB>;
    
  public:
    
    //
    // @brief flag threads whose value is the tail of a run of identical
    // values.
    // @param v data over which to compute discontinuities
    // @param finalV value that follows last thread's value
    //
    // @return true for the tail of each run of identical values
    //
    __device__
    static bool flagTails(const T &v, const T &finalV)
    {
      __shared__ typename Discontinuity::TempStorage CUB_tmp; 
      
      T vs[1];
      bool tailFlags[1];
      
      vs[0] = v;
      
      Discontinuity(CUB_tmp).FlagTails(tailFlags,
				       vs,
				       cub::Inequality(), 
				       finalV);
      
      return tailFlags[0];
    }
  };
  
}

#endif
