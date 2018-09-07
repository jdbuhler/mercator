#ifndef __WARPSORT_CUH
#define __WARPSORT_CUH

#include <cub/cub.cuh>

namespace Mercator {
  
  //
  // class WarpSort()
  // Intrawarp sort using 32-way bitonic sorting.  Data type KeyT must
  // be a type that can be accessed by the CUDA shfl functions.
  //
  // WARNING: this version is specialized to look at only bits of the
  // key specified by KEYMASK!  This lets us carry along other data with
  // the key in the other bits.
  //
  template <typename KeyT, unsigned int KEYMASK>
  class WarpSort {
  private:
  
    // mask for all threads in a warp
    static const int ARCH_WARP_SIZE = 32;
    static const unsigned int WARP_MASK = ((1ULL << ARCH_WARP_SIZE) - 1ULL);
  
    __device__
    static KeyT 
    swap(KeyT x, unsigned int mask, unsigned int dir)
    {
      unsigned int y = __shfl_xor_sync(WARP_MASK, x, mask);
    
      int diff = (x & KEYMASK) - (y & KEYMASK);
      int v    = (dir ? diff : -diff);
    
      return (v < 0 ? y : x);
    }
  
    __device__ 
    static unsigned int
    bfe(unsigned int i, unsigned int k) { return cub::BFE(i, k, 1); }

  public:
  
    // sort()
    // Sort the values in the variable x within each warp.  Return the
    // value of x in each thread after sorting.
    //
    __device__ 
    static KeyT sort(KeyT x)
    {
      unsigned int laneid = cub::LaneId();
      
      x = swap(x, 0x01, bfe(laneid, 1) ^ bfe(laneid, 0));   //  2
      
      x = swap(x, 0x02, bfe(laneid, 2) ^ bfe(laneid, 1));   //  4
      x = swap(x, 0x01, bfe(laneid, 2) ^ bfe(laneid, 0));  
    
      x = swap(x, 0x04, bfe(laneid, 3) ^ bfe(laneid, 2));   //  8
      x = swap(x, 0x02, bfe(laneid, 3) ^ bfe(laneid, 1));
      x = swap(x, 0x01, bfe(laneid, 3) ^ bfe(laneid, 0));
    
      x = swap(x, 0x08, bfe(laneid, 4) ^ bfe(laneid, 3));   // 16
      x = swap(x, 0x04, bfe(laneid, 4) ^ bfe(laneid, 2));
      x = swap(x, 0x02, bfe(laneid, 4) ^ bfe(laneid, 1));
      x = swap(x, 0x01, bfe(laneid, 4) ^ bfe(laneid, 0));
    
      x = swap(x, 0x10,                  bfe(laneid, 4));   // 32
      x = swap(x, 0x08,                  bfe(laneid, 3));
      x = swap(x, 0x04,                  bfe(laneid, 2));
      x = swap(x, 0x02,                  bfe(laneid, 1));
      x = swap(x, 0x01,                  bfe(laneid, 0));

      return x;
    }
  };


  //
  // class WarpSortWithValues() 
  //
  // Intrawarp sort using 32-way bitonic sorting.  Data type KeyT must
  // be a type that can be accessed by the CUDA shfl functions.  Type
  // ValT is the type of a value that is moved between threads along
  // with the sort key.  It can even be a struct or other complex type.
  //
  template <typename KeyT, typename ValT>
  class WarpSortWithValues {
  private:

    // mask for all threads in a warp
    static const int ARCH_WARP_SIZE = 32;
    static const unsigned int WARP_MASK = ((1ULL << ARCH_WARP_SIZE) - 1ULL);
  
    __device__
    static ValT 
    ShuffleXor(ValT v, unsigned int mask)
    {
      if (std::is_integral<ValT>::value && sizeof(ValT) <= sizeof(uint32_t))
	return (ValT) __shfl_xor_sync(WARP_MASK, (uint32_t) v, mask);
      else if (std::is_floating_point<ValT>::value && 
	       sizeof(ValT) <= sizeof(float))
	return (ValT) __shfl_xor_sync(WARP_MASK, (float) v, mask);
      else
	{
	  unsigned int lane = cub::LaneId() ^ mask;
	  return cub::ShuffleIndex(v, cub::LaneId() ^ mask);
	}
    }
  
    __device__
    static KeyT 
    swap(KeyT x, ValT &v, unsigned int mask, unsigned int dir)
    {
      unsigned int y = __shfl_xor(x, mask);
      ValT         u = ShuffleXor(v, mask);
    
      bool pred = (dir ? x < y : y < x);
      v =    (pred ? u : v);
      return (pred ? y : x);
    }
  
    __device__ 
    static unsigned int
    bfe(unsigned int i, unsigned int k) { return cub::BFE(i, k, 1); }

  public:

    // sort()
    // Sort the values in the variable x within each warp.  The auxiliary
    // values v in each warp are permuted into the sorted order of the keys.
    //
    // Return the value of x in each thread after sorting.
    //
    __device__ 
    static KeyT sort(KeyT x, ValT &v)
    {
      unsigned int laneid = cub::LaneId();
    
      x = swap(x, v, 0x01, bfe(laneid, 1) ^ bfe(laneid, 0));   //  2
    
      x = swap(x, v, 0x02, bfe(laneid, 2) ^ bfe(laneid, 1));   //  4
      x = swap(x, v, 0x01, bfe(laneid, 2) ^ bfe(laneid, 0));  
    
      x = swap(x, v, 0x04, bfe(laneid, 3) ^ bfe(laneid, 2));   //  8
      x = swap(x, v, 0x02, bfe(laneid, 3) ^ bfe(laneid, 1));
      x = swap(x, v, 0x01, bfe(laneid, 3) ^ bfe(laneid, 0));
    
      x = swap(x, v, 0x08, bfe(laneid, 4) ^ bfe(laneid, 3));   // 16
      x = swap(x, v, 0x04, bfe(laneid, 4) ^ bfe(laneid, 2));
      x = swap(x, v, 0x02, bfe(laneid, 4) ^ bfe(laneid, 1));
      x = swap(x, v, 0x01, bfe(laneid, 4) ^ bfe(laneid, 0));
    
      x = swap(x, v, 0x10,                  bfe(laneid, 4));   // 32
      x = swap(x, v, 0x08,                  bfe(laneid, 3));
      x = swap(x, v, 0x04,                  bfe(laneid, 2));
      x = swap(x, v, 0x02,                  bfe(laneid, 1));
      x = swap(x, v, 0x01,                  bfe(laneid, 0));
    
      return x;
    }
  };
}

#endif
