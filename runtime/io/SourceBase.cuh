#ifndef __SOURCEBASE_CUH
#define __SOURCEBASE_CUH

//
// @file SourceBase.cuh
// Base for device-side source objects associated with Node_Source
//
// MERCATOR
// Copyright (C) 2021 Washington University in St. Louis; all rights reserved.
//

#include <cassert>

#include "support/devutil.cuh"

namespace Mercator {
  
  template <typename T>  
  class SourceBase {
    
  public:
    
    using EltT = typename ConstReturnType<T>::EltT;
    
    //
    // @brief constructor 
    // 
    // @param rangeData object describing range
    // @param tail external tail pointer
    //
    __device__
    SourceBase(size_t *tail)
      : tail(tail),
	size(0)
    {}    
    
    //
    // @brief get advisory request limit based on input stream size, so
    // that every block in the grid will receive work.
    //
    __device__
    size_t getRequestLimit() const
    {
      return reqLimit;
    }
    
    
    //
    // @brief reserve up to reqSize elements from the end of the array.
    // If fewer elements are available, we reserve what we can.  This
    // function is single-threaded per block but safe to call from many
    // blocks concurrently.
    //
    // @param reqSize number of elements requested
    // @param base OUTPUT parameter; receives start of requested range
    // 
    // @return number of elements actually reserved
    //
    __device__
    size_t reserve(size_t reqSize,
		   size_t *base)
    {
      if (*tail >= size)
	return 0;
      
      // try to reserve reqSize items
      *base = myAtomicAdd(tail, reqSize);
      
      // how many items did we actually get?
      return min(reqSize, (*base >= size
			   ? 0 
			   : size - *base));
    }

    
  protected:
    
    __device__
    void setStreamSize(size_t isize)
    {
      size = isize;
      reqLimit = max((size_t) 1, size / gridDim.x);
    }
    
  private:
    
    size_t* const tail;
    
    size_t size;
    size_t reqLimit;
    
    //
    // @brief CUDA does not provide an atomic add for size_t, despite
    // its being a 64-bit integer type.  Provide one here.
    //
    // @param address pointer to value to increment
    // @param val size of increment
    //
    __device__ 
    size_t myAtomicAdd(size_t *address, size_t val)
    {
      typedef unsigned long long ull;
      
      static_assert(sizeof(size_t) == sizeof(ull),
		    "ERROR: sizeof(size_t) != sizeof(ULL)");
      
      return (size_t) atomicAdd((ull *) address, (ull) val);
    }
  };
}

#endif
