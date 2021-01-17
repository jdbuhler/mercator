#ifndef __SOURCE_CUH
#define __SOURCE_CUH

//
// @file Source.cuh
// device-side source object associated with Node_Source
//
// MERCATOR
// Copyright (C) 2018 Washington University in St. Louis; all rights reserved.
//

#include <type_traits>
#include <cassert>

namespace Mercator {
  
  template <typename T, typename Enable = void>
  class SourceBase;

  template <typename T>
  class SourceBase<T, std::enable_if_t<std::is_scalar<T>::value> > {
  public:
    using EltT = T;
  };

  template <typename T>
  class SourceBase<T, std::enable_if_t<!std::is_scalar<T>::value> > {
  public:
    using EltT = T&;
  };
  
  // Actual Source class begins here
  
  template <typename T>  
  class Source : public SourceBase<T> {
    
  public:
    
    //
    // @brief constructor 
    // 
    // @param rangeData object describing range
    // @param tail external tail pointer
    //
    __device__
    Source()
      : size(0),
	tail(nullptr)
    {}
    
    __device__
    void init(size_t nInputs,
	      size_t *itail)
    {
      size = nInputs;
      reqLimit = max((size_t) 1, size / gridDim.x);
      
      tail = itail;
    }
    
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
    
    
    //
    // @brief get an element from the array
    //
    __device__
    typename Source<T>::EltT get(size_t idx) const 
    {
      assert(idx <= size);
            
      return idx;
    }
    
    
    //
    // @brief return a dummy element we don't care about
    //
    __device__
    typename Source<T>::EltT getDummy() const { return get(0); }
    
  private:
    
    size_t size;
    size_t *tail;
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
