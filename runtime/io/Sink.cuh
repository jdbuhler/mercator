#ifndef __SINK_CUH
#define __SINK_CUH

//
// @file Sink.cuh
// device-side sink object associated with Node_Sink
//
// MERCATOR
// Copyright (C) 2018 Washington University in St. Louis; all rights reserved.
//

#include <cassert>

#include "Buffer.cuh"

namespace Mercator {
  
  template <typename T>
  struct SinkData {
    BufferData<T> *bufferData; // Buffer object
  };

  //
  // A generic Sink<T> presents as a write-only array with an
  // associated size, which is managed with atomic increments to
  // support concurrent claiming of space to write output by many
  // processors.
  //
  template <typename T>
  class Sink {
    
  public:    
    
    //
    // @brief constructor 
    // 
    // @param capacity capacity if target array
    // @param isize pointer to shared size field
    //
    __device__
    Sink()
      : capacity(0),
	data(nullptr)	
    {}
    
    __device__
    void init(BufferData<T> *bufferData)
    {
      capacity = bufferData->capacity;
      size = &bufferData->size;
      data = bufferData->data;
    }
    
    //
    // @brief reserve up to reqSize elements's worth of space from the
    // end of the array.  If not enough space exists, crash.
    //
    // @param reqSize amount of space requested
    // 
    // @return indx of beginning of reserved space in array
    //
    __device__
    size_t reserve(unsigned int reqSize)
    {
      // try to reserve reqSize items
      size_t oldSize = myAtomicAdd(size, reqSize);
      
      assert(oldSize <= capacity - reqSize);
      
      return oldSize;
    }

    //
    // @brief write data to the buffer
    // 
    // @param base starting offset for write
    // @param offset to add to base
    // @elt element to write
    //
    __device__
    void put(size_t base, unsigned int offset,
	     const T &elt) const
    {
      data[base + offset] = elt;
    }
    
  private:
    
    size_t capacity;
    size_t *size;
    
    T* data;
    
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
