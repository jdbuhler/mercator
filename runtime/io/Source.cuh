#ifndef __SOURCE_CUH
#define __SOURCE_CUH

//
// @file Source.cuh
// device-side source object associated with Node_Source
//
// MERCATOR
// Copyright (C) 2018 Washington University in St. Louis; all rights reserved.
//

#include <cassert>

#include "Buffer.cuh"
#include "Range.cuh"

namespace Mercator {
  
  //
  // POD type holding tagged union of possible source objects;
  // we cannot use virtual functions across host/device boundary
  //
  template <typename T>
  struct SourceData {
    enum Kind { Buffer, Range } kind; // what kind of source are we?
    
    union {
      const BufferData<T> *bufferData; // Buffer object
      const RangeData<T>  *rangeData;  // Range object
    };
  };
  
  
  //
  // A generic Source<T> presents as a read-only array with an
  // associated (external) tail pointer, which is managed with atomic
  // increments to support concurrent claiming of inputs by many
  // processors.
  //
  template <typename T>  
  class Source {
    
  public:
    
    //
    // @brief constructor 
    // 
    // @param isize size of input source array
    // @param itail pointer to a shared tail ptr
    //
    __device__
    Source(size_t isize, size_t *itail)
      : size(isize),
	tail(itail)
    {}
    
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
    virtual
    T get(size_t idx) const = 0;
    
  private:
    
    const size_t size;
    size_t *tail;
    
    //
    // @brief CUDA does not provide an atomic add for size_t, despite
    // its being a 64 bit integer type.  Provide one here.
    //
    // @param address pointer to value to increment
    // @param val size of increment
    //
    __device__ 
    size_t myAtomicAdd(size_t *address, size_t val)
    {
      static_assert(sizeof(size_t) == sizeof(unsigned long long),
		    "ERROR: sizeof(size_t) != sizeof(ULL)");
      
      unsigned long long * address_as_ull =
	(unsigned long long *) address;
      
      unsigned long long v = val;
      
      return (size_t) atomicAdd(address_as_ull, v);
    }
  };
  
  
  //
  // @class SourceBuffer
  // @brief a source that reads from a Buffer
  //
  template<typename T>
  class SourceBuffer : public Source<T> {
    
  public:
    
    //
    // @brief constructor
    //
    // @param bufferData data describing buffer object (copied)
    // @param tail external tail pointer
    //
    __device__
    SourceBuffer(const BufferData<T> *bufferData,
		 size_t *tail)
      : Source<T>(bufferData->size, tail),
	bd(*bufferData)
    {}
    
    //
    // @brief get an element from the buffer
    //
    // @param idx index of element to get
    // @return element gotten
    //
    __device__
    T get(size_t idx) const
    {
      assert(idx <= bd.size);
      
      return bd.data[idx];
    }
    
  private:
    
    const BufferData<T> bd;
  };
  
  // As with the Range object, we need a compilable but not runnable
  // SourceRange for arbitrary types, plus the "real" class for
  // arithmetic types.
  template<typename S, bool b = std::is_arithmetic<S>::value >
  class SourceRange : public Source<S> {
  public:
    __device__
    SourceRange(const RangeData<S> *rangeData,
		size_t *tail)
      : Source<S>(0, nullptr)
    {}
    
    __device__
    S get(size_t idx) const { return *dummy; }
    
    S *dummy;
  };
  
  //
  // @class SourceRange
  // @brief a source that enumerates values in a range
  //
  template<typename S>
  class SourceRange<S, true> : public Source<S> {
    
  public:
    
    //
    // @brief constructor 
    // 
    // @param rangeData object describing range
    // @param tail external tail pointer
    //
    __device__
    SourceRange(const RangeData<S> *rangeData,
		size_t *tail)
      : Source<S>(rangeData->size, tail),
	rd(*rangeData)
    {}
    
    //
    // @brief get a value from the range
    //
    // @param idx index of value in range to get
    // @return value requested
    //
    __device__
    S get(size_t idx) const
    {
      assert(idx <= rd.size);
      
      S v = rd.start + idx * rd.step;
      
      return v;
    }
    
  private:
    
    const RangeData<S> rd;
  };

  
  //
  // Memory suitable for holding a Source object of any subtype
  //
  template <typename T>
  struct alignas(SourceBuffer<T>) alignas(SourceRange<T>) 
    SourceMemory {
    
    // C++11's max is not constexpr.  Sigh.
    static constexpr size_t mymax(size_t a, size_t b)
    { return (a > b ? a : b); }
    
    char c[mymax(sizeof(typename Mercator::SourceBuffer<T>),
		 sizeof(typename Mercator::SourceRange<T>))];
  };
  
}

#endif
