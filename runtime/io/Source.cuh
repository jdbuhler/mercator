#ifndef __SOURCE_CUH
#define __SOURCE_CUH

//
// @file Source.cuh
// device-side source object associated with Node_Source
//
// MERCATOR
// Copyright (C) 2018 Washington University in St. Louis; all rights reserved.
//

#include <climits>
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
  
  // We use get() to return an element from the source array.  For
  // efficiency reasons, we would like to return scalar types such
  // as integers by value, but return complex types like structures
  // by reference.  Moreover, range sources, which materialize their
  // integer-valued elements on the fly, *must* return them by value.
  // Hence, a Source<T> has an exported element type EltT that is
  // either T or const T& as appropriate for T, and the source's get() 
  // uses EltT as its return type.
  //
  // NB: in the future, we plan to deprecate any type for the source
  // other than size_t and make the user access input streams of
  // complex types by indexing into an array passed as a parameter.
  // At that point, this value vs reference hack will move into
  // the Node internals to permit nodes to export a uniform
  // doRun() interface that works whether inputs are ints from
  // a Source or arbitrary values from a Queue.
  
  template <typename T, typename Enable = void>
  class SourceBase;

  template <typename T>
  class SourceBase<T, std::enable_if_t<std::is_scalar<T>::value> > {
  public:
    using EltT = T;
    
    __device__
    virtual
    EltT get(size_t idx) const = 0;
  };

  template <typename T>
  class SourceBase<T, std::enable_if_t<!std::is_scalar<T>::value> > {
  public:
    using EltT = T&;
    
    __device__
    virtual
    EltT get(size_t idx) const = 0;
  };
  
  // Actual Source class begins here
  
  template <typename T>  
  class Source : public SourceBase<T> {
    
  public:
    
    // type returned by get()
    using EltT = typename SourceBase<T>::EltT;
    
    //
    // @brief constructor 
    // 
    // @param isize size of input source array
    // @param itail pointer to a shared tail ptr
    //
    __device__
    Source(size_t isize, size_t *itail)
      : size(isize),
	tail(itail),
	reqLimit(max((size_t) 1, size / gridDim.x))
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
    
    //
    // @brief get an element from the array
    //
    __device__
    virtual
    typename SourceBase<T>::EltT get(size_t idx) const = 0;
    
    //
    // @brief return a dummy element we don't care about
    //
    __device__
    typename SourceBase<T>::EltT getDummy() const { return get(0); }
    
  private:
    
    const size_t size;
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
  
  
  //
  // @class SourceBuffer
  // @brief a source that reads from a Buffer
  //
  template<typename T>
  class SourceBuffer : public Source<T> {
    
  public:
    
    using EltT = typename Source<T>::EltT;
    
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
    // @return element gotten (by value if scalar, 
    //   or by const reference othereise)
    //
    __device__
    EltT get(size_t idx) const
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
    
    using EltT = typename Source<S>::EltT;
    
    __device__
    SourceRange(const RangeData<S> *rangeData,
		size_t *tail)
      : Source<S>(0, nullptr)
    {}
    
    __device__
    EltT get(size_t idx) const { return *dummy; }
    
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
    // @return value requested (always the case for arithmetic type)
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
