#ifndef __RANGE_H
#define __RANGE_H

//
// @file Range.cuh
// Compact description of a numerical range as an input to a MERCATOR app
//
#include <cstddef>
#include <type_traits>

namespace Mercator {
  
  // POD type hold the essential properties of a range, for
  // transfer between host and device
  template <typename S>
  struct RangeData {
    S start;      // first element in range
    S step;       // distance between elements (may be +/- for signed S)
    size_t size;  // # of elements
  };
  
  // The parameter type S of a range must be an arithmetic type. 
  // To support polymorphism, we need an implementation that
  // at least compiles (but does not run) for non-arithmetic types.
  template<typename S, bool b = std::is_arithmetic<S>::value >
  class Range {
  public:
    RangeData<S> *getData() const { return nullptr; }
  };
  
  // Use partial template specialization to generate proper code
  // only when S is an arithmetic type.
  template<typename S>
  class Range<S, true> {
    
  public:
    
    //
    // @brief constructor takes range bounds, step and computes size.
    //   Allocates a managed object to facilitate host-device communication
    //   about the range.
    //
    // @param start start of range
    // @param end end of range
    // @param step step size for range
    // @param includeEndpoint true if range is [start, end]; 
    //           false if [start, end)
    Range(S start, S end, S step, bool includeEndpoint = false)
    {
      cudaMallocManaged(&rangeData, sizeof(RangeData<S>));
      
      rangeData->start = start;
      rangeData->step = step;
      
      // make sure range is bounded
      assert((end - start) / step >= 0);
      
      rangeData->size = 
	(includeEndpoint
	 ? (size_t) ((end - start) / step) + 1
	 : (size_t) ((end - start + step/2) / step));
    }
    
    //
    // @brief return number of elements in the range
    //
    size_t size() const 
    { return rangeData->size; }
    
    //
    // @brief extract the managed data object from the range
    //
    RangeData<S> *getData() const { return rangeData; }
    
  private:
    
    RangeData<S> *rangeData;
    
  };
}

#endif
