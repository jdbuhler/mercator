#ifndef __DEVUTIL_H
#define __DEVUTIL_H

//
// DEVUTIL.H
// Device-side utilities
//
// MERCATOR
// Copyright (C) 2021 Washington University in St. Louis; all rights reserved.
//

#include <type_traits>

namespace Mercator {
  
  //
  // define a type that is by value for scalar types or by
  // reference for non-scalar types
  //
  
  template <typename T, typename Enable = void>
  struct ReturnType;
  
  template <typename T>
  struct ReturnType<T, std::enable_if_t<std::is_scalar<T>::value> > 
  {
    using EltT = T;
  };
  
  template <typename T>
  struct ReturnType<T, std::enable_if_t<!std::is_scalar<T>::value> > 
  {
    using EltT = T&;
  };
  
  /////////////////////////////////////////////////////////////////
  
  //
  // define a type that is by value for scalar types or by
  // *const* reference for non-scalar types.  Allow the caller
  // to specify a non-default second argument "true" to force
  // the type to be by value.
  //
  
  template <typename T, bool forceByValue = false, typename Enable = void>
  struct ConstReturnType;
  
  template <typename T, bool forceByValue>
  struct ConstReturnType<T, forceByValue, 
			 std::enable_if_t<forceByValue || 
					  std::is_scalar<T>::value>>
  {
    using EltT = T;
  };
  
  template <typename T, bool forceByValue>
  struct ConstReturnType<T, forceByValue, 
			 std::enable_if_t<!forceByValue && 
					  !std::is_scalar<T>::value>>
  {
    using EltT = T const &;
  };
  
}

#endif
