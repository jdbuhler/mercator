//
// TUPLE_UTIL.CUH
// A magic iterator that applies a fucntor to each element of a tuple.
// C++17 defines apply() to do this, but we are stuck with C+14 for CUDA.
//

#ifndef __TUPLE_UTIL_CUH
#define __TUPLE_UTIL_CUH

#include <tuple>
#include <utility>


namespace detail {
  template <typename Tuple, typename Func, size_t... index>
  __device__
  inline void tuple_foreach_internal(Tuple &tuple, const Func &func, 
				     std::index_sequence<index...>)
  {
    auto ignore = { (func(std::get<index>(tuple)), nullptr)... };
    (void) ignore;
  }
}

template <typename Tuple, typename Func>
__device__
inline void tuple_foreach(Tuple &tuple, const Func &func)
{
  detail::tuple_foreach_internal(tuple, func, 
				 std::make_index_sequence< std::tuple_size< Tuple >::value >());
}

#endif
