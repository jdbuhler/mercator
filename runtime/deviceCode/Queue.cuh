#ifndef __QUEUE_CUH
#define __QUEUE_CUH

//
// @file Queue.cuh
// @brief MERCATOR queue object
//
// MERCATOR
// Copyright (C) 2019 Washington University in St. Louis; all rights reserved.
//

#include <cassert>
#include <cstdio>

#include "QueueBase.cuh"
#include "tuple_util.cuh"

#include "device_config.cuh"

namespace Mercator  {

  //
  // @class Queue
  // @brief FIFO queue holding items that are in transit from one
  //         node to the next.
  //
  // Notes on safety:
  //  - Queue is not concurrent -- we assume that different callers cannot
  //     call get/put/reserve/release asynchronously.
  //  - There are separate calls to reserve space at the tail and actually
  //     write to that space, and separate calls to read from the head
  //     and release storage from the head.  Hence, it is posisble to
  //     use the queue in such a way that we always reserve before writing
  //     and always read before releasing.
  //
  // @tparam T Type of data item held in this Queue
  //
  template<typename... Ts>
  class Queue : public QueueBase {
  
  public:

    //
    // @brief Constructor.
    //
    // @param icapacity capacity of queue
    //
    __device__
      Queue(unsigned int capacity)
	: QueueBase(capacity)
    {
      // allocate an array of n elts and save ref to it in pointer p
      auto alloc =
	[this](auto * &p)
      {
        using EltType = typename std::remove_reference<decltype(*p)>::type;
        p = new EltType [dataSize];

	// ensure allocation succeeded
	if (p == nullptr)
	  {
	    printf("ERROR: failed to allocate queue [block %d]\n",
		   blockIdx.x);
	    
	    crash();
	  }
      };
      
      tuple_foreach(data, alloc);
    }
    
    
    __device__
    ~Queue()
    {
      // free an array pointed to by p
      auto dealloc = [](auto *p) { delete [] p; };
      
      tuple_foreach(data, dealloc);
    }
        

    //
    // @brief write all elements of an item to the specified queue
    //        location base must be a value previously returned by
    //        reserve()
    //        
    // May be called multithreaded; does NOT reserve space
    //
    // @param base base pointer for write (returned by reserve())
    // @param offset offset of element to write relative to base
    // @param elt value to write
    //
    __device__
    void putElt(unsigned int base,
		unsigned int offset,
		const Ts&... elts)
    {
      unsigned int myIdx = addModulo(base, offset, dataSize);
      
      putEltInternal(myIdx,
		     std::make_index_sequence< std::tuple_size< decltype(data) >::value >(),
		     elts...);

    }

    //
    // @brief read an item from a queue location, specified as
    //        an offset relative to the queue's head
    //        
    // May be called multithreaded; does NOT release space
    //
    // @param offset offset of element to read relative to head
    // @return tuple of const refs to elements of item
    //
    __device__
    std::tuple<const Ts&...> getElt(unsigned int offset) const
    {
      assert(getOccupancy() > offset);
      
      unsigned int myIdx = addModulo(head, offset, dataSize);
      
      return getEltInternal(myIdx,
			    std::make_index_sequence< std::tuple_size< decltype(data) >::value >());  
    }
    

    //
    // @brief return a reference to an actual item of type T that
    // will never be dereferenced.  This is useful to avoid creating
    // a null reference (which invokes undefined behavior if we touch
    // it when caling run) where we need an "invalid" element.
    //
    __device__
    std::tuple<const Ts&...> getDummy() const
    {
      return getElt(0);
    }
    
  private:
    
    std::tuple<Ts* ...> data;
    
    
    //
    // @brief internal putElt() helper tht maps writes of item's elements
    // across each member of the tuple 'data'.
    //
    template <size_t... N>
    __device__
    void putEltInternal(unsigned int myIdx,
			std::index_sequence<N...>, const Ts&... elt)
    {
      auto ignore = { (std::get<N>(data)[myIdx] = elt, nullptr)... };
      (void) ignore;
    }

    //
    // @brief internal getElt() helper tht maps reads of item's elements
    // across each member of the tuple 'data' and forms a result tuple.
    //
    template <size_t... N>
    __device__
    std::tuple<const Ts&...> getEltInternal(unsigned int myIdx,
					    std::index_sequence<N...>) const
      
    {
      return std::tuple<const Ts&...>{std::get<N>(data)[myIdx]...};
    }
    
  };  // end class Queue

}   // end Mercator namespace

#endif
