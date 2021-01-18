#ifndef __QUEUE_CUH
#define __QUEUE_CUH

//
// @file Queue.cuh
// @brief MERCATOR queue object
//
// MERCATOR
// Copyright (C) 2020 Washington University in St. Louis; all rights reserved.
//

#include <type_traits>
#include <cassert>

#include "QueueBase.cuh"

#include "device_config.cuh"

namespace Mercator  {

    template <typename T, typename Enable = void>
  class TypedQueueBase;

  template <typename T>
  class TypedQueueBase<T, std::enable_if_t<std::is_scalar<T>::value> > {
  public:
    using EltT = T;
  };

  template <typename T>
  class TypedQueueBase<T, std::enable_if_t<!std::is_scalar<T>::value> > {
  public:
    using EltT = T&;
  };

  
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
  template<typename T>
  class Queue : public QueueBase, public TypedQueueBase<T> {

  public:

    using EltT = typename TypedQueueBase<T>::EltT;
    
    ///////////////////////////////////////////////////////
    // INIT/CLEANUP KERNEL FUNCIIONS
    ///////////////////////////////////////////////////////
    
    //
    // @brief Constructor.
    //
    // @param icapacity capacity of queue
    //
    __device__
      Queue(unsigned int capacity)
	: QueueBase(capacity)
    {
      data = new T [dataSize];
      
      // ensure allocation succeeded
      if (data == nullptr)
	{
	  printf("ERROR: failed to allocate queue [block %d]\n",
		 blockIdx.x);
	  
	  crash();
	}
    }
    
    
    __device__
    ~Queue()
    {
      delete [] data;
    }
    
    ////////////////////////////////////////////////////////////
    
    //
    // @brief write an element to the specified queue location
    //        base must be a value previously returned by reserve()
    //        
    // May be called MULTITHREADED to write multiple values
    //
    // @param base base pointer for write (returned by reserve()
    // @param offset offset of element to write relative to base
    // @param elt value to write
    //
    __device__
    void put(size_t base,
	     unsigned int offset,
	     const T &elt) const
    {
      unsigned int myIdx = addModulo(base, offset, dataSize);
      
      data[myIdx] = elt;
    }
    
    //
    // @brief read an element from a queue location, specified as
    //        an offset relative to the queue's head
    //         
    // @param offset offset of element to read relative to head
    // @return value read
    //
    __device__
    EltT get(size_t offset) const
    {
      assert(getOccupancy() > offset);
      
      unsigned int myIdx = addModulo(head, offset, dataSize);
      
      return data[myIdx]; 
    }
    
    //
    // @brief read the element at the head of the queue
    //
    __device__
    EltT getHead() const
    { return get(0); }
    
    //
    // @brief return a reference to an actual item of type T that
    // will never be dereferenced.  This is useful to avoid creating
    // a null reference (which invokes undefined behavior if we touch
    // it when caling run) where we need an "invalid" element.
    //
    __device__
    const EltT getDummy() const
    { return data[0]; }
    
    
    //
    // @brief reserve and put an element at the tail of the queue in a
    // single call.
    //
    __device__
    T &enqueue(const T &v)
    {
      assert(IS_BOSS());
      assert(getOccupancy() < getCapacity());
      
      data[tail] = v;
      
      unsigned int oldTail = tail;
      tail = addModulo(tail, 1, dataSize);
      
      return data[oldTail];
    }
    
    //
    // @brief get an element and release its space in a single call.
    //
    __device__
    T dequeue()
    {
      assert(IS_BOSS());
      assert(getOccupancy() > 0);
      
      T &v = data[head];
      head = addModulo(head, 1, dataSize);
      
      return v;
    }
    
  private:
    
    T* data;               // actual queue space
  };  // end class Queue

}   // end Mercator namespace

#endif
