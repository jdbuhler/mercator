#ifndef __QUEUE_CUH
#define __QUEUE_CUH

//
// @file Queue.cuh
// @brief MERCATOR queue object
//
// MERCATOR
// Copyright (C) 2020 Washington University in St. Louis; all rights reserved.
//

#include <cassert>

#include "QueueBase.cuh"

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
  template<typename T>
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
    
    
    //
    // @brief write an element to the specified queue location
    //        base must be a value previously returned by reserve()
    //        
    // May be called multithreaded; does NOT reserve space
    //
    // @param base base pointer for write (returned by reserve()
    // @param offset offset of element to write relative to base
    // @param elt value to write
    //
    __device__
    void putElt(unsigned int base,
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
    // May be called multithreaded; does NOT release space
    //
    // @param offset offset of element to read relative to head
    // @return value read
    //
    __device__
    T &getElt(unsigned int offset) const
    {
      unsigned int db = getOccupancy();
      assert(getOccupancy() > offset);
      
      unsigned int myIdx = addModulo(head, offset, dataSize);
      
      return data[myIdx]; 
    }
    
    __device__
    T &getHead() const
    { return getElt(0); }
    
    //
    // @brief return a reference to an actual item of type T that
    // will never be dereferenced.  This is useful to avoid creating
    // a null reference (which invokes undefined behavior if we touch
    // it when caling run) where we need an "invalid" element.
    //
    __device__
    const T &getDummy() const
    { return data[0]; }
    
    
    //
    // @brief reserve and put an element at the tail of the queue in a
    // single call.
    //
    __device__
    T &enqueue(const T &v)
    {
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
