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
  //  - However, we also provide an "unsafe" write that writes an element
  //     at an offset relative to the current tail.  This can only be
  //     used if there is no possibility that a second write will
  //     happen before the first write has been committed with a reserve()
  //     call.
  //
  // @tparam T Type of data item held in this Queue
  //
  template<typename T>
  class Queue {
  
  public:

    //
    // @brief Constructor.
    //
    // @param icapacity capacity of queue
    //
    __device__
      Queue(unsigned int capacity)
    {
      dataSize  = capacity + 1;
      head      = 0;
      tail      = 0;
      
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
    // @brief get free space on queue
    //
    __device__
    unsigned int getFreeSpace() const
    {
      return getCapacity() - getOccupancy();
    }

    //
    // @brief get capacity of queue
    //
    // NB: capacity is < actual allocated size to support
    // efficient circular queue operations
    //
    __device__
    unsigned int getCapacity() const
    {
      return dataSize - 1;
    }

    //
    // @brief get occupancy of queue
    //
    __device__
    unsigned int getOccupancy() const 
    { 
      return (tail - head + (tail < head ? dataSize : 0));      
    }
    
    //
    // @brief return true iff queue is empty
    //
    __device__
    bool empty() const
    { return (head == tail); }
    
    
    //
    // @brief reserve space at the tail of the queue for elts
    //
    // Should be called SINGLE-THREADED.
    //
    // @param nElts number of elements to reserve
    // @return index of start of reserved space
    //
    __device__
    unsigned int reserve(unsigned int nElts)
    {
      assert(getOccupancy() <= getCapacity() - nElts);
      
      unsigned int oldTail = tail;
      
      tail = addModulo(tail, nElts, dataSize);
      
      return oldTail;
    }
    
    //
    // @brief release space occupied by elements at the head of the queue
    //
    // Should be called SINGLE-THREADED.
    //
    // @param nElts number of elements to release
    //
    __device__
    void release( unsigned int nElts)
    {
      assert(getOccupancy() >= nElts);
      
      head = addModulo(head, nElts, dataSize);
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
    const T &getElt(unsigned int offset) const
    {
      assert(getOccupancy() > offset);
      
      unsigned int myIdx = addModulo(head, offset, dataSize);
      
      return data[myIdx]; 
    }
    
    
    //
    // @brief return a reference to an actual item of type T that
    // will never be dereferenced.  This is useful to avoid creating
    // a null reference (which invokes undefined behavior if we touch
    // it when caling run) where we need an "invalid" element.
    //
    __device__
    const T &getDummy() const
    { return data[0]; }
    
  private:

    T* data;
    unsigned int dataSize; // space allocated
    unsigned int head;     // head ptr -- pts to next *available elt*
    unsigned int tail;     // tail ptr -- pts to next *free slot*
    
    // add two numbers x, y modulo m
    // we assume that x and y are each < m, so we can implement
    // modulus with one conditional subtraction rather than division.
    __device__
    static unsigned int addModulo(unsigned int x, 
				  unsigned int y, 
				  unsigned int m)
    {
      unsigned int s = x + y;
      s -= (s >= m ? m : 0);
      return s;
    }
    
  };  // end class Queue

}   // end Mercator namespace

#endif
