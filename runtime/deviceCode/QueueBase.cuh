#ifndef __QUEUEBASE_CUH
#define __QUEUEBASE_CUH

//
// @file QueueBase.cuh
// @brief MERCATOR queue object
//
// MERCATOR
// Copyright (C) 2020 Washington University in St. Louis; all rights reserved.
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
  //
  class QueueBase {
  
  public:
    
    ///////////////////////////////////////////////////////
    // INIT/CLEANUP KERNEL FUNCIIONS
    ///////////////////////////////////////////////////////
    
    //
    // @brief Constructor.
    //
    // @param icapacity capacity of queue
    //
    __device__
      QueueBase(unsigned int capacity)
	: dataSize(capacity + 1),
	  head(0), tail(0)
    {}
    
    
    __device__
    virtual ~QueueBase() {}
    
    /////////////////////////////////////////////////////////
    
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
    size_t reserve(unsigned int nElts)
    {
      assert(IS_BOSS());
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
    void release(unsigned int nElts)
    {
      assert(IS_BOSS());
      assert(getOccupancy() >= nElts);
      
      head = addModulo(head, nElts, dataSize);
    }
    
  protected:

    const unsigned int dataSize; // space allocated
    
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
    
  };  // end class QueueBase

}   // end Mercator namespace

#endif
