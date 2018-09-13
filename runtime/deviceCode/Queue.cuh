#ifndef __QUEUE_CUH
#define __QUEUE_CUH
//
// @file Queue.cuh
// @brief MERCATOR queue object
//

#include <cassert>

#include "device_config.cuh"

namespace Mercator  {

  //
  // @class Queue
  // @brief FIFO queue holding items that are in transit from one
  //         node to the next.
  //
  // Note that one Queue is actually an array of (# instances) queues.
  // Operations to reserve or release space on the Queue are intended
  // to run simultaneously on its ocmponent queues.
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
  class Queue  {
  
  public:

    //
    // @brief Constructor.
    //
    // @param inumInstances Number of instances this queue holds
    // @param icapacity capacity for each instance's queue
    //
    __device__
    Queue(unsigned int inumInstances, const unsigned int *capacities)
      : numInstances(inumInstances)
    {
      totalCapacity = 0;
      for (unsigned int i = 0; i < numInstances; ++i)
	{
	  dataSizes[i] = capacities[i] + 1;
	  heads[i]     = 0;
	  tails[i]     = 0;
	  
	  data[i]      = new T [dataSizes[i]];
	  
	  // ensure allocation succeeded
	  assert(data[i] != nullptr);
	  
	  totalCapacity += capacities[i];
	}
    }
    
    
    __device__
    ~Queue()
    {
      for (unsigned int i = 0; i < numInstances; i++)
	delete [] data[i];
    }
    
    
    //
    // @ brief return total capacity of all our component queues
    //
    __device__
    unsigned int getTotalCapacity() const
    { return totalCapacity; }

    //
    // @brief get capacity of queue
    //
    // @param instIdx instance to query
    //
    // NB: capacity is < actual allocated size to support
    // efficient circular queue operations
    //
    __device__
    unsigned int getCapacity(unsigned int instIdx) const
    {
      assert(instIdx < numInstances);
      return dataSizes[instIdx] - 1;
    }

    //
    // @brief get occupancy of queue
    //
    // @param instIdx instance to query
    //
    __device__
    unsigned int getOccupancy(unsigned int instIdx) const 
    { 
      assert(instIdx < numInstances);
      
      unsigned int head = heads[instIdx];
      unsigned int tail = tails[instIdx];
      
      return (tail - head + (tail < head ? dataSizes[instIdx] : 0));      
    }
    
    //
    // @brief return true iff queue is empty
    //
    // @param instIdx instance to query
    //
    __device__
    bool empty(unsigned int instIdx) const
    { return (heads[instIdx] == tails[instIdx]); }
    
    
    //
    // @brief reserve space at the tail of the queue for elts
    //
    // Should be called SINGLE-THREADED.
    //
    // @param instIdx instance to queue
    // @param nElts number of elements to reserve
    // @return index of start of reserved space
    //
    __device__
    unsigned int reserve(unsigned int instIdx, unsigned int nElts)
    {
      assert(instIdx < numInstances);
      
      assert(getOccupancy(instIdx) <= getCapacity(instIdx) - nElts);
      
      unsigned int oldTail = tails[instIdx];
      
      tails[instIdx] = 
	addModulo(tails[instIdx], nElts, dataSizes[instIdx]);
      
      return oldTail;
    }
    
    //
    // @brief release space occupied by elements at the head of the queue
    //
    // Should be called SINGLE-THREADED.
    //
    // @param instIdx instance to queue
    // @param nElts number of elements to release
    //
    __device__
    void release(unsigned int instIdx, unsigned int nElts)
    {
      assert(instIdx < numInstances);
      
      assert(getOccupancy(instIdx) >= nElts);
      
      heads[instIdx] = 
	addModulo(heads[instIdx], nElts, dataSizes[instIdx]);
    }
    
    
    //
    // @brief write an element to the specified queue location
    //        base must be a value previously returned by reserve()
    //        
    // May be called multithreaded; does NOT reserve space
    //
    // @param instIdx instance to queue
    // @param base base pointer for write (returned by reserve()
    // @param offset offset of element to write relative to base
    // @param elt value to write
    //
    __device__
    void putElt(unsigned int instIdx, 
		unsigned int base,
		unsigned int offset,
		const T &elt) const
    {
      assert(instIdx < numInstances);
      
      unsigned int myIdx = addModulo(base, offset, dataSizes[instIdx]);
      
      data[instIdx][myIdx] = elt;
    }
    
    //
    // @brief read an element from a queue location, specified as
    //        an offset relative to the queue's head
    //        
    // May be called multithreaded; does NOT release space
    //
    // @param instIdx instance to queue
    // @param offset offset of element to read relative to head
    // @return value read
    //
    __device__
    const T &getElt(unsigned int instIdx, 
		    unsigned int offset) const
    {
      assert(instIdx < numInstances);
      
      assert(getOccupancy(instIdx) > offset);
      
      unsigned int head = heads[instIdx];
      unsigned int myIdx = addModulo(head, offset, dataSizes[instIdx]);
      
      return data[instIdx][myIdx]; 
    }
    
    
    //
    // @brief return a reference to an actual item of type T that
    // will never be dereferenced.  This is useful to avoid creating
    // a null reference (which invokes undefined behavior if we touch
    // it when caling run) where we need an "invalid" element.
    //
    __device__
    const T &getDummy() const
    { return data[0][0]; }
    
  private:

    const unsigned int numInstances;  // # instances in this queue
    
    // number of instances per module is limited to <= WARP_SIZE
    
    T* data[WARP_SIZE];                // queue elements
    unsigned int dataSizes[WARP_SIZE]; // space allocated
    unsigned int heads[WARP_SIZE];   // head ptr -- pts to next *available elt*
    unsigned int tails[WARP_SIZE];   // tail ptr -- pts to next *free slot*
    
    unsigned int totalCapacity; // total capacity of all queues
    
    // add two numbers x, y modulo m
    // we assume that x and y are each < m, so we can implement
    // modulus with one conditional subtraction rather than division.
    __device__
    static unsigned int addModulo(unsigned int x, unsigned int y, unsigned int m)
    {
      unsigned int s = x + y;
      s -= (s >= m ? m : 0);
      return s;
    }
    
  };  // end class Queue

}   // end Mercator namespace

#endif
