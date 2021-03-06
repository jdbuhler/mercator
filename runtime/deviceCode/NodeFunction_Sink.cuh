#ifndef __NODEFUNCTION_SINK_CUH
#define __NODEFUNCTION_SINK_CUH

//
// @file NodeFunction_Sink.cuh
// @brief Node function that writes its input to a sink object
//
// MERCATOR
// Copyright (C) 2021 Washington University in St. Louis; all rights reserved.
//

#include "NodeFunction.cuh"

#include "io/Sink.cuh"

namespace Mercator  {

  //
  // @class NodeFunction_Sink
  // @brief Contains all functions and datatype-dependent info
  //         for a "sink" node.
  //
  // @tparam T type of input item
  // @tparam InputView type of input view passed to doRun()
  // @tparam THREADS_PER_BLOCK constant giving thread block size
  //
  template<typename T, 
	   typename InputView,
	   unsigned int THREADS_PER_BLOCK>
  class NodeFunction_Sink : public NodeFunction<0> {

    using BaseType = NodeFunction<0>;
    
    using BaseType::node;
    
  public: 
    
    ///////////////////////////////////////////////////////
    // INIT/CLEANUP KERNEL FUNCIIONS
    ///////////////////////////////////////////////////////
    
    //
    // @brief constructor
    // @param queueSize - size for input queue
    //
    __device__
    NodeFunction_Sink(RefCountedArena *parentArena)
      : BaseType(parentArena)
    {}
    
    ////////////////////////////////////////////////////////////
    
    //
    // doRun() writes full thread-widths of inputs at a time
    //
    static const unsigned int inputSizeHint = THREADS_PER_BLOCK;
    
    //
    // @brief function stub to execute the function code specific
    // to this node.  This function does NOT remove data from the
    // queue.
    //
    // @param queue data queue containing items to be consumed
    // @param start index of first item in queue to consume
    // @param limit max number of items that this call may consume
    // @return number of items ACTUALLY consumed (may be 0).
    //
    __device__
    unsigned int doRun(const InputView &view,
		       size_t start,
		       unsigned int limit)
    {
      unsigned int tid = threadIdx.x;
      
      //
      // Consume next nItems data items
      //
      
      __syncthreads(); // BEGIN WRITE basePtr (ds sink ptr is not read)
      
      __shared__ size_t basePtr;
      if (IS_BOSS())
	basePtr = sink.reserve(limit);
      
      __syncthreads(); // END WRITE basePtr
      
      // use every thread to copy from view to sink
      for (unsigned int base = 0; 
	   base < limit; 
	   base += THREADS_PER_BLOCK)
	{
#ifdef INSTRUMENT_OCC
	  unsigned int vecSize = min(limit - base, THREADS_PER_BLOCK);
	  NODE_OCC_COUNT(vecSize, THREADS_PER_BLOCK);
#endif
	  
	  int srcIdx = base + tid;
	  
	  if (srcIdx < limit)
	    {
	      const typename InputView::EltT myData = 
		view.get(start + srcIdx);
	      sink.put(basePtr, srcIdx, myData);
	    }
	}
      
      return limit;
    }
    
  protected:

    Sink<T> sink;

  };
  
}; // namespace Mercator

#endif
