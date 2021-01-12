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

  //
  template<typename T, 
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
      : BaseType(parentArena),
	sink(nullptr)
    {}
    
    //
    // @brief construct a Sink object from the raw data passed down
    // to the device.
    //
    // @param sinkData sink data passed from host to device
    // @return a Sink object whose subtype matches the input data
    //
    __device__
    static
    Sink<T> *createSink(const SinkData<T> &sinkData,
			SinkMemory<T> *mem)
    {
      Sink<T> *sink;
      
      switch (sinkData.kind)
	{
	case SinkData<T>::Buffer:
	  sink = new (mem) SinkBuffer<T>(sinkData.bufferData);
	  break;
	}
      
      return sink;
    }
    
    ////////////////////////////////////////////////////////////
    
    //
    // @brief set our sink at the beginning of a run of the main kernel
    //
    __device__
    void setOutputSink(Sink<T>* isink)
    { 
      assert(IS_BOSS());
      
      sink = isink; 
    }
    
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
    unsigned int doRun(const Queue<T> &queue, 
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
	basePtr = sink->reserve(limit);
      
      __syncthreads(); // END WRITE basePtr
      
      // use every thread to copy from our queue to sink
      for (unsigned int base = 0; 
	   base < limit; 
	   base += THREADS_PER_BLOCK)
	{
#ifdef INSTRUMENT_OCC
	  unsigned int vecSize = min(limit - base, THREADS_PER_BLOCK);
	  NODE_OCC_COUNT(vecSize);
#endif
	  
	  int srcIdx = base + tid;
	  
	  if (srcIdx < limit)
	    {
	      const typename Queue<T>::EltT myData = 
		queue.get(start + srcIdx);
	      sink->put(basePtr, srcIdx, myData);
	    }
	}
      
      return limit;
    }
    
  private:

    Sink<T>* sink;

  };
  
}; // namespace Mercator

#endif
