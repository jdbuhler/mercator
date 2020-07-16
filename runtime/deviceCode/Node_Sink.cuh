#ifndef __NODE_SINK_CUH
#define __NODE_SINK_CUH

//
// @file Node_Sink.cuh
// @brief Node that writes its input to a sink object
//
// MERCATOR
// Copyright (C) 2019 Washington University in St. Louis; all rights reserved.
//

#include "Node.cuh"

#include "io/Sink.cuh"

namespace Mercator  {

  //
  // @class Node_Sink
  // @brief Contains all functions and datatype-dependent info
  //         for a "sink" node.
  //
  // @tparam T type of input item

  //
  template<typename T, 
	   unsigned int THREADS_PER_BLOCK>
  class Node_Sink : 
    public Node<T, 
		0,                 // no output channels
		THREADS_PER_BLOCK> {
    
    using BaseType = Node<T, 
			  0,
			  THREADS_PER_BLOCK>;
    
  private:
    
#ifdef INSTRUMENT_OCC
    using BaseType::occCounter;
#endif
    
  public: 
    
    //
    // @brief constructor
    // @param queueSize - size for input queue
    //
    __device__
    Node_Sink(unsigned int queueSize,
	      Scheduler *scheduler,
	      unsigned int region)
      : BaseType(queueSize, scheduler, region),
	sink(nullptr)
    {
#ifdef INSTRUMENT_OCC
      occCounter.setMaxRunSize(THREADS_PER_BLOCK);
#endif
    }
    
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
    
    
    //
    // @brief prepare for a run of the app's main kernel
    // Set our output sinks
    //
    __device__
    void setOutputSink(Sink<T> * isink)
    {
      sink = isink;
    }
    

    __device__
    unsigned int getMaxInputs() const
    { return THREADS_PER_BLOCK; }
    
    __device__
    unsigned int doRun(const Queue<T> &queue, 
		       unsigned int start,
		       unsigned int limit)
    {
      unsigned int tid = threadIdx.x;
      
      if (limit > 0)
	{
	  //
	  // Consume next nItems data items
	  //
	  
	  __shared__ unsigned int basePtr;
	  if (IS_BOSS())
	    basePtr = sink->reserve(limit);
	  __syncthreads(); // make sure all threads see base ptr
	  
	  // use every thread to copy from our queue to sink
	  for (unsigned int base = 0; base < limit; base += THREADS_PER_BLOCK)
	    {
#ifdef INSTRUMENT_OCC
	      unsigned int vecSize = min(limit - base, THREADS_PER_BLOCK);
	      NODE_OCC_COUNT(vecSize);
#endif

	      int srcIdx = base + tid;
	      
	      if (srcIdx < limit)
		{
		  const T &myData = queue.getElt(start + srcIdx);
		  sink->put(basePtr, srcIdx, myData);
		}
	    }
	}
      
      return limit;
    }
    
  private:

    Sink<T> * sink;
  };
  
}; // namespace Mercator

#endif
