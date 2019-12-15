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

#include "timing_options.cuh"

namespace Mercator  {

  //
  // @class Node_Sink
  // @brief Contains all functions and datatype-dependent info
  //         for a "sink" node.
  //
  // @tparam T type of input item
  //
  template<unsigned int THREADS_PER_BLOCK,
	   typename T> 
  class Node_Sink : 
    public Node<0,                 // no output channels
		1, 1,              // no run/scatter functions
		THREADS_PER_BLOCK, // use all threads
		true,
		THREADS_PER_BLOCK,
		T> {
    
    using BaseType = Node<0,
			  1, 1,
			  THREADS_PER_BLOCK,
			  true,
			  THREADS_PER_BLOCK,
			  T>;
  public: 
    
    //
    // @brief constructor
    // @param queueSize - size for input queue
    //
    __device__
    Node_Sink(unsigned int queueSize,
	      Scheduler *scheduler)
      : BaseType(queueSize, scheduler),
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
    
    
    //
    // @brief prepare for a run of the app's main kernel
    // Set our output sinks
    //
    __device__
    void setOutputSink(Sink<T> * isink)
    {
      sink = isink;
    }
    
  private:
    
    using BaseType::maxRunSize;
    using BaseType::isFlushing;
    
#ifdef INSTRUMENT_TIME
    using BaseType::inputTimer;
    using BaseType::runTimer;
    using BaseType::outputTimer;
#endif

#ifdef INSTRUMENT_OCC
    using BaseType::occCounter;
#endif

#ifdef INSTRUMENT_COUNTS
    using BaseType::itemCounter;
#endif
    
    Sink<T> * sink;

    //
    // @brief fire the node, consuming pending inputs and
    // moving them directly to the output sink
    //
    
    __device__
    virtual
    void fire()
    {
      unsigned int tid = threadIdx.x;

      TIMER_START(input);
      
      Queue<T> &queue = this->queue; 
      
      unsigned int numToWrite = queue.getOccupancy();
      
      // unless we are flushing all our input, round down to a full
      // ensemble.  Since we are active, if we aren't flushing, we
      // have at least one full ensemble to write.
      if (!isFlushing())
	numToWrite = (numToWrite / maxRunSize) * maxRunSize;
      
      TIMER_STOP(input);
      
      TIMER_START(output);
      
      if (numToWrite > 0)
	{
	  unsigned int basePtr;
	  if (IS_BOSS())
	    basePtr = sink->reserve(numToWrite);
	  basePtr = __shfl_sync(0xffffffff, basePtr, 0);
	  
	  // use every thread to copy from our queue to sink
	  for (int base = 0; base < numToWrite; base += maxRunSize)
	    {
	      int srcIdx = base + tid;
	      
	      if (srcIdx < numToWrite)
		{
		  const T &myData = std::get<0>(queue.getElt(srcIdx));
		  sink->put(basePtr, srcIdx, myData);
		}
	    }
	}
      
      TIMER_STOP(output);
      
      TIMER_START(input);
      
      // we consumed enough input that we are no longer active
      if (IS_BOSS())
	{
	  COUNT_ITEMS(numToWrite);
	  queue.release(numToWrite);
	  
	  this->deactivate();
	}
      
      TIMER_STOP(input);
    }
    
  };

}; // namespace Mercator

#endif
