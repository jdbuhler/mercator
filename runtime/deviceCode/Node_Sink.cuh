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
  template<typename T, 
	   unsigned int THREADS_PER_BLOCK>
  class Node_Sink : 
    public Node< 
    NodeProperties<T,
		   0,                 // no output channels
		   1, 1,              // no run/scatter functions
		   THREADS_PER_BLOCK, // use all threads
		   true,
		   THREADS_PER_BLOCK> > {
    
    typedef Node< NodeProperties<T,
				 0,
				 1, 1,
				 THREADS_PER_BLOCK,
				 true,
				 THREADS_PER_BLOCK> > BaseType;
    
  private:
    
    using BaseType::maxRunSize;
    
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
    
    //
    // @brief fire the node, consuming pending inputs and
    // moving them directly to the output sink
    //

    __device__
    void fire()
    {
      unsigned int tid = threadIdx.x;
      
      const unsigned int maxInputSize = maxRunSize;
      
      TIMER_START(input);
      
      Queue<T> &queue = this->queue; 
      Queue<Signal> &signalQueue = this->signalQueue; 
      
      // # of items available to consume from queue
      unsigned int nDataToConsume = queue.getOccupancy();
      unsigned int nSignalsToConsume = signalQueue.getOccupancy();
      
      unsigned int nCredits = (nSignalsToConsume == 0
			       ? 0
			       : signalQueue.getHead().credit);
      
      // # of items already consumed from queue
      unsigned int nDataConsumed = 0;
      unsigned int nSignalsConsumed = 0;

      // threshold for declaring data queue "empty" for scheduling      
      unsigned int emptyThreshold = (this->isFlushing() 
				     ? 0 
				     : maxInputSize - 1);
      
      bool anyDSActive = false;

      while ((nDataToConsume - nDataConsumed > emptyThreshold ||
	      nSignalsConsumed < nSignalsToConsume) &&
	     !anyDSActive)
	{
#if 0
	  if (IS_BOSS())
	    printf("%d %p %d %d %d %d %d\n", 
		   blockIdx.x, this, 
		   nDataConsumed, nDataToConsume,  
		   nSignalsConsumed, nSignalsToConsume,
		   nCredits);
#endif
	  
	  unsigned int nItems = 
	    (nSignalsConsumed < nSignalsToConsume
	     ? nCredits 
	     : nDataToConsume - nDataConsumed);
	  
	  TIMER_STOP(input);
	  
	  TIMER_START(run);
	  
	  if (nItems > 0)
	    {
	      //
	      // Consume next nItems data items
	      //
	      
	      NODE_OCC_COUNT(nItems);
	      
	      __shared__ unsigned int basePtr;
	      if (IS_BOSS())
		basePtr = sink->reserve(nItems);
	      __syncthreads(); // make sure all threads see base ptr
	      
	      // use every thread to copy from our queue to sink
	      for (int base = 0; base < nItems; base += maxRunSize)
		{
		  int srcIdx = base + tid;
		  
		  if (srcIdx < nItems)
		    {
		      const T &myData = queue.getElt(srcIdx);
		      sink->put(basePtr, srcIdx, myData);
		    }
		}
	      
	      nDataConsumed += nItems;
	    }
	  
	  //
	  // Track credit to next signal, and consume if needed.
	  //
	  if (nSignalsToConsume > 0)
	    {
	      nCredits -= nItems;
	      
	      if (nCredits == 0)
		nCredits = this->handleSignal(nSignalsConsumed++);
	    }
	  
	  TIMER_STOP(run);
	  
	  TIMER_START(input);
	}
      
      // protect code above from queue changes below
      __syncthreads();
      
      if (IS_BOSS())
	{
	  COUNT_ITEMS(nDataConsumed);  // instrumentation
	  
	  queue.release(nDataConsumed);
	  signalQueue.release(nSignalsConsumed);
	  
	  // sink is never output blocked, so we always stop
	  // because of our input lower bound.
	  assert(signalQueue.empty());
	  this->deactivate(); 
	  
	  this->clearFlush(); // disable flushing
	}
      
      TIMER_STOP(input);
    }
    
  private:

    Sink<T> * sink;
  };
  
}; // namespace Mercator

#endif
