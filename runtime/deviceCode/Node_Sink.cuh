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
      if (!isFlushing)
	numToWrite = (numToWrite / maxRunSize) * maxRunSize;
      
      // # of total items that need to be written
      unsigned int numTotalToWrite = 0;

      TIMER_STOP(input);
      
      TIMER_START(output);
      
      while(this->numSignalsPending() > 0)
	{
		numToWrite = this->currentCreditCounter;
	      if (numToWrite > 0)
		{
		  __shared__ unsigned int basePtr;
		  if (IS_BOSS())
		    basePtr = sink->reserve(numToWrite);
		  __syncthreads(); // make sure all threads see base ptr
		  
		  // use every thread to copy from our queue to sink
		  for (int base = 0; base < numToWrite; base += maxRunSize)
		    {
		      int srcIdx = base + tid;
		      
		      if (srcIdx < numToWrite)
			{
			  const T &myData = queue.getElt(srcIdx);
			  sink->put(basePtr, srcIdx, myData);
			}
		    }
		numTotalToWrite += numToWrite;
		}
		this->currentCreditCounter -= numToWrite;

		__syncthreads();

		//stimcheck: We don't care about the ds signal queues being full here, since there are no ds signal queues.
		this->signalHandler();	
	}
	

	__syncthreads();

      numToWrite = queue.getOccupancy();
            
      // unless we are flushing all our input, round down to a full
      // ensemble.  Since we are active, if we aren't flushing, we
      // have at least one full ensemble to write.
      if (!isFlushing)
	numToWrite = (numToWrite / maxRunSize) * maxRunSize;

	//Perform normal AFIE Scheduling once all signals are processed.
      if (numToWrite > 0)
	{
	  __shared__ unsigned int basePtr;
	  if (IS_BOSS())
	    basePtr = sink->reserve(numToWrite);
	  __syncthreads(); // make sure all threads see base ptr
	  
	  // use every thread to copy from our queue to sink
	  for (int base = 0; base < numToWrite; base += maxRunSize)
	    {
	      int srcIdx = base + tid;
	      
	      if (srcIdx < numToWrite)
		{
		  const T &myData = queue.getElt(srcIdx);
		  sink->put(basePtr, srcIdx, myData);
		}
	    }
	    numTotalToWrite += numToWrite;
	}

	//numTotalToWrite += numToWrite;

      TIMER_STOP(output);
      
      TIMER_START(input);
      
      // we consumed enough input that we are no longer active
      if (IS_BOSS())
	{
	  //COUNT_ITEMS(numToWrite);
	  //queue.release(numToWrite);
	  COUNT_ITEMS(numTotalToWrite);
	  queue.release(numTotalToWrite);
	  
	  this->deactivate();
	}
      
      TIMER_STOP(input);
    }
    
  };

}; // namespace Mercator

#endif
