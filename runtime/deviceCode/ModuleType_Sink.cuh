#ifndef __MODULE_TYPE_SINK_CUH
#define __MODULE_TYPE_SINK_CUH

//
// @file ModuleType_Sink.cuh
// @brief Module type that writes its input to a sink object
//
// MERCATOR
// Copyright (C) 2018 Washington University in St. Louis; all rights reserved.
//

#include "ModuleType.cuh"

#include "io/Sink.cuh"

#include "module_options.cuh"

namespace Mercator  {

  //
  // @class ModuleType_Sink
  // @brief Contains all functions and datatype-dependent info
  //         for a "sink" module type.
  //
  // @tparam T type of input item to module

  //
  template<typename T, 
	   unsigned int numInstances,
	   unsigned int THREADS_PER_BLOCK>
  class ModuleType_Sink : 
    public ModuleType< 
    ModuleTypeProperties<T,
			 numInstances,
			 0,                 // no output channels
			 1, 1,              // no run/scatter functions
			 THREADS_PER_BLOCK, // use all threads
			 true,
			 THREADS_PER_BLOCK> > {
    
    typedef ModuleType< ModuleTypeProperties<T,
					     numInstances,
					     0,
					     1, 1,
					     THREADS_PER_BLOCK,
					     true,
					     THREADS_PER_BLOCK> > BaseType;
  public: 
    
    //
    // @brief constructor
    // @param queueSizes - sizes for input queue of each instance
    //
    __device__
    ModuleType_Sink(const unsigned int *queueSizes)
      : BaseType(queueSizes)
    {
      for (unsigned int j = 0; j < numInstances; j++)
	sinks[threadIdx.x] = nullptr;
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
    void setOutputSink(int nodeIdx, Sink<T> * isink)
    {
      sinks[nodeIdx] = isink;
    }
    
  private:
    
    using typename BaseType::InstTagT;
    using          BaseType::NULLTAG;
    
    using BaseType::getFireableCount;
    using BaseType::maxRunSize;
    
#ifdef INSTRUMENT_TIME
    using BaseType::gatherTimer;
    using BaseType::runTimer;
    using BaseType::scatterTimer;
#endif

#ifdef INSTRUMENT_OCC
    using BaseType::occCounter;
#endif

#ifdef INSTRUMENT_COUNTS
    using BaseType::itemCounter;
#endif
    
    Sink<T> * sinks [numInstances];

    //
    // @brief fire the module, consuming pending inputs and
    // moving them directly to the output sinks
    //
    // This version of fire() needs to see into the input queues
    // in a way that run() cannot, so we customize it.
    //
    __device__
    virtual
    void fire()
    {
	//if(IS_BOSS())
	//	printf("SINK CALLED\n");
	this->signalHandler();
      unsigned int tid = threadIdx.x;
      
      MOD_TIMER_START(gather);
      
      // obtain number of inputs that can be consumed by each instance
      unsigned int fireableCount = 
	(tid < numInstances ? getFireableCount(tid) : 0);
      
      // compute progressive sums of items to be consumed in each instance,
      // and replicate these sums in each WARP as Ai.
      using Gather = QueueGather<numInstances>;
      
      unsigned int totalFireable;
      unsigned int Ai = Gather::loadExclSums(fireableCount, totalFireable);  

	if(totalFireable <= 0)
		return;      

      assert(totalFireable > 0);
      
      MOD_OCC_COUNT(totalFireable);
      
      // reserve enough room to hold the fireable data in each output sink
      __shared__ unsigned int basePtrs[numInstances];
      if (tid < numInstances)
	basePtrs[tid] = sinks[tid]->reserve(fireableCount);
      
      __syncthreads(); // make sure all threads see base values
      
      Queue<T> &queue = this->queue; 
      
      // Iterate over inputs to be run in block-sized chunks.
      // Transfer data directly from input queues for each instance
      // to output sinks.
      for (unsigned int base = 0;
	   base < totalFireable; 
	   base += maxRunSize)
	{
	  unsigned int idx = base + tid;
	  InstTagT instIdx = NULLTAG;
	  unsigned int instOffset;
	  
	  // activeWarps = ceil( max run size / WARP_SIZE )
	  unsigned int activeWarps = 
	    (maxRunSize + WARP_SIZE - 1)/WARP_SIZE;
	  
	  // only execute warps that need to pull at least one input value
	  if (tid / WARP_SIZE < activeWarps)
	    {
	      // Compute queue and offset values for each thread's input 
	      Gather::BlockComputeQueues(Ai, idx, instIdx, instOffset);
	    }
	  
	  const T &myData =
	    (idx < totalFireable
	     ? queue.getElt(instIdx, instOffset)
	     : queue.getDummy()); // don't create a null reference

	  MOD_TIMER_STOP(gather);
	  MOD_TIMER_START(scatter);
	  
	  if (idx < totalFireable)
	    {
	      sinks[instIdx]->put(basePtrs[instIdx],
				  instOffset,
				  myData);
	    }
	  
	  MOD_TIMER_STOP(scatter);
	  MOD_TIMER_START(gather);
	}
      
      // protect use of queue->getElt() from changes to head pointer due
      // to release.
      __syncthreads();
      
      // release any items that we consumed in this firing
      if (tid < numInstances)
	{
	  COUNT_ITEMS(fireableCount);
	  queue.release(tid, fireableCount);
	}
      
      MOD_TIMER_STOP(gather);
    }
    
  };

}; // namespace Mercator

#endif
