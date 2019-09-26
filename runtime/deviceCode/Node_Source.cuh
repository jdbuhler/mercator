#ifndef __NODE_SOURCE_CUH
#define __NODE_SOURCE_CUH

//
// @file Node_Source.cuh
// @brief Node that gets its input from a source object
//
// MERCATOR
// Copyright (C) 2019 Washington University in St. Louis; all rights reserved.
//

#include <climits>

#include "Node.cuh"

#include "io/Source.cuh"

#include "timing_options.cuh"

namespace Mercator  {

  //
  // @class Node_Source
  // @brief Contains all functions and datatype-dependent info
  //         for a "source" node.
  //
  // @tparam T type of input item
  // @tparam numChannels  number of output channels
  //
  template<typename T, 
	   unsigned int numChannels,
	   unsigned int THREADS_PER_BLOCK>
  class Node_Source : 
    public Node< 
    NodeProperties<T,
		   numChannels,
		   1, 1,              // no run/scatter functions
		   THREADS_PER_BLOCK, // use all threads
		   true,              
		   THREADS_PER_BLOCK> > { 
    
    typedef Node< NodeProperties<T,
				 numChannels,
				 1, 1,
				 THREADS_PER_BLOCK,
				 true,
				 THREADS_PER_BLOCK> > BaseType;
    
  public: 
    
    //
    // @brief constructor
    //
    // @param itailPtr -- pointer to a global tail pointer shared by
    // all blocks, used to collectively manage data allocations from
    // the source.
    //
    __device__
    Node_Source(size_t *itailPtr,
		Scheduler *scheduler)
      : BaseType(0, nullptr, scheduler),
	source(nullptr),
	tailPtr(itailPtr)
    {}
    
    //
    // @brief construct a Source object from the raw data passed down
    // to the device.
    //
    // @param sourceData source data passed from host to device
    // @return a Source object whose subtype matches the input data
    //
    __device__
    Source<T> *createSource(const SourceData<T> &sourceData,
			    SourceMemory<T> *mem)
    {
      Source<T> *source;
      
      switch (sourceData.kind)
	{
	case SourceData<T>::Buffer:
	  source = new (mem) SourceBuffer<T>(sourceData.bufferData,
					     tailPtr);
	  break;
	  
	case SourceData<T>::Range:
	  source = new (mem) SourceRange<T>(sourceData.rangeData,
					    tailPtr);
	  
	  break;
	}
      
      return source;
    }
    
    
    //
    // @brief prepare for the app's main kernel to run
    // Set our input source, then try to get an initial reservation
    // from the input source, so that the application has work to do.
    // If no work is available, set our tail state true to so indicate.
    //
    // Called single-threaded
    //
    // @param source input source  to use
    //
    __device__
    void setInputSource(Source<T> *isource)
    {
      source = isource;
    }
    
  private:

    using BaseType::getChannel;
    using BaseType::nDSActive;
    
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
    
    Source<T>* source;
    
    
    //
    // @brief fire the node, copying as much input as possible from
    // the source to the downstream queues.  This will cause at least
    // one downstream queue to activate OR will exhaust the source.
    //
    
    __device__
    virtual
    void fire()
    {
      int tid = threadIdx.x;

      TIMER_START(input);
      
      __shared__ unsigned int pendingOffset;
      __shared__ unsigned int numToWrite;
      
      if (IS_BOSS())
	{
	  // determine the amount of data needed to activate at least one
	  // downstream node by filling its queue.
	  
	  unsigned int numToRequest = UINT_MAX;
	  for (unsigned int c = 0; c < numChannels; c++)
	    numToRequest = min(numToRequest, getChannel(c)->dsCapacity());
	  
	  // round down to a full ensemble width, since the node with the
	  // least available space still has at least one ensemble's worth
	  // (given that it was inactive when fire() was called), and
	  // we can still get its free space to < one ensemble width.
	  numToRequest = (numToRequest / maxRunSize) * maxRunSize;
	  
	  // ask the source buffer for as many inputs as we want
	  numToWrite = source->reserve(numToRequest, &pendingOffset);
	}
      
      __syncthreads(); // all threads must see numToWrite and pendingOffset
      
      TIMER_STOP(input);
      TIMER_START(output);
      
      if (numToWrite > 0)
	{
	  __shared__ unsigned int dsBase[numChannels];
	  if (tid < numChannels)
	    dsBase[tid] = getChannel(tid)->directReserve(numToWrite);
	  
	  __syncthreads(); // all threads must see dsBase[] values
	  
	  // use every thread to copy from source to downstream queues
	  for (int base = 0; base < numToWrite; base += maxRunSize)
	    {
	      int srcIdx = base + tid;
	      T myData;
	      
	      if (srcIdx < numToWrite)
		{
		  myData = source->get(pendingOffset + srcIdx);
		  
		  for (unsigned int c = 0; c < numChannels; c++){
		    getChannel(c)->directWrite(myData, dsBase[c], srcIdx);
		  }
		}
	    }
	}
      
      TIMER_STOP(output);
      TIMER_START(input);
      
      if (IS_BOSS())
	{
	  if (numToWrite < numToRequest) // source is out of inputs
	    {
	      this->deactivate();
	      
	      for (unsigned int c = 0; c < numChannels; c++)
		{
		  getChannel(c)->getDSNode()->setFlushing();
		  getChannel(c)->getDSNode()->activate();
		}
	    }
	  else
	    {
	      nDSActive = 0;
	      
	      for (unsigned int c = 0; c < numChannels; c++)
		if (getChannel(c)->dsCapacity() < maxRunSize)
		  {
		    getChannel(c)->getDSNode()->activate();
		    nDSActive++;
		  }
	    }
	}
      
      TIMER_STOP(input);
    }
    
  private:
    
    size_t *tailPtr;
  };

}; // namespace Mercator

#endif
