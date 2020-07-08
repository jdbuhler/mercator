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
		Scheduler *scheduler,
		unsigned int region)
      : BaseType(0, scheduler, region),
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

    using BaseType::maxRunSize;
    using BaseType::getChannel;
    
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
    void fire()
    {
      // type of our downstream channels matchses our input type,
      // since the source module just copies its inputs downstream
      using Channel = typename BaseType::Channel<T>;
      
      int tid = threadIdx.x;
      
      TIMER_START(input);
      
      // determine the amount of data needed to activate at least one
      // downstream node by filling its queue.
      
      size_t numToRequest = UINT_MAX;
      for (unsigned int c = 0; c < numChannels; c++)
	numToRequest = min(numToRequest, (size_t) getChannel(c)->dsCapacity());
      
      // round down to a full ensemble width, since the node with the
      // least available space still has at least one ensemble's worth
      // (given that it was inactive when fire() was called), and
      // we can still get its free space to < one ensemble width.
      numToRequest = (numToRequest / maxRunSize) * maxRunSize;
      
      // if the source advises a lower request size than what we planned,
      // honor that.  Note that this may cause us to neither fill any
      // output queue nor exhaust the input.
      numToRequest = min(numToRequest, source->getRequestLimit());
      
      __shared__ size_t pendingOffset;
      __shared__ size_t numToWrite;
      
      if (IS_BOSS())
	{	  
	  // ask the source buffer for as many inputs as we want
	  numToWrite = source->reserve(numToRequest, &pendingOffset);
	  COUNT_ITEMS(numToWrite);
	}
      
      __syncthreads(); // all threads must see shared vars
      
      TIMER_STOP(input);

#if 0
	  if (IS_BOSS())
	    printf("%d %p SRC\n", blockIdx.x, this);
#endif
      
      TIMER_START(output);
      
      if (numToWrite > 0)
	{
	  __shared__ unsigned int dsBase[numChannels];
	  if (tid < numChannels)
	    {
	      const Channel *channel = 
		static_cast<Channel *>(getChannel(tid));
	      
	      dsBase[tid] = channel->directReserve(numToWrite);
	    }
	  
	  __syncthreads(); // all threads must see dsBase[] values
	  
	  // use every thread to copy from source to downstream queues
	  for (int base = 0; base < numToWrite; base += maxRunSize)
	    {
	      int srcIdx = base + tid;
	      
	      if (srcIdx < numToWrite)
		{
		  T myData = source->get(pendingOffset + srcIdx);
		  
		  for (unsigned int c = 0; c < numChannels; c++)
		    {
		      const Channel *channel = 
			static_cast<Channel *>(getChannel(c));
		      
		      channel->directWrite(myData, dsBase[c], srcIdx);
		    }
		}
	    }
	}
      
      TIMER_STOP(output);
      
      TIMER_START(input);
      
      if (IS_BOSS())
	{
	  if (numToWrite < numToRequest)
	    {
	      // no inputs remaining in source
	      this->deactivate();
	      
	      // no more inputs to read -- force downstream nodes
	      // into flushing mode and activate them (if not
	      // already active).  Even if they have no input,
	      // they must fire once to propagate flush mode and
	      // activate *their* downstream nodes.
	      for (unsigned int c = 0; c < numChannels; c++)
		{
		  NodeBase *dsNode = getChannel(c)->getDSNode();
		  this->initiateFlush(dsNode);
		  dsNode->activate();
		}
              
	      this->clearFlush(); // disable flushing
	    }
	  else
	    {
	      bool anyChildActive = false;
	      
	      // activate any downstream nodes whose queues are now full
	      for (unsigned int c = 0; c < numChannels; c++)
		anyChildActive = getChannel(c)->checkDSFull();
	      
	      // If we did not fill any downstream queues or exhaust
	      // the input stream, we need to forcibly re-enqueue
	      // ourselves and fire again.  This can happen only if
	      // the source artificially limited our input request
	      // size.
	      if (!anyChildActive)
		{
		  this->forceReschedule();
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
