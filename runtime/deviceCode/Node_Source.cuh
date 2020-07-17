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

#include "NodeBaseWithChannels.cuh"

#include "Channel.cuh"

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
    public NodeBaseWithChannels<numChannels> {
    
    using BaseType = NodeBaseWithChannels<numChannels>;
    
  private:
    
    using BaseType::getChannel;
    using BaseType::getDSNode;
    
#ifdef INSTRUMENT_TIME
    using BaseType::inputTimer;
    using BaseType::runTimer;
    using BaseType::outputTimer;
#endif

#ifdef INSTRUMENT_OCC
    using BaseType::occCounter;
#endif
    
  public: 

    ///////////////////////////////////////////////////////
    // INIT/CLEANUP KERNEL FUNCIIONS
    ///////////////////////////////////////////////////////
    
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
      : BaseType(scheduler, region),
	source(nullptr),
	tailPtr(itailPtr)
    {
#ifdef INSTRUMENT_OCC
      occCounter.setMaxRunSize(THREADS_PER_BLOCK);
#endif
    }
    
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

  protected:

    //
    // @brief Create and initialize an output channel.
    //
    // @param c index of channel to initialize
    // @param minFreeSpace minimum free space before channel's
    // downstream queue is considered full
    //
    template<typename DST>
    __device__
    void initChannel(unsigned int c, 
		     unsigned int minFreeSpace)
    {
      assert(c < numChannels);
      assert(outputsPerInput > 0);
      
      // init the output channel -- should only happen once!
      assert(getChannel(c) == nullptr);
      
      setChannel(c, new Channel<DST>(minFreeSpace, false));
      
      // make sure alloc succeeded
      if (getChannel(c) == nullptr)
	{
	  printf("ERROR: failed to allocate channel object [block %d]\n",
		 blockIdx.x);
	  
	  crash();
	}
    }
    
    /////////////////////////////////////////////////////////////////
    
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
      assert(IS_BOSS());
      source = isource;
    }
    
    //
    // @brief is any input queued for this node?
    // (Only used for debugging.)
    //
    __device__
    bool hasPending() const
    {
      return false; // cannot get this info from source
    }

    //
    // @brief fire the node, copying as much input as possible from
    // the source to the downstream queues.  This will cause at least
    // one downstream queue to activate OR will exhaust the source.
    //
    // MUST BE CALLED WITH ALL THREADS
    //
    __device__
    void fire()
    {
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
      numToRequest = (numToRequest / THREADS_PER_BLOCK) * THREADS_PER_BLOCK;
      
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
	}
      __syncthreads(); // all threads must see shared vars
      
      TIMER_STOP(input);
      
      TIMER_START(output);
      
      for (size_t base = 0; base < numToWrite; base += THREADS_PER_BLOCK)
	{
	  unsigned int vecSize = 
	    min(numToWrite - base, (size_t) THREADS_PER_BLOCK);
	  
	  NODE_OCC_COUNT(vecSize);
	  
	  size_t srcIdx = base + tid;
	  
	  T myData = (srcIdx < numToWrite 
		      ? source->get(pendingOffset + srcIdx)
		      : source->get(pendingOffset)); // dummy
	  
	  for (unsigned int c = 0; c < numChannels; c++)
	    {
	      using Channel = Channel<T>;
	      
	      Channel *channel = static_cast<Channel*>(getChannel(c));
	      
	      channel->pushCount(myData, vecSize);
	    }
	}
      
      TIMER_STOP(output);
      
      TIMER_START(input);
      
      if (IS_BOSS())
	{
	  if (numToWrite < numToRequest)
	    {
	      // no inputs remain in source
	      this->deactivate();
	      
	      // no more inputs to read -- force downstream nodes
	      // into flushing mode and activate them (if not
	      // already active).  Even if they have no input,
	      // they must fire once to propagate flush mode to
	      // *their* downstream nodes.
	      for (unsigned int c = 0; c < numChannels; c++)
		{
		  NodeBase *dsNode = getDSNode(c);
		  
		  if (this->initiateFlush(dsNode, 0)) // 0 = global region ID
		    dsNode->activate();
		}
	    }
	  else
	    {
	      bool dsActive = false;
	      
	      //
	      // Check whether any child needs to be activated
	      //
	      for (unsigned int c = 0; c < numChannels; c++)
		{
		  if (getChannel(c)->checkDSFull())
		    {
		      dsActive = true;
		      getDSNode(c)->activate();
		    }
		}
	      
	      // If we did not fill any downstream queues or exhaust
	      // the input stream, we need to forcibly re-enqueue
	      // ourselves and fire again.  This can happen only if
	      // the source artificially limited our input request
	      // size.
	      if (!dsActive)
		this->forceReschedule();
	    }
	}
      
      TIMER_STOP(input);
    }
    
  private:
    
    size_t* const tailPtr;
    
    Source<T>* source;
  };
  
}; // namespace Mercator

#endif
