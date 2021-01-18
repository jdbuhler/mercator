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
	   typename Source,
	   template<typename View> typename NodeFnKind>
  class Node_Source : public NodeBaseWithChannels<numChannels> {
    
    using BaseType = NodeBaseWithChannels<numChannels>;
    using NodeFnType = NodeFnKind<Source>;
    
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
    Node_Source(Scheduler *scheduler,
		unsigned int region,
		Source *isource,
		size_t *itailPtr,
		const size_t *inInputsPtr,
		NodeFnType *inodeFunction)
      : BaseType(scheduler, region, nullptr),
	source(isource),
	tailPtr(itailPtr),
	nInputsPtr(inInputsPtr),
	nodeFunction(inodeFunction),
	nDataPending(0),
	basePtr(0),
	sourceExhausted(false)
    {
      nodeFunction->setNode(this);
    }
    
    __device__
    virtual
    ~Node_Source()
    {
      delete nodeFunction;
      delete source;
    }    
    
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
      assert(minFreeSpace > 0);
      
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
    
    __device__
    void init() 
    { 
      if (IS_BOSS())
	{
	  source->init();
	  source->setup(*nInputsPtr, tailPtr);
	}
      
      nodeFunction->init(); 
    }
    
    __device__
    void cleanup() 
    { 
      nodeFunction->cleanup(); 
      
      if (IS_BOSS())
	source->cleanup();
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
      if (nDataPending == 0)
	{
	  // determine the amount of data needed to fill at least one
	  // downstream queue
	  
	  size_t numToRequest = UINT_MAX;
	  for (unsigned int c = 0; c < numChannels; c++)
	    numToRequest = min(numToRequest, 
			       (size_t) getChannel(c)->dsCapacity());
	  
	  // if the source advises a lower request size than what we planned,
	  // honor that.  Note that this may cause us to neither fill any
	  // output queue nor exhaust the input.
	  numToRequest = min(numToRequest, source->getRequestLimit());
	  
	  // BEGIN WRITE nDataPending, basePtr, sourceExhausted
	  __syncthreads();      
	  
	  if (IS_BOSS())
	    {
	      // ask the source buffer for as many inputs as we want
	      nDataPending = source->reserve(numToRequest, &basePtr);
	      if (nDataPending < numToRequest)
		sourceExhausted = true;
	    }
	  
	  // END WRITE nDataPending, basePtr, sourceExhausted
	  __syncthreads();
	}

      // # of items available to consume from queue
      unsigned int nDataToConsume = nDataPending;
      
      // # of items already consumed from queue
      unsigned int nDataConsumed = 0;
      
      // threshold for declaring data queue "empty" for scheduling
      // To actually use the input hint from the nodeFunction, we
      // need to be able to stop with nDataToConsume > 0 and ask for
      // another input block to get the # of elts over the input hint 
      // size.  This will require an InputView that can fuse two
      // source ranges.
      const unsigned int emptyThreshold = 0;
      
      bool dsActive = false;
      
      //
      // run until input queue satisfies EMPTY condition, or 
      // writing output causes some downstream neighbor to activate.
      //
      while (nDataToConsume - nDataConsumed > emptyThreshold && !dsActive)
	{
	  // determine the max # of items we may safely consume this time
	  unsigned int limit = nDataToConsume - nDataConsumed;
	  
	  unsigned int nFinished;
	  
	  // doRun() tries to consume input; could cause node to block
	  nFinished = nodeFunction->doRun(*source, basePtr + nDataConsumed, 
					  limit);
	  
	  nDataConsumed += nFinished;
	  
	  //
	  // Check whether any child needs to be activated
	  //
	  for (unsigned int c = 0; c < numChannels; c++)
	    {
	      if (getChannel(c)->checkDSFull())
		{
		  dsActive = true;
		  
		  if (IS_BOSS())
		    getDSNode(c)->activate();
		}
	    }
	  
	  // don't keep trying to run the node if it is blocked
	  if (this->isBlocked())
	    break;
	}
      
      // BEGIN WRITE nDataPending, state changes in flushComplete()
      __syncthreads(); 
      
      if (IS_BOSS())
	{
	  nDataPending -= nDataConsumed;
	  basePtr      += nDataConsumed;
	 
	  if (nDataToConsume - nDataConsumed <= emptyThreshold)
	    {
	      if (sourceExhausted)
		{
		  this->deactivate();
		  
		  // no more inputs to read -- force downstream nodes
		  // into flushing mode and activate them (if not
		  // already active).  Even if they have no input,
		  // they must fire once to propagate flush mode to
		  // *their* downstream nodes.
		  for (unsigned int c = 0; c < numChannels; c++)
		    {
		      NodeBase *dsNode = getDSNode(c);
		      
		      // 0 = global region ID
		      if (this->initiateFlush(dsNode, 0)) 
			dsNode->activate();
		    }
		  
		  nodeFunction->flushComplete();
		}
	      else if (!dsActive && !this->isBlocked())
		{
		  // If we did not activate a downstream node or exhaust
		  // the input stream, we need to forcibly re-enqueue
		  // ourselves and fire again, since we remain fireable
		  // and our descendants might not be.
		  this->forceReschedule();
		}
	    }
	}
      
      // END WRITE nDataPending, state changes in flushComplete()
      // [suppressed because we are assumed to sync before next firing]
      // __syncthreads(); 
    }
    
  private:

    Source* const source;
    
    size_t* const tailPtr;
    const size_t *const nInputsPtr;
    NodeFnType* const nodeFunction;
    
    size_t nDataPending;
    size_t basePtr;
    bool sourceExhausted;
  };

}; // namespace Mercator

#endif
