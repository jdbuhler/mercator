#ifndef __NODE_SOURCE_CUH
#define __NODE_SOURCE_CUH

//
// @file Node_Source.cuh
// @brief a node that gets its input from an external source
//
// MERCATOR
// Copyright (C) 2021 Washington University in St. Louis; all rights reserved.
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
  // @tparam Source type of source object from which we read inputs
  // @tparam NodeFnKind type of node function that supplies doRun()
  //
  template<typename T, 
	   unsigned int numChannels,
	   typename Source,
	   template<typename View> typename NodeFnKind>
  class Node_Source final : public NodeBaseWithChannels<numChannels> {
    
    using BaseType = NodeBaseWithChannels<numChannels>;
    using NodeFnType = NodeFnKind<Source>;
    
    using BaseType::getChannel;
    using BaseType::getDSNode;
    
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
		NodeFnType *inodeFunction)
      : BaseType(scheduler, region, nullptr),
	source(isource),
	nodeFunction(inodeFunction),
	nDataPending(0),
	basePtr(0)
    {
      nodeFunction->setNode(this);
    }
    
    __device__
    virtual ~Node_Source()
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
	  source->setup();
	  
	  sourceExhausted = false; // indicate that we have data to consume

	  //
	  // Figure out how many items to request each time we
	  // ask for a chunk of input from the source.
	  //
	  
	  // don't ask for more than the source suggests
	  numToRequest = source->getRequestLimit();
	  
	  // size the request proportional to the size of our smallest
	  // downstream queue, assuming that the queue will be nearly
	  // empty when the source node fires.
	  for (unsigned int c = 0; c < numChannels; c++)
	    {
	      numToRequest = min(numToRequest,
				 (size_t) getChannel(c)->dsSize());
	    }
	  
	  // Round the request size down to a multiple of the node function's
	  // preferred input width (but don't make it 0!).
	  const unsigned int vecsize = NodeFnType::inputSizeHint;
	  if (numToRequest >= vecsize)
	    numToRequest = (numToRequest / vecsize) * vecsize;
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
      return (nDataPending > 0);
    }
    
    //
    // @brief fire the node, processing much input from the source as
    // possible. This will cause at least one downstream queue to
    // activate OR will exhaust the source.
    //
    // MUST BE CALLED WITH ALL THREADS
    //
    __device__
    void fire()
    {
      TIMER_START(overhead);
      
      bool dsActive = false;
      
      do 
	{
	  if (nDataPending == 0)
	    {
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
	
	  //
	  // run until input queue satisfies EMPTY condition, or 
	  // writing output causes some downstream neighbor to activate.
	  //
	  while (nDataToConsume - nDataConsumed > emptyThreshold && !dsActive)
	    {
	      // determine the max # of items we may safely consume this time
	      unsigned int limit = nDataToConsume - nDataConsumed;
	      
	      TIMER_STOP(overhead);
	      TIMER_START(user);
	      
	      // doRun() tries to consume input; could cause node to block
	      unsigned int nFinished = 
		nodeFunction->doRun(*source, 
				    basePtr + nDataConsumed, 
				    limit);
	      
	      TIMER_STOP(user);
	      TIMER_START(overhead);
	      
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
	  
	  // BEGIN WRITE nDataPending, basePtr, state changes
	  // in flushComplete()
	  __syncthreads(); 
	  
	  if (IS_BOSS())
	    {
	      nDataPending -= nDataConsumed;
	      basePtr      += nDataConsumed;
	      
	      if (sourceExhausted && nDataPending == 0)
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
	    }
	  
	  // END WRITE nDataPending, basePtr, state changes in flushComplete()
	  __syncthreads();
	
	  // keep going until we are blocked, either by full output or
	  // by actual blockage, or we have nothing left to do.
	} 
      while (!dsActive && !this->isBlocked() &&
	     !(sourceExhausted && nDataPending == 0));
      
      TIMER_STOP(overhead);
    }
    
  private:

    Source* const source;
    NodeFnType* const nodeFunction;
    size_t numToRequest;
    
    size_t nDataPending;
    size_t basePtr;
    bool sourceExhausted;
  };

}; // namespace Mercator

#endif
