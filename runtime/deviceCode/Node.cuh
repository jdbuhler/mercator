#ifndef __NODE_CUH
#define __NODE_CUH

//
// @file Node.cuh
// @brief a MERCATOR node that knows its input type
//
// MERCATOR
// Copyright (C) 2020 Washington University in St. Louis; all rights reserved.
//

#include <cstdio>
#include <cassert>

#include "NodeBaseWithChannels.cuh"

#include "ChannelBase.cuh"

#include "Queue.cuh"
#include "Signal.cuh"

#include "ParentBuffer.cuh"

#include "device_config.cuh"

#include "options.cuh"

#include "timing_options.cuh"

namespace Mercator  {

  //
  // @class Node
  // @brief most general typed node
  //
  // @tparam T type of input
  // @tparam numChannels  number of channels
  // @tparam DerivedNodeType subtype of node for CRTP-based call to doRun()
  //
  template <typename T, 
	    unsigned int numChannels,
	    unsigned int THREADS_PER_BLOCK, // FIXME: needed?
	    typename NodeFcnType>
  class Node : public NodeBaseWithChannels<numChannels> {
    
    using BaseType = NodeBaseWithChannels<numChannels>;

    using BaseType::getChannel;
    using BaseType::getDSNode;
    
  public:
    
    ///////////////////////////////////////////////////////
    // INIT/CLEANUP KERNEL FUNCIIONS
    ///////////////////////////////////////////////////////

    //
    // @brief Constructor
    //
    // @param maxActiveThreads max allowed # of active threads per run
    // @param queueSize requested size for node's input queue
    //
    __device__
    Node(Scheduler *scheduler, 
	 unsigned int region,
	 NodeBase *usNode,
	 unsigned int usChannel,
	 unsigned int queueSize,
	 NodeFcnType *inodeFunction)
      : BaseType(scheduler, region, usNode),
	queue(queueSize),
        signalQueue(queueSize), // could be smaller?
	nodeFunction(inodeFunction)
    {
      usNode->setDSEdge(usChannel, this, &queue, &signalQueue);
      nodeFunction->setNode(this);
    }

    __device__
    ~Node()
    {
      delete nodeFunction;
    }
    
    __device__
    void init() { nodeFunction->init(); }
    
    __device__
    void cleanup() { nodeFunction->cleanup(); }
    
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
		     unsigned int minFreeSpace,
		     bool isAgg = false)
    {
      assert(c < numChannels);
      assert(minFreeSpace > 0);
      
      // init the output channel -- should only happen once!
      assert(getChannel(c) == nullptr);
      
      setChannel(c, new Channel<DST>(minFreeSpace, isAgg));
      
      // make sure alloc succeeded
      if (getChannel(c) == nullptr)
	{
	  printf("ERROR: failed to allocate channel object [block %d]\n",
		 blockIdx.x);

	  crash();
	}
    }

    /////////////////////////////////////////////////////////
    
    //
    // @brief is any input queued for this node?
    // (Only used for debugging.)
    //
    __device__
    bool hasPending() const
    {
      return (!queue.empty() || !signalQueue.empty());
    }

    
    //
    // @brief main firing loop. Consume data and signals according to
    // the synchronization dictated by the scheduling protocol.
    // Implement the checks needed to determine when the input queue
    // empties or an output queue fills, and make the necessary
    // scheduling status changes for this node and its neighbors when
    // one of those things happens.
    //
    // MUST BE CALLED WITH ALL THREADS
    //
    __device__
    void fire()
    {
      TIMER_START(input);
      
      Queue<T> &queue = this->queue;
      Queue<Signal> &signalQueue = this->signalQueue; 
      
      // # of items available to consume from queue
      unsigned int nDataToConsume = queue.getOccupancy();
      unsigned int nSignalsToConsume = signalQueue.getOccupancy();
      
      // # of credits before next signal, if one exists
      unsigned int nCredits = (nSignalsToConsume == 0
			       ? 0
			       : signalQueue.getHead().credit);
      
      // # of items already consumed from queue
      unsigned int nDataConsumed = 0;
      unsigned int nSignalsConsumed = 0;
      
      // threshold for declaring data queue "empty" for scheduling
      unsigned int emptyThreshold = (this->isFlushing() 
				     ? 0
				     : NodeFcnType::inputSizeHint - 1);
      
      bool dsActive = false;
      
      //
      // run until input queue satisfies EMPTY condition, or 
      // writing output causes some downstream neighbor to activate.
      //
      while ((nDataToConsume - nDataConsumed > emptyThreshold || 
	      nSignalsConsumed < nSignalsToConsume) &&
	     !dsActive)
	{
	  // determine the max # of items we may safely consume this time
	  unsigned int limit =
	    (nSignalsConsumed < nSignalsToConsume
	     ? nCredits 
	     : nDataToConsume - nDataConsumed);
	  
	  TIMER_STOP(input);
	  
	  TIMER_START(run);
	  
	  unsigned int nFinished;
	  if (limit > 0)
	    {
	      // doRun() tries to consume input; could cause node to block
	      nFinished = nodeFunction->doRun(queue, nDataConsumed, limit);
	      
	      nDataConsumed += nFinished;
	    }
	  else
	    nFinished = 0;
	  
	  //
	  // Track credit to next signal, and consume if needed.
	  //
	  if (nSignalsConsumed < nSignalsToConsume)
	    {
	      nCredits -= nFinished;
	      
	      if (nCredits == 0 && !this->isBlocked())
		{
		  nCredits = this->handleSignal(nSignalsConsumed);
		  nSignalsConsumed++;
		}
	    }
	  
	  TIMER_STOP(run);
	  
	  TIMER_START(output);
	  
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
	  
	  TIMER_STOP(output);
	  
	  // don't keep trying to run the node if it is blocked
	  if (this->isBlocked())
	    break;
	  
	  TIMER_START(input);
	}
      
      // BEGIN WRITE queue ptrs, credit, state changes in flushComplete()
      __syncthreads(); 
      
      if (IS_BOSS())
	{
	  // release any input we have consumed
	  queue.release(nDataConsumed);
	  signalQueue.release(nSignalsConsumed);
	  
	  // store any unused credits before next signal, if one exists
	  if (nSignalsToConsume > nSignalsConsumed)
	    signalQueue.getHead().credit = nCredits;
	  
	  // did we empty our input queue?
	  if (nDataToConsume - nDataConsumed <= emptyThreshold &&
	      nSignalsConsumed == nSignalsToConsume)
	    {
	      this->deactivate(); 
	      
	      if (this->isFlushing())
	      {
		// force downstream neighbors into flushing mode and
		// activate them (if not already active).  Even if
		// they have no input, they must fire once to
		// propagate the flush and activate *their* downstream
		// neighbors.
		for (unsigned int c = 0; c < numChannels; c++)
		  {
		    NodeBase *dsNode = getDSNode(c);
		    
		    if (this->propagateFlush(dsNode))
		      dsNode->activate();
		  }
		
		nodeFunction->flushComplete();
		this->clearFlush();  // disable flushing
	      }
	    }
	}
      
      // END WRITE queue ptrs, credit, state changes in flushComplete()
      // [suppressed because we are assumed to sync before next firing]
      // __syncthreads(); 
      
      TIMER_STOP(input);
    }
    
  private:

    Queue<T> queue;                     // node's input queue
    Queue<Signal> signalQueue;          // node's input signal queue
    
    NodeFcnType* const nodeFunction;
    
#ifdef INSTRUMENT_TIME
    using BaseType::inputTimer;
    using BaseType::runTimer;
    using BaseType::outputTimer;
#endif

    ////////////////////////////////////////////////////////////////////
    // SIGNAL HANDLING LOGIC
    //
    // As new signal types Foo are added in Signal.h, we need to add a
    // handler here named handleFoo() and update the switch stmt in
    // handleSignal() accordingly.  Handlers are always VIRTUAL so
    // that subclasses may override them; if a generic Node needs no
    // handler for a signal, it should be given an empty function
    // here, and the subclass should provide a new version.
    ///////////////////////////////////////////////////////////////////

    // 
    // @brief Signal handling dispatch for a node.  Consume the
    // sigIdx-th signal in the queue and perform whatever action it
    // demands (which may generate additional downstream signals).
    //
    // MUST BE CALLED WITH ALL THREADS
    //
    // @param index of signal to be handled in signal queue 
    // @return credit associated with the signal at index sigIdx+1
    // in the queue, if any exists; otherwise, 0.
    //
    __device__
    unsigned int handleSignal(unsigned int sigIdx)
    {
      const Queue<Signal> &signalQueue = this->signalQueue;
      
      /////////////////////////////
      // SIGNAL HANDLING SWITCH
      /////////////////////////////
      
      const Signal &s = signalQueue.get(sigIdx);
      
      switch (s.tag)
	{
	case Signal::Enum:
	  handleEnum(s);
	  break;
	  
	default:
	  if (IS_BOSS())
	    printf("ERROR: unhandled signal type %d detected\n", s.tag);
	  assert(false);
	}
      
      // return credit from next signal if there is one
      return (sigIdx + 1 < signalQueue.getOccupancy()
	      ? signalQueue.get(sigIdx + 1).credit
	      : 0);
    }
    
    //
    // @brief handle signal of type Enum
    // MUST BE CALLED WITH ALL THREADS
    //
    // @param s signal to handle
    //
    __device__
    void handleEnum(const Signal &s)
    {
      unsigned int pIdx = nodeFunction->getParentIdx();
      
      // is old parent valid?
      if (pIdx != RefCountedArena::NONE)
	nodeFunction->end();
      
      __syncthreads(); // BEGIN WRITE parentIdx, ds signal queue
      
      if (IS_BOSS())
	{
	  RefCountedArena *parentArena = nodeFunction->getParentArena();
	  
	  parentArena->unref(pIdx); // remove this node's reference
	  
	  // set the parent object for this node (specified
	  // as an index into its parent arena)

	  pIdx = s.parentIdx;
	  nodeFunction->setParentIdx(pIdx);
	  
	  parentArena->ref(pIdx);
	  
	  //Reserve space downstream for the new signal
	  for (unsigned int c = 0; c < numChannels; ++c)
	    {
	      ChannelBase *channel = getChannel(c);
	      
	      // propagate the signal unless we are at region frontier
	      if (!channel->isAggregate())
		{
		  // add reference for newly created signal
		  parentArena->ref(pIdx); 
		  channel->pushSignal(s);
		}
	    }
	  
	  parentArena->unref(pIdx); // signal is destroyed
	}
      
      __syncthreads(); // END WRITE parentIdx, ds signal queue
      
      if (pIdx != RefCountedArena::NONE) // is new parent valid?
	nodeFunction->begin();
    }
  };  // end Node class
}  // end Mercator namespace

#endif
