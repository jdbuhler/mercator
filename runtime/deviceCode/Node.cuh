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
  // @tparam numEltsPerGroup number of input elements/thread
  // @tparam threadGroupSize  number of threads in a thread group
  // @tparam maxActiveThreads max # of live threads in any call to run()
  // @tparam runWithAllThreads call run() with all threads, or just as many
  //           as have inputs?
  //
  template <typename T, 
	    unsigned int numChannels,
	    unsigned int THREADS_PER_BLOCK>
  class Node : public NodeBaseWithChannels<numChannels> {
    
    using BaseType = NodeBaseWithChannels<numChannels>;
    
  protected:
    
    using BaseType::getChannel;
    using BaseType::getDSNode;

  public:

    //
    // @brief Constructor
    //
    // @param maxActiveThreads max allowed # of active threads per run
    // @param queueSize requested size for node's input queue
    //
    __device__
    Node(const unsigned int queueSize,
	 Scheduler *scheduler, unsigned int region)
      : BaseType(scheduler, region),
	queue(queueSize),
        signalQueue(queueSize), // could be smaller?
	parentArena(nullptr)
    {}
    
    __device__
    void setParentArena(RefCountedArena *a)
    {
      parentArena = a;
    }
    
    //
    // @brief return our queue (needed for channel's setDSEdge()).
    //
    __device__
    Queue<T> *getQueue()
    { 
      return &queue; 
    }
    
    //
    // @brief return our signal queue (needed for channel's setDSEdge()).
    //
    __device__
    Queue<Signal> *getSignalQueue()
    { 
      return &signalQueue; 
    }
    
    //
    // @brief is any input queued for this node?
    // (Only used for debugging.)
    //
    __device__
    bool hasPending() const
    {
      return (!queue.empty() || !signalQueue.empty());
    }
    
    __device__
    virtual
    unsigned int doRun(const Queue<T> &queue,
		       unsigned int start,
		       unsigned int limit) = 0;
    
    __device__
    virtual
    unsigned int getMaxInputs() const = 0;
    
    __device__
    void fire()
    {
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
				     : getMaxInputs() - 1);
      
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
	  
	  // determine the max # of items we may safely consume 
	  unsigned int limit =
	    (nSignalsConsumed < nSignalsToConsume
	     ? nCredits 
	     : nDataToConsume - nDataConsumed);
	  
	  TIMER_STOP(input);
	  
	  TIMER_START(run);
	  
	  unsigned int nFinished;
	  if (limit > 0)
	    {
	      nFinished = doRun(queue, nDataConsumed, limit);
	      
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
	      
	      __syncthreads(); // protect channel # of items written
	      
	      if (nCredits == 0 && !this->isBlocked())
		{
		  nCredits = this->handleSignal(nSignalsConsumed);
		  nSignalsConsumed++;
		}
	    }
	  
	  TIMER_STOP(run);
	  
	  TIMER_START(output);
	  
	  __syncthreads(); // protect channel changes
	  
	  //
	  // Check whether any child needs to be activated
	  //
	  for (unsigned int c = 0; c < numChannels; c++)
	    {
	      if (getChannel(c)->checkDSFull())
		{
		  anyDSActive = true;
		  
		  if (IS_BOSS())
		    getDSNode(c)->activate();
		}
	    }
	  
	  TIMER_STOP(output);
	  
	  if (this->isBlocked())
	    break;
	  
	  TIMER_START(input);
	}
      
      // protect code above from queue changes below
      __syncthreads();

      if (IS_BOSS())
	{
	  queue.release(nDataConsumed);
	  signalQueue.release(nSignalsConsumed);
	  
	  if (!signalQueue.empty())
	    signalQueue.getHead().credit = nCredits;
	  
	  if (nDataToConsume - nDataConsumed <= emptyThreshold &&
	      nSignalsConsumed == nSignalsToConsume)
	  {
	    // less than a full ensemble remains, or 0 if flushing
	    this->deactivate(); 

	    if (this->isFlushing())
	      {
		// no more inputs to read -- force downstream nodes
		// into flushing mode and activate them (if not
		// already active).  Even if they have no input,
		// they must fire once to propagate flush mode and
		// activate *their* downstream nodes.
		for (unsigned int c = 0; c < numChannels; c++)
		  {
		    NodeBase *dsNode = getDSNode(c);
		    
		    if (this->propagateFlush(dsNode))
		      dsNode->activate();
		  }
		
		this->clearFlush();  // disable flushing
	      }
	  }
	}
      
      TIMER_STOP(input);
    }
    
    // begin and end stubs for enumeration and aggregation 
    __device__
    virtual
    void begin() {}

    __device__
    virtual
    void end() {}
    
  protected:
    
    Queue<T> queue;                     // node's input queue
    Queue<Signal> signalQueue;          // node's input signal queue
    
    // state for nodes in enumerated regions
    RefCountedArena *parentArena;       // ptr to any associated parent buffer
    unsigned int parentIdx;             // index of parent obj in buffer

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
      assert(outputsPerInput > 0);
      
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
    // Return the credit associated with the signal at index sigIdx+1
    // in the queue, if any exists; otherwise, return 0.
    //
    __device__
    unsigned int handleSignal(unsigned int sigIdx)
    {
      const Queue<Signal> &signalQueue = this->signalQueue;
      
      /////////////////////////////
      // SIGNAL HANDLING SWITCH
      /////////////////////////////
      
      const Signal &s = signalQueue.getElt(sigIdx);
      
      switch (s.tag)
	{
	case Signal::Enum:
	  handleEnum(s);
	  break;
	  
	case Signal::Agg:
	  handleAgg(s);
	  break;
	  
	default:
	  if (IS_BOSS())
	    printf("ERROR: unhandled signal type %d detected\n", s.tag);
	  assert(false);
	}
      
      // return credit from next signal if there is one
      return (sigIdx + 1 < signalQueue.getOccupancy()
	      ? signalQueue.getElt(sigIdx + 1).credit
	      : 0);
    }
    
  protected:    

    __device__
    virtual
    void handleEnum(const Signal &s)
    {
      if (IS_BOSS())
	{
	  // set the parent object for this node (specified
	  // as an index into its parent arena)
	  
	  parentIdx = s.parentIdx;
	  parentArena->ref(parentIdx);
	  
	  //Reserve space downstream for the new signal
	  for (unsigned int c = 0; c < numChannels; ++c)
	    {
	      ChannelBase *channel = getChannel(c);
	      
	      // propagate the signal unless we are at region frontier
	      if (!channel->isAggregate())
		{
		  // add reference for newly created signal
		  parentArena->ref(parentIdx); 
		  channel->pushSignal(s);
		}
	    }
	  
	  parentArena->unref(parentIdx); // signal is destroyed
	}
      
      //Call the begin stub of this node
      this->begin();
    }
    
    
    __device__
    virtual
    void handleAgg(const Signal &s)
    {
      //Call the end stub of this node
      this->end();
      
      if (IS_BOSS())
	{
	  //Reserve space downstream for the new signal
	  for (unsigned int c = 0; c < numChannels; ++c)
	    {
	      ChannelBase *channel = getChannel(c);
	      
	      // if we're not at a region frontier, propagate
	      // the signal; if we are, we can remove our reference
	      // to the parent object.
	      if(!channel->isAggregate())
		channel->pushSignal(s);	    
	    }
	  
	  parentArena->unref(parentIdx); // remove this node's reference
	}
    }
    
  };  // end Node class
}  // end Mercator namespace

#endif
