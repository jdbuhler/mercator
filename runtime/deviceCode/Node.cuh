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
    Node(const unsigned int queueSize,
	 Scheduler *scheduler, 
	 unsigned int region,
	 RefCountedArena *iparentArena)
      : BaseType(scheduler, region),
	queue(queueSize),
        signalQueue(queueSize), // could be smaller?
	parentArena(iparentArena),
	parentIdx(RefCountedArena::NONE)
    {}
    
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
    
  public:
    
    //
    // @brief return our queue (needed for upstream channel's
    // setDSEdge()).
    //
    __device__
    QueueBase *getQueue()
    { return &queue; }
    
    //
    // @brief return our signal queue (needed for upstream channel's
    // setDSEdge()).
    //
    __device__
    Queue<Signal> *getSignalQueue()
    { return &signalQueue; }
    
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
				     : getMaxInputs() - 1);
      
      bool dsActive = false;
      
      //
      // run until input queue satisfies EMPTY condition, or 
      // writing output causes some downstream neighbor to activate.
      //
      while ((nDataToConsume - nDataConsumed > emptyThreshold || 
	      nSignalsConsumed < nSignalsToConsume) &&
	     !dsActive)
	{
#if 0
	  if (IS_BOSS())
	    printf("%d %p %d %d %d %d %d\n", 
		   blockIdx.x, this, 
		   nDataConsumed, nDataToConsume,  
		   nSignalsConsumed, nSignalsToConsume,
		   nCredits);
#endif
	  
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
      
      // protect code above from queue changes below
      __syncthreads();

      if (IS_BOSS())
	{
	  // release any input we have consumed
	  queue.release(nDataConsumed);
	  signalQueue.release(nSignalsConsumed);
	  
	  // store any unused credits before next signal
	  if (!signalQueue.empty())
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
		
		flushComplete();
		this->clearFlush();  // disable flushing
	      }
	  }
	}
      
      TIMER_STOP(input);
    }

  protected:
    
    // state for nodes in enumerated regions
    RefCountedArena* const parentArena;   // ptr to any associated parent buffer
    unsigned int parentIdx;               // index of parent obj in buffer
    
  private:
    
    Queue<T> queue;                     // node's input queue
    Queue<Signal> signalQueue;          // node's input signal queue
    
    // begin and end stubs for enumeration and aggregation 
    __device__
    virtual
    void begin() {}

    __device__
    virtual
    void end() {}
    
    //
    // @brief get the maximum number of inputs that will ever
    // be consumed by one call to doRun()
    //
    __device__
    virtual
    unsigned int getMaxInputs() const = 0;
    //
    
    //
    // @brief function stub to execute the function code specific
    // to this node.  This function does NOT remove data from the
    // queue.
    //
    // @param queue data queue containing items to be consumed
    // @param start index of first item in queue to consume
    // @param limit max number of items that this call may consume
    // @return number of items ACTUALLY consumed (may be 0).
    //
    __device__
    virtual
    unsigned int doRun(const Queue<T> &queue,
		       unsigned int start,
		       unsigned int limit) = 0;
    
    
    //
    // @brief callback from fire() when node empties its queues
    // after completing a flush operation.
    //
    __device__
    virtual
    void flushComplete()
    {}
    
  private:
    
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
      
      const Signal &s = signalQueue.getElt(sigIdx);
      
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
	      ? signalQueue.getElt(sigIdx + 1).credit
	      : 0);
    }
    
    //
    // @brief handle signal of type Enum
    // MUST BE CALLED WITH ALL THREADS
    //
    // @param s signal to handle
    //
    __device__
    virtual
    void handleEnum(const Signal &s)
    {
      if (parentIdx != RefCountedArena::NONE) // is old parent valid?
	end();
      
      __syncthreads(); // protect parent above against unref below
      
      if (IS_BOSS())
	{
	  parentArena->unref(parentIdx); // remove this node's reference
	  
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
      
      __syncthreads(); // for parentIdx
      
      if (parentIdx != RefCountedArena::NONE) // is new parent valid?
	begin();
    }
  };  // end Node class
}  // end Mercator namespace

#endif
