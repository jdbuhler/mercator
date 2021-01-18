#ifndef __NODE_QUEUE_CUH
#define __NODE_QUEUE_CUH

//
// @file Node_Queue.cuh
// @brief a node that gets its input from a Queue
//
// MERCATOR
// Copyright (C) 2021 Washington University in St. Louis; all rights reserved.
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
  // @class Node_Queue
  // @brief most general typed node
  //
  // @tparam T type of input
  // @tparam numChannels  number of channels
  // @tparam UsesSignals true iff node ever needs to process signals
  // @tparam NodeFnKind type of node function that supplies doRun()
  //
  template <typename T, 
	    unsigned int numChannels,
	    bool UseSignals,
	   template<typename View> typename NodeFnKind>
  class Node_Queue : public NodeBaseWithChannels<numChannels> {
    
    using BaseType = NodeBaseWithChannels<numChannels>;
    using NodeFnType = NodeFnKind<Queue<T>>;
    
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
    Node_Queue(Scheduler *scheduler, 
	       unsigned int region,
	       NodeBase *usNode,
	       unsigned int usChannel,
	       unsigned int queueSize,
	       NodeFnType *inodeFunction)
      : BaseType(scheduler, region, usNode),
	queue(queueSize),
        signalQueue(UseSignals ? queueSize : 0), // could be smaller?
	nodeFunction(inodeFunction)
    {
      usNode->setDSEdge(usChannel, this, &queue, &signalQueue);
      nodeFunction->setNode(this);
    }

    __device__
    virtual ~Node_Queue()
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
      // # of items available to consume from data queue
      unsigned int nDataToConsume = queue.getOccupancy();
      
      // # of items already consumed from data queue
      unsigned int nDataConsumed = 0;
      
      
      // # of signals available to consume from signal queue
      unsigned int nSignalsToConsume = 0;

      // # of signals already consumed from signal queue
      unsigned int nSignalsConsumed = 0;
	  
      // # of credits before next signal, if one exists
      unsigned int nCredits = 0;
      
      if (UseSignals)
	{
	  // # of signals available to consume from signal queue
	  nSignalsToConsume = signalQueue.getOccupancy();
	  
	  // # of credits before next signal, if one exists
	  nCredits = (nSignalsToConsume == 0
		      ? 0
		      : signalQueue.getHead().credit);
	}
      
      // threshold for declaring data queue "empty" for scheduling
      unsigned int emptyThreshold = (this->isFlushing() 
				     ? 0
				     : NodeFnType::inputSizeHint - 1);
      
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
	  
	  unsigned int nFinished;
	  if (limit > 0)
	    {
	      // doRun() tries to consume input; could cause node to block
	      nFinished = nodeFunction->doRun(queue, nDataConsumed, limit);
	      
	      nDataConsumed += nFinished;
	    }
	  else
	    nFinished = 0;
	  
	  if (UseSignals)
	    {
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
	    }
	  
	  //
	  // Check whether any child needs to be activated
	  //
	  for (unsigned int c = 0; c < numChannels; c++)
	    {
	      ChannelBase *channel = getChannel(c);
	      if (channel->checkDSFull() || 
		  (UseSignals && channel->checkDSSigFull()))
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
      
      // BEGIN WRITE queue ptrs, credit, state changes in flushComplete()
      __syncthreads(); 
      
      if (IS_BOSS())
	{
	  // release any input we have consumed
	  queue.release(nDataConsumed);
	  
	  if (UseSignals)
	    {
	      signalQueue.release(nSignalsConsumed);
	      
	      // store any unused credits before next signal, if one exists
	      if (nSignalsToConsume > nSignalsConsumed)
		signalQueue.getHead().credit = nCredits;
	    }
	  
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
    }
    
  private:
    
    Queue<T> queue;                     // node's input queue
    Queue<Signal> signalQueue;          // node's input signal queue
    
    NodeFnType* const nodeFunction;
    
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
    // handleSignal() accordingly.
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
  };  // end Node_Queue class
}  // end Mercator namespace

#endif