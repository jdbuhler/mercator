#ifndef __NODE_SINGLEITEM_CUH
#define __NODE_SINGLEITEM_CUH

//
// @file Node_SingleItem.cuh
// @brief general MERCATOR node that assumes that each thread
//        group processes a single input per call to run()
//
// MERCATOR
// Copyright (C) 2019 Washington University in St. Louis; all rights reserved.
//

#include <cassert>

#include "Node.cuh"

#include "ChannelBase.cuh"

#include "timing_options.cuh"


namespace Mercator  {

  //
  // @class Node_SingleItem
  // @brief MERCATOR node whose run() fcn takes one input per thread group
  // We use CRTP rather than virtual functions to derive subtypes of this
  // nod, so that the run() function can be inlined in fire().
  // The expected signature of run is
  //
  //   __device__ void run(const T &data)
  //
  // @tparam T type of input item
  // @tparam numChannels  number of output channels 
  // @tparam runWithAllThreads call run with all threads, or just as many
  //           as have inputs?
  // @tparam DerivedNodeType subtype that defines the run() function
  template<typename T, 
	   unsigned int numChannels,
	   unsigned int threadGroupSize,
	   unsigned int maxActiveThreads,
	   bool runWithAllThreads,
	   unsigned int THREADS_PER_BLOCK,
	   typename DerivedNodeType>
  class Node_SingleItem
    : public Node< NodeProperties<T, 
				  numChannels,
				  1, 
				  threadGroupSize,
				  maxActiveThreads,
				  runWithAllThreads,
				  THREADS_PER_BLOCK> > {
    
    typedef Node< NodeProperties<T,
				 numChannels,
				 1,
				 threadGroupSize,
				 maxActiveThreads,
				 runWithAllThreads,
				 THREADS_PER_BLOCK> > BaseType;
    
  public:
    
    __device__
    Node_SingleItem(unsigned int queueSize,
		    Scheduler *scheduler,
		    unsigned int region)
      : BaseType(queueSize, scheduler, region)
    {}
    
  protected:

    using BaseType::getChannel;
    using BaseType::maxRunSize; 
    
    // make these downwardly available to the user
    using BaseType::getNumActiveThreads;
    using BaseType::getThreadGroupSize;
    using BaseType::isThreadGroupLeader;
    
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
    
    //
    // @brief fire a node, consuming as much input 
    // from its queue as possible
    //
    // PRECONDITION: node is active (hence either flushing or has at
    // least maxRunSize inputs in its queue), and all its downstream
    // nodes are inactive (hence have at least enough space to hold
    // outputs from maxRunSize inputs in their queues).
    //
    // called with all threads
    
    __device__
    void fire()
    {
      unsigned int tid = threadIdx.x;
      
      TIMER_START(input);
      
      Queue<T> &queue = this->queue; 
      Queue<Signal> &signalQueue = this->signalQueue; 
      
      // # of items available to consume from queue
      unsigned int nDataToConsume = queue.getOccupancy();
      unsigned int nSignalsToConsume = signalQueue.getOccupancy();
      
      unsigned int nCredits = (nSignalsToConsume == 0
			       ? 0
			       : signalQueue.getHead().getCredit());
      
      
      // # of items already consumed from queue
      unsigned int nDataConsumed = 0;
      unsigned int nSignalsConsumed = 0;
      
      bool anyDSActive = false;

      while ((nDataConsumed < nDataToConsume ||
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
	  
	  unsigned int limit =
	    (nSignalsConsumed < nSignalsToConsume
	     ? nCredits 
	     : nDataToConsume - nDataConsumed);
	  
	  unsigned int nItems = min(limit, maxRunSize); 
	  
	  TIMER_STOP(input);
	  
	  TIMER_START(run);
	  
	  if (nItems > 0)
	    {
	      //
	      // Consume next nItems data items
	      //
	      
	      NODE_OCC_COUNT(nItems);
	      
	      const T &myData = 
		(tid < nItems
		 ? queue.getElt(nDataConsumed + tid)
		 : queue.getDummy()); // don't create a null reference
	      
	      if (runWithAllThreads || tid < nItems)
		{
		  DerivedNodeType *n = static_cast<DerivedNodeType *>(this);
		  n->run(myData);
		}
	      
	      __syncthreads();
	      
	      for (unsigned int c = 0; c < numChannels; c++)
		getChannel(c)->moveOutputToDSQueue();
	      
	      nDataConsumed += nItems;
	    }
	  
	  //
	  // Track credit to next signal, and consume if needed.
	  //
	  if (nSignalsToConsume > 0)
	    {
	      nCredits -= nItems;
	      
	      if (nCredits == 0)
		{
		  nCredits = this->signalHandler(nSignalsConsumed);
		  nSignalsConsumed++;
		}
	    }

	  TIMER_STOP(run);
	  
	  TIMER_START(output);
	  
	  __syncthreads();
	      
	  //
	  // Check whether any child has been activated
	  //
	  for (unsigned int c = 0; c < numChannels; c++)
	    anyDSActive |= getChannel(c)->checkDSFull();
	  
	  TIMER_STOP(output);
	  
	  TIMER_START(input);
	}
      
      // protect code above from queue changes below
      __syncthreads();
      
      if (IS_BOSS())
	{
	  COUNT_ITEMS(nDataConsumed);  // instrumentation
	  
	  queue.release(nDataConsumed);
	  signalQueue.release(nSignalsConsumed);

	  if (!signalQueue.empty())
	    signalQueue.getHead().setCredit(nCredits);
	  
	  if (nDataConsumed == nDataToConsume &&
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
		    NodeBase *dsNode = getChannel(c)->getDSNode();
		    this->propagateFlush(dsNode);
		    dsNode->activate();
		  }
		
		this->clearFlush();  // disable flushing
	      }
	  }
	}
      
      TIMER_STOP(input);
    }
  };
}  // end Mercator namespace

#endif
