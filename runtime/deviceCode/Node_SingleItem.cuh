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

#include "Queue.cuh"


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
		    Scheduler *scheduler)
      : BaseType(queueSize, scheduler)
    {}
    
  protected:

    using BaseType::getChannel;
    using BaseType::maxRunSize; 
    
    using BaseType::nDSActive;
    using BaseType::isFlushing;
    
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
    virtual
    void fire()
    {
      unsigned int tid = threadIdx.x;
      
      TIMER_START(input);
      
      Queue<T> &queue = this->queue; 
      DerivedNodeType *n = static_cast<DerivedNodeType *>(this);
      
      // # of items available to consume from queue
      unsigned int nToConsume = queue.getOccupancy();
      
      // unless we are flushing, round # to consume down to a multiple
      // of ensemble width.
      //if (!isFlushing)
	//nToConsume = (nToConsume / maxRunSize) * maxRunSize;
      
      // # total of items consumed from queue
      unsigned int nTotalConsumed = 0;

      // # total of items to consume from the queue
      //unsigned int nTotalToConsume = queue.getOccupancy();
      unsigned int nTotalToConsume = 0;

      //if (!isFlushing)
	//nTotalToConsume = (nToConsume / maxRunSize) * maxRunSize;

      unsigned int mynDSActive = 0;

      // True when a downstream signal queue is full, so we stop firing.
      bool dsSignalFull = false;
      unsigned int nConsumed = 0;
      
	//Perform SAFIrE scheduling while we have signals.
      while (this->numSignalsPending() > 0 && !dsSignalFull && mynDSActive == 0)
	{
	      // # of items already consumed from queue
	      nConsumed = 0;
	      nToConsume = this->currentCreditCounter;

		//Can ignore flushing here, since we need to get to signal boundary.
	      //if (!isFlushing)
		//nToConsume = (this->currentCreditCounter / maxRunSize) * maxRunSize;

	      nTotalToConsume += nToConsume;

	      while (nConsumed < nToConsume && mynDSActive == 0)
		{
		  unsigned int nItems = min(nToConsume - nConsumed, maxRunSize);
		  
		  NODE_OCC_COUNT(nItems);
		  
		  const T &myData = 
		    (tid < nItems
		     //? queue.getElt(nConsumed + tid)
		     ? queue.getElt(nTotalConsumed + nConsumed + tid)
		     : queue.getDummy()); // don't create a null reference
		  
		  TIMER_STOP(input);
		  
		  TIMER_START(run);
	
		  if (runWithAllThreads || tid < nItems)
		    {
		      n->run(myData);
		    }
		  nConsumed += nItems;
		  
		  TIMER_STOP(run);
		  
		  TIMER_START(output);
		  
		  for (unsigned int c = 0; c < numChannels; c++)
		    {
		      // check whether each channel's downstream node was activated
		      mynDSActive += getChannel(c)->moveOutputToDSQueue();
		    }
		  
		  TIMER_STOP(output);
		  
		  TIMER_START(input);
		}
		nTotalConsumed += nConsumed;	//nConsumed Should be the same as nToConsume here
		this->currentCreditCounter -= nConsumed;

		//Syncthreads before entering the signal handeler, need to make sure that every
		//thread knows the current consumed totals.
		__syncthreads();

		dsSignalFull = this->signalHandler();

		__syncthreads();
	}

	//Use normal nConsumed and nToConsume values after signals are all handled.
      //unsigned int nConsumed = 0;

      __syncthreads();
      nToConsume = queue.getOccupancy()- nTotalConsumed;

      nConsumed = 0;

      __syncthreads();

      // unless we are flushing, round # to consume down to a multiple
      // of ensemble width.
      if (!isFlushing)
	nToConsume = (nToConsume / maxRunSize) * maxRunSize;

      __syncthreads();

      nTotalToConsume += nToConsume;

      __syncthreads();

	//Resume normal AFIE scheduling once we have no signals remaining.
      while (nConsumed < nToConsume && mynDSActive == 0 && !dsSignalFull)
	{
	      while (nConsumed < nToConsume && mynDSActive == 0)
		{
		  unsigned int nItems = min(nToConsume - nConsumed, maxRunSize);
		  
		  NODE_OCC_COUNT(nItems);
		  
		  const T &myData = 
		    (tid < nItems
		     //? queue.getElt(nConsumed + tid)
		     ? queue.getElt(nTotalConsumed + nConsumed + tid)
		     : queue.getDummy()); // don't create a null reference
		  
		  TIMER_STOP(input);
		  
		  TIMER_START(run);
	
		  if (runWithAllThreads || tid < nItems)
		    {
		      n->run(myData);
		    }
		  nConsumed += nItems;
		  
		  TIMER_STOP(run);
		  
		  TIMER_START(output);
		  
		  for (unsigned int c = 0; c < numChannels; c++)
		    {
		      // check whether each channel's downstream node was activated
		      mynDSActive += getChannel(c)->moveOutputToDSQueue();
		    }
		  
		  TIMER_STOP(output);
		  
		  TIMER_START(input);
		}
	}
	__syncthreads();
      nTotalConsumed += nConsumed;
	__syncthreads();
      
	//Release items as normal.
      if (IS_BOSS())
	{
	  COUNT_ITEMS(nTotalConsumed);  // instrumentation
	  queue.release(nTotalConsumed);
	  
	  nDSActive = mynDSActive;

	  if (nTotalConsumed == nTotalToConsume)
	  //if (nTotalConsumed == nToConsume)
	    {
	      // less than a full ensemble remains, or 0 if flushing
	      this->deactivate(); 
	      
	      if (isFlushing)
		{
		  // no more inputs to read -- force downstream nodes
		  // into flushing mode and activte them (if not
		  // already active).  Even if they have no input,
		  // they must fire once to propagate flush mode and
		  // activate *their* downstream nodes.
		  for (unsigned int c = 0; c < numChannels; c++)
		    {
		      NodeBase *dsNode = getChannel(c)->getDSNode();
		      dsNode->setFlushing();
		      dsNode->activate();
		    }
		}
	    }
	}
      
      TIMER_STOP(input);
    }
  };
}  // end Mercator namespace

#endif
