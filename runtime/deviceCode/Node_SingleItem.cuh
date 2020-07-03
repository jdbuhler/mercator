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
      if (!isFlushing)
        nToConsume = (nToConsume / maxRunSize) * maxRunSize;
      
      // # of items already consumed from queue
      unsigned int nConsumed = 0;
      
      unsigned int nCredits = this->currentCreditCounter;
      
      unsigned int mynDSActive = 0;
      
      while (nConsumed < nToConsume && mynDSActive == 0)
	{
#if 1
	  if (IS_BOSS())
	    printf("%d %p %d %d %d %d\n", 
		   blockIdx.x, this, nConsumed, nToConsume,  
		   this->numSignalsPending(), nCredits);
#endif
	  
	  // consume up to either next signal boundary or vector width
	  unsigned int limit =
	    (this->numSignalsPending() > 0 ? nCredits : maxRunSize);
	  
	  unsigned int nItems = min(nToConsume - nConsumed, limit); 
	  
	  NODE_OCC_COUNT(nItems);
	  
	  const T &myData = 
	    (tid < nItems
	     ? queue.getElt(nConsumed + tid)
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
	      // check whether each channel's downstream node was
	      // activated
	      mynDSActive += 
		getChannel(c)->moveOutputToDSQueue(this->getWriteThruId());
	    }
	  
	  TIMER_STOP(output);
	  
	  TIMER_START(input);
	  
	  if (this->numSignalsPending() > 0) 
	    {
	      nCredits -= nItems;
	      
	      // call the signal handler if we have reached 0 credit
	      if (nCredits == 0)
		{
		  // protect loop code from changes to numSignalsPending()
		  // and signal handler from changes to numOutputsProduced
		  __syncthreads();
		  
		  if (IS_BOSS())
		    this->currentCreditCounter = 0;
		  __syncthreads();
		  
		  bool dsSignalFull = this->signalHandler();
		  
		  // protect loop code from changes to numSignalsPending()
		  __syncthreads();
		  
		  nCredits = this->currentCreditCounter;
		  if (dsSignalFull)
		    break;
		}
	    }
	}
      
      // protect loop from credit changes below
      __syncthreads();
      
      if (IS_BOSS())
	{
	  // store final state of credit counter
	  this->currentCreditCounter = nCredits;
	  
	  COUNT_ITEMS(nConsumed);  // instrumentation
	  queue.release(nConsumed);
	  
	  nDSActive = mynDSActive;
	  
	  //Send the write thru ID you have if you have one
	  if (this->getWriteThruId() > 0) 
	    {
	      for(unsigned int c = 0; c < numChannels; ++c) 
		{
		  NodeBase *dsNode = getChannel(c)->getDSNode();
		  dsNode->setWriteThruId(this->getWriteThruId());
		  dsNode->activate();
		}
	    }
	  
	  if (nConsumed == nToConsume)
	  {
	    // less than a full ensemble remains, or 0 if flushing
	    this->deactivate(); 
	    
	    if (isFlushing)
	      {
		// no more inputs to read -- force downstream nodes
		// into flushing mode and activate them (if not
		// already active).  Even if they have no input,
		// they must fire once to propagate flush mode and
		// activate *their* downstream nodes.
		for (unsigned int c = 0; c < numChannels; c++)
		  {
		    NodeBase *dsNode = getChannel(c)->getDSNode();
		    dsNode->setFlushing(true);
		    dsNode->activate();
		  }
		
		nDSActive = numChannels;
		this->setFlushing(false);
	      }
	  }
	}
      
      TIMER_STOP(input);
    }
  };
}  // end Mercator namespace

#endif
