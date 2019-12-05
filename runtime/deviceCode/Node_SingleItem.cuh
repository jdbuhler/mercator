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
		    Scheduler *scheduler)
      : BaseType(queueSize, scheduler)
    {}
    
  protected:

    using BaseType::getChannel;
    using BaseType::getDSNode;
    using BaseType::maxRunSize; 
    
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
      if (!isFlushing())
	nToConsume = (nToConsume / maxRunSize) * maxRunSize;
            
      // # of items already consumed from queue
      unsigned int nConsumed = 0;
      
      // is at least one downstream queue full?
      bool dsFull = false;
      
      while (nConsumed < nToConsume && !dsFull)
	{
	  unsigned int nItems = min(nToConsume - nConsumed, maxRunSize);
	  
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
	  
	  __syncthreads(); // all threads can see ds queue state
	  
	  for (unsigned int c = 0; c < numChannels; c++)
	    {
	      // check whether each channel's downstream node should
	      // be activated.  If so, its queue is full, and we
	      // need to stop firing.
	      if (getChannel(c)->checkDSFull(maxRunSize))
		{
		  dsFull = true;
		  
		  if (IS_BOSS())
		    getDSNode(c)->activate();
		}
	    }
	  
	  TIMER_STOP(output);
	  
	  TIMER_START(input);
	}
      
      if (IS_BOSS())
	{
	  COUNT_ITEMS(nConsumed);  // instrumentation
	  queue.release(nConsumed);
	  
	  if (nConsumed == nToConsume)
	    {
	      // less than a full ensemble remains, or 0 if flushing
	      this->deactivate(); 
	      
	      if (isFlushing())
		{
		  // no more inputs to read -- force downstream nodes
		  // into flushing mode and activte them (if not
		  // already active).  Even if they have no input,
		  // they must fire once to propagate flush mode and
		  // activate *their* downstream nodes.
		  for (unsigned int c = 0; c < numChannels; c++)
		    {
		      NodeBase *dsNode = getDSNode(c);
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
