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

#include "module_options.cuh"

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
		    NodeBase *parent,
		    Scheduler *scheduler)
      : BaseType(queueSize, parent, scheduler)
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
      using BaseType::gatherTimer;
      using BaseType::runTimer;
      using BaseType::scatterTimer;
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
      
      MOD_TIMER_START(gather);
      
      Queue<T> &queue = this->queue; 
      DerivedNodeType *n = static_cast<DerivedNodeType *>(this);
      
      // # of items left to consume from queue
      unsigned int nLeft = queue->getOccupancy();
      
      // # of items already consumed from queue
      unsigned int nConsumed = 0;

      // value of nLeft below which we shoul stop consuming items
      unsigned int lowerLimit = (isFlushing ? 0 : nLeft % maxRunSize);
      
      unsigned int mynDSActive = 0;
      
      while (nLeft > lowerLimit && mynDSActive == 0)
	{
	  unsigned int nToConsume = min(nLeft, maxRunSize);
	  
	  const T &myData = 
	    (tid < nToConsume
	     ? queue.getElt(nConsumed + tid);
	     : queue.getDummy()); // don't create a null reference
	  
	  MOD_TIMER_STOP(gather);
	  
	  MOD_TIMER_START(run);
	  
	  if (runWithAllThreads || tid < nToConsume)
	    n->run(myData);
	  
	  nLeft     -= nToConsume;
	  nConsumed += nToConsume;
	 
	  __syncthreads(); // all threads must see active channel state
	  
	  MOD_TIMER_STOP(run);
	  
	  MOD_TIMER_START(scatter);
	  
	  for (unsigned int c = 0; c < numChannels; c++)
	    {
	      // check whether each channel's downstream node was activated
	      mynDSActive += getChannel(c)->moveOutputToDSQueue();
	    }
	  
	  __syncthreads(); // all threads must see reset channel state
	  
	  MOD_TIMER_STOP(scatter);
	  MOD_TIMER_START(gather);
	}
      
      if (IS_BOSS())
	{
	  nDSActive = mynDSActive;
	  
	  if (nLeft <= lowerLimit) // not enough input to keep running
	    {
	      this->deactivate();
	      
	      if (isFlushing)
		{
		  // no more inputs to write -- force downstream nodes into
		  // flushing mode and activte them so that they fire.  Even if
		  // they have no input, they must fire once to propagate
		  // flush mode and the active flag to *their* downstreams.
		  for (unsigned int c = 0; c < numChannels; c++)
		    {
		      getChannel(c)->getDSNode()->setFlushing();
		      getChannel(c)->getDSNode()->activate();
		    }
		}
	    }
	  
	  COUNT_ITEMS(nConsumed);  // instrumentation
	  queue.release(nConsumed);
	}
      
      MOD_TIMER_STOP(gather);
      
      // FIXME: is this required?
      __syncthreads();
    }
  };
}  // end Mercator namespace

#endif
