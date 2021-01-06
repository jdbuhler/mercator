#ifndef __NODE_SINGLEITEM_CUH
#define __NODE_SINGLEITEM_CUH

//
// @file Node_SingleItem.cuh
// @brief general MERCATOR node that assumes that each thread
//        group processes a single input per call to run()
//
// MERCATOR
// Copyright (C) 2020 Washington University in St. Louis; all rights reserved.
//

#include <cassert>

#include "Node.cuh"

#include "Channel.cuh"

#include "support/collective_ops.cuh"

namespace Mercator  {

  //
  // @class Node_SingleItem
  // @brief MERCATOR node whose run() fcn takes one input per thread group
  // We use CRTP rather than virtual functions to derive subtypes of this
  // nod, so that the run() function can be inlined in fire().
  // The expected signature of run is
  //
  //   __device__ void run(const T &data, unsigned int nInputs)
  //
  // where only the first nInputs threads have input items
  //
  // @tparam T type of input item
  // @tparam numChannels  number of output channels 
  // @tparam runWithAllThreads call run with all threads, or just as many
  //           as have inputs?
  // @tparam DerivedNodeType subtype that defines the run() function
  //
  template<typename T, 
	   unsigned int numChannels,
	   unsigned int threadGroupSize,
	   unsigned int maxActiveThreads,
	   unsigned int THREADS_PER_BLOCK,
	   typename DerivedNodeType>
  class Node_SingleItem
    : public Node<T,
		  numChannels,
		  THREADS_PER_BLOCK,
		  Node_SingleItem<T, 
				  numChannels, 
				  threadGroupSize, 
				  maxActiveThreads, 
				  THREADS_PER_BLOCK, 
				  DerivedNodeType>> {
    
    using BaseType = Node<T,
			  numChannels,
			  THREADS_PER_BLOCK,
			  Node_SingleItem<T, 
					  numChannels, 
					  threadGroupSize, 
					  maxActiveThreads, 
					  THREADS_PER_BLOCK, 
					  DerivedNodeType>>;
    
    using BaseType::getChannel;
    
#ifdef INSTRUMENT_OCC
    using BaseType::occCounter;
#endif
    
    // actual maximum # of possible active threads in this block
    static const unsigned int deviceMaxActiveThreads =
      (maxActiveThreads > THREADS_PER_BLOCK 
       ? THREADS_PER_BLOCK 
       : maxActiveThreads);
    
    // number of thread groups (no partial groups allowed!)
    static const unsigned int numThreadGroups = 
      deviceMaxActiveThreads / threadGroupSize;
    
    // max # of active threads assumes we only run full groups
    static const unsigned int numActiveThreads =
      numThreadGroups * threadGroupSize;
    
  protected:
    
    // maximum number of inputs that can be processed in a single 
    // call to the node's run() function
    static const unsigned int maxRunSize =
      numThreadGroups;
    
  public:

    ///////////////////////////////////////////////////////
    // INIT/CLEANUP KERNEL FUNCIIONS
    ///////////////////////////////////////////////////////
    
    __device__
    Node_SingleItem(Scheduler *scheduler,
		    unsigned int region,
		    NodeBase *usNode,
		    unsigned int usChannel,
		    unsigned int queueSize,
		    RefCountedArena *parentArena)
      : BaseType(scheduler, region, usNode, usChannel,
		 queueSize, parentArena)
    {
#ifdef INSTRUMENT_OCC
      occCounter.setMaxRunSize(maxRunSize);
#endif
    }

    ////////////////////////////////////////////////////////
    
    //
    // doRun() prefers to have a full width of inputs for
    // the user's run function.
    //
    static const unsigned int inputSizeHint = maxRunSize;
    
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
    unsigned int doRun(const Queue<T> &queue, 
		       unsigned int start,
		       unsigned int limit)
    {
      unsigned int tid = threadIdx.x;
      
      unsigned int nItems = min(limit, maxRunSize);
      
      //
      // Consume next nItems data items
      //
      
      NODE_OCC_COUNT(nItems);
      
      const T &myData =
	(tid < nItems
	 ? queue.getElt(start + tid)
	 : queue.getDummy()); // don't create a null reference
      
      DerivedNodeType *n = static_cast<DerivedNodeType *>(this);
      n->run(myData, nItems);
      
      return nItems;
    }
    
  protected:
    
    ///////////////////////////////////////////////////////////////////
    // RUN-FACING FUNCTIONS 
    // These functions expose documented properties and behavior of the 
    // node to the user's run(), init(), and cleanup() functions.
    ///////////////////////////////////////////////////////////////////
  
    //
    // @brief get the max number of active threads
    //
    __device__
    unsigned int getNumActiveThreads() const
    { return numActiveThreads; }
    
    //
    // @brief get the size of a thread group
    //
    __device__
    unsigned int getThreadGroupSize() const
    { return threadGroupSize; }
    
    //
    // @brief return true iff we are the 0th thread in our group
    //
    __device__
    bool isThreadGroupLeader() const
    { return (threadIdx.x % threadGroupSize == 0); }
    
    //
    // @brief Write an output item to the indicated channel.
    //
    // @tparam DST Type of item to be written
    // @param item Item to be written
    // @param pred predicate indicating whether thread should write
    // @param channelIdx channel to which to write the item
    //
    template<typename DST>
    __device__
    void push(const DST &item, bool pred, unsigned int channelIdx = 0) const
    {
      //
      // assign offsets in the output queue to threads that want to write
      // a value, and compute the total number of values to write
      //
      BlockScan<unsigned int, THREADS_PER_BLOCK> scanner;
      unsigned int totalToWrite;
      
      unsigned int dsOffset = scanner.exclusiveSum(pred, totalToWrite);
      
      using Channel = Channel<DST>;
      
      Channel *channel = static_cast<Channel*>(getChannel(channelIdx));
      
      __syncthreads(); // BEGIN WRITE basePtr, ds queue
      
      __shared__ unsigned int basePtr;
      if (IS_BOSS())
	basePtr = channel->dsReserve(totalToWrite);
      
      __syncthreads(); // END WRITE basePtr, ds queue
      
      if (pred)
	channel->dsWrite(basePtr, dsOffset, item);
    }
  };
}  // end Mercator namespace

#endif
