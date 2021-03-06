#ifndef __NODEFUNCTION_USER_CUH
#define __NODEFUNCTION_USER_CUH

//
// @file NodeFunction_User.cuh
// @brief general MERCATOR node that assumes that each thread
//        group processes a single input per call to run()
//
// MERCATOR
// Copyright (C) 2021 Washington University in St. Louis; all rights reserved.
//

#include <cassert>

#include "NodeFunction.cuh"

#include "Channel.cuh"

#include "timing_options.cuh"

#include "support/collective_ops.cuh"

namespace Mercator  {

  //
  // @class NodeFunction_User
  // @brief MERCATOR node function whose run() fcn takes one input per thread
  // group. We use CRTP to derive subtypes of this node so that the
  // run() function can be inlined.  The expected signature
  // of run is
  //
  //   __device__ void run(const T &data, unsigned int nInputs)
  //
  // where only the first nInputs threads have input items
  //
  // @tparam T type of input item
  // @tparam numChannels  number of output channels
  // @tparam InputView type of input view passed to doRun()
  // @tparam THREADS_PER_BLOCK constant giving thread block size
  // @tparam threadGroupSize number of threads per input
  // @tparam maxActiveThreads max # of threads that can take input at once 
  // @tparam DerivedNodeFnKind subtype that defines the run() function
  //
  template<typename T,
	   unsigned int numChannels,
	   typename InputView,
	   unsigned int THREADS_PER_BLOCK,
	   unsigned int threadGroupSize,
	   unsigned int maxActiveThreads,
	   template <typename View> typename DerivedNodeFnKind>
  class NodeFunction_User : public NodeFunction<numChannels> {
    
    using BaseType = NodeFunction<numChannels>;
    using DerivedNodeFnType = DerivedNodeFnKind<InputView>;
    
    using BaseType::node;
    
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
    NodeFunction_User(RefCountedArena *parentArena)
      : BaseType(parentArena)
    {}
    
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
    unsigned int doRun(const InputView &view, 
		       size_t start,
		       unsigned int limit)
    {
      DerivedNodeFnType* const nf = static_cast<DerivedNodeFnType *>(this);
      
      unsigned int tid = threadIdx.x;
      unsigned int nFinished = 0;
      
      do
	{
	  unsigned int nItems = min(limit - nFinished, maxRunSize);
	  
	  //
	  // Consume next nItems data items
	  //
	  
	  const typename InputView::EltT myData =
	    (tid < nItems
	     ? view.get(start + nFinished + tid)
	     : view.get(start)); // don't create null ref -- assumes nItems > 0
	  
	  nf->run(myData, nItems);
	  
	  nFinished += nItems;
	  NODE_OCC_COUNT(nItems, maxRunSize);
	}
      while (nFinished < limit && !node->isDSActive());
      
      return nFinished;
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
      NODE_TIMER_STOP(node, user);
      NODE_TIMER_START(node, push);
      
      using Channel = Channel<DST>;
      
      Channel *channel = static_cast<Channel*>(node->getChannel(channelIdx));
      
      //
      // assign offsets in the output queue to threads that want to write
      // a value, and compute the total number of values to write
      //
      BlockScan<unsigned int, THREADS_PER_BLOCK> scanner;
      unsigned int totalToWrite;
      
      unsigned int dsOffset = scanner.exclusiveSum(pred, totalToWrite);
      
      // BEGIN WRITE basePtr, ds queue, node dsActive status
      // __syncthreads();   // elided due to sync inside exclusiveSum()
      
      __shared__ size_t basePtr;
      if (IS_BOSS())
	basePtr = channel->dsReserve(totalToWrite);
      
      __syncthreads(); // END WRITE basePtr, ds queue, node dsActive status
      
      if (pred)
	channel->dsWrite(basePtr, dsOffset, item);

      NODE_TIMER_STOP(node, push);
      NODE_TIMER_START(node, user);
    }
  };
}  // end Mercator namespace

#endif
