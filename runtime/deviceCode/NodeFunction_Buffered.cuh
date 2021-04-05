#ifndef __NODEFUNCTION_BUFFERED_CUH
#define __NODEFUNCTION_BUFFERED_CUH

//
// @file Node_Buffered.cuh
// @brief MERCATOR node function whose run() fcn takes one input per thread
// group and performs output buffering in push(), so that it may be called
// with a subset of threads.
//
// MERCATOR
// Copyright (C) 2021 Washington University in St. Louis; all rights reserved.
//

#include <cassert>

#include "NodeFunction.cuh"

#include "Channel.cuh"

#include "ChannelBuffer.cuh"

namespace Mercator  {

  //
  // @class NodeFunction_Buffered
  // @brief MERCATOR node whose run() fcn takes one input per thread group
  // and performs output buffering in push(), so that it may be called with
  // a subset of threads.  Note that each thread is assumed to take 0 or
  // 1 inputs.
  //
  // We use CRTP to derive subtypes of this node so that the
  // run() function can be inlined.  The expected signature
  // of run is
  //
  //   __device__ void run(const T &data)
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
  class NodeFunction_Buffered : public NodeFunction<numChannels> {
    
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
    NodeFunction_Buffered(RefCountedArena *parentArena)
      : BaseType(parentArena)
    {
      // init channel buffer array
      for (unsigned int c = 0; c < numChannels; ++c)
	channelBuffers[c] = nullptr;
    }
      
    //
    // @brief Create and initialize a buffer for an output channel.
    //
    // @param c index of channel to initialize
    // @param outputsperInput max # of outputs produced per input
    //
    template<typename DST>
    __device__
    void initChannelBuffer(unsigned int c, 
			   unsigned int outputsPerInput)
    {
      assert(c < numChannels);
      assert(outputsPerInput > 0);
	
      channelBuffers[c] = 
	new ChannelBuffer<DST, THREADS_PER_BLOCK>(outputsPerInput,
						  numThreadGroups,
						  threadGroupSize,
						  1 /* numEltsPerGroup */);
      // make sure alloc succeeded
      if (channelBuffers[c] == nullptr)
	{
	  printf("ERROR: failed to allocate buffer object [block %d]\n",
		 blockIdx.x);
	  crash();
	}
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
	  
	  __syncthreads(); // BEGIN WRITE output buffer through push()
	  
	  if (tid < nItems)
	    {
	      const typename InputView::EltT myData = 
		view.get(start + nFinished + tid);
	      nf->run(myData);
	    }
	  
	  __syncthreads(); // END WRITE output buffer through push()
      
	  //
	  // have each ChannelBuffer complete a push to the corresponding
	  // channel (which we can pass as an argument).
	  //
	  
	  for (unsigned int c = 0; c < numChannels; c++)
	    channelBuffers[c]->finishWrite(node->getChannel(c));

	  nFinished += nItems;
	  NODE_OCC_COUNT(nItems, maxRunSize);
	}
      while (nFinished < limit && !node->isDSActive());
      
      return nFinished;
    }
      
  protected:
      
    ChannelBufferBase* channelBuffers[numChannels];
      
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
    // @brief Write an output item to the indicated channel's buffer.
    //
    // @tparam DST Type of item to be written
    // @param item Item to be written
    // @param channelIdx channel to which to write the item
    //
    template<typename DST>
    __device__
    void push(const DST &item, unsigned int channelIdx = 0) const
    {
      //
      // get the ChannelBuffer object instead and push to that
      //
	
      using ChannelBuffer = ChannelBuffer<DST, THREADS_PER_BLOCK>;
	
      ChannelBuffer *cb = 
	static_cast<ChannelBuffer*>(channelBuffers[channelIdx]);
	
      cb->store(item);
    }
  };
}  // end Mercator namespace

#endif
