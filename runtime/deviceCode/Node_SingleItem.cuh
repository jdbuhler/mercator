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

#include "BufferedChannel.cuh"

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
  //
  template<typename T, 
	   unsigned int numChannels,
	   unsigned int threadGroupSize,
	   unsigned int maxActiveThreads,
	   bool runWithAllThreads,
	   unsigned int THREADS_PER_BLOCK,
	   typename DerivedNodeType>
  class Node_SingleItem
    : public Node<T,
		  numChannels,
		  THREADS_PER_BLOCK> {
    
    using BaseType = Node<T,
			  numChannels,
			  THREADS_PER_BLOCK>;

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
      numThreadGroups /* * numEltsPerGroup*/;
    
  public:

    ///////////////////////////////////////////////////////
    // INIT/CLEANUP KERNEL FUNCIIONS
    ///////////////////////////////////////////////////////
    
    __device__
    Node_SingleItem(unsigned int queueSize,
		    Scheduler *scheduler,
		    unsigned int region,
		    RefCountedArena *parentArena)
      : BaseType(queueSize, scheduler, region, parentArena)
    {
#ifdef INSTRUMENT_OCC
      occCounter.setMaxRunSize(maxRunSize);
#endif
    }

  protected:
    
    //
    // @brief Create and initialize a buffered output channel.
    //
    // @param c index of channel to initialize
    // @param outputsperInput max # of outputs produced per input
    //
    template<typename DST>
    __device__
    void initBufferedChannel(unsigned int c, 
			     unsigned int outputsPerInput,
			     bool isAgg = false)
    {
      assert(c < numChannels);
      assert(outputsPerInput > 0);
      
      // init the output channel -- should only happen once!
      assert(getChannel(c) == nullptr);
      
      using Channel = BufferedChannel<DST, THREADS_PER_BLOCK>;
      setChannel(c, new Channel(outputsPerInput,
				isAgg,
				numThreadGroups,
				threadGroupSize,
				1 /* numEltsPerGroup*/));
      
      // make sure alloc succeeded
      if (getChannel(c) == nullptr)
	{
	  printf("ERROR: failed to allocate channel object [block %d]\n",
		 blockIdx.x);

	  crash();
	}
    }
    
    ////////////////////////////////////////////////////////
    
  private:
    
    //
    // @brief doRun() prefers to have a full width of inputs for
    // the user's run function.
    //
    __device__
    unsigned int inputSizeHint() const
    { return maxRunSize; }
    
    
    //
    // @brief Feed inputs to the user's run function.
    // 
    __device__
    unsigned int doRun(const Queue<T> &queue, 
		       unsigned int start,
		       unsigned int limit)
    {
      unsigned int tid = threadIdx.x;
      
      unsigned int nItems = min(limit, maxRunSize);
      
      if (nItems > 0)
	{
	  //
	  // Consume next nItems data items
	  //
	  
	  NODE_OCC_COUNT(nItems);
	  
	  const T &myData =
	    (tid < nItems
	     ? queue.getElt(start + tid)
	     : queue.getDummy()); // don't create a null reference
	  
	  __syncthreads(); // BEGIN WRITE output buffer through push()
	  
	  if (runWithAllThreads || tid < nItems)
	    {
	      DerivedNodeType *n = static_cast<DerivedNodeType *>(this);
	      n->run(myData);
	    }
	  
	  __syncthreads(); // END WRITE output buffer through push()
	  
	  for (unsigned int c = 0; c < numChannels; c++)
	    {
	      BufferedChannelBase *channel =
		static_cast<BufferedChannelBase *>(getChannel(c));
	      channel->completePush();
	    }
	}
      
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
    // @param channelIdx channel to which to write the item
    //
    template<typename DST>
    __device__
    void push(const DST &item, unsigned int channelIdx = 0) const
    {
      using Channel = BufferedChannel<DST, THREADS_PER_BLOCK>;
      
      Channel *channel = static_cast<Channel*>(getChannel(channelIdx));
      
      channel->push(item);
    }
  };
}  // end Mercator namespace

#endif
