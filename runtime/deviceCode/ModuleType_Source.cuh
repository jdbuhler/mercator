#ifndef __MODULE_TYPE_SOURCE_CUH
#define __MODULE_TYPE_SOURCE_CUH

//
// @file ModuleType_Source.cuh
// @brief Module type that gets its input from a source object
//
// MERCATOR
// Copyright (C) 2018 Washington University in St. Louis; all rights reserved.
//

#include <cstddef>

#include "ModuleType.cuh"

#include "io/Source.cuh"

#include "module_options.cuh"

namespace Mercator  {

  //
  // @class ModuleType_Source
  // @brief Contains all functions and datatype-dependent info
  //         for a "source" module type.
  //
  // @tparam T type of input item to module
  // @tparam numChannels  number of channels in module
  //
  template<typename T, 
	   unsigned int numChannels,
	   unsigned int THREADS_PER_BLOCK>
  class ModuleType_Source : 
    public ModuleType< 
    ModuleTypeProperties<T,
			 1,                 // one instance only
			 numChannels,
			 1, 1,              // no run/scatter functions
			 THREADS_PER_BLOCK, // use all threads
			 true,              
			 THREADS_PER_BLOCK> > { 
    
    typedef ModuleType< ModuleTypeProperties<T,
					     1,
					     numChannels,
					     1, 1,
					     THREADS_PER_BLOCK,
					     true,
					     THREADS_PER_BLOCK> > BaseType;
    
    // request size when we go the input source.  Empirically,
    // we want this to be quite small to optimize load-balancing,
    // so that we don't claim a big chunk of input and then take 
    // longer to process it than other blocks.
    static const size_t REQ_SIZE = 2*THREADS_PER_BLOCK;
    
    const unsigned int zero;
    
  public: 
    
    //
    // @brief constructor
    //
    // @param itailPtr -- pointer to a global tail pointer shared by
    // all blocks, used to collectively manage data allocations from
    // the source.
    //
    __device__
    ModuleType_Source(size_t *itailPtr)
      : zero(0),
	BaseType(&zero),
	source(nullptr),
	numPending(0),
	tailPtr(itailPtr)
    {}
    
    //
    // @brief construct a Source object from the raw data passed down
    // to the device.
    //
    // @param sourceData source data passed from host to device
    // @return a Source object whose subtype matches the input data
    //
    __device__
    Source<T> *createSource(const SourceData<T> &sourceData,
			    SourceMemory<T> *mem)
    {
      Source<T> *source;
      
      switch (sourceData.kind)
	{
	case SourceData<T>::Buffer:
	  source = new (mem) SourceBuffer<T>(sourceData.bufferData,
					     tailPtr);
	  break;
	  
	case SourceData<T>::Range:
	  source = new (mem) SourceRange<T>(sourceData.rangeData,
					    tailPtr);
	  
	  break;
	}
      
      return source;
    }
    
    
    //
    // @brief prepare for the app's main kernel to run
    // Set our input source, then try to get an initial reservation
    // from the input source, so that the application has work to do.
    // If no work is available, set our tail state true to so indicate.
    //
    // Called single-threaded
    //
    // @param source input source  to use
    //
    __device__
    void setInputSource(Source<T> *isource)
    {
      source = isource;
      this->setInTail(false);
      
      numPending = source->reserve(REQ_SIZE, &pendingOffset);
      
	//stimcheck:  NEED TO SEND SIGNAL HERE, BLOCK DOES NOT NEED TO RUN ANY INPUT
      if (numPending == 0) {
	this->setInTail(true);
	this->setInTailInit(true);
	//printf("GOT HERE, blockIdx = %d\n", blockIdx.x);
      }

	      if (numPending == 0) { // no more input left to request!
      		using Channel = typename BaseType::Channel<T>;
		this->setInTail(true);
		//stimcheck:  Send tail Signal instead of setting for all downstream nodes immediately
		Signal s;
		s.setTag(Signal::SignalTag::Tail);


		//stimcheck: Send the isTail Signal to all channels
     		 // Reserve space on all channels' downstream queues
  		 // for the data we are about to write, and get the
      		// base pointer at which to write it for each queue.
      		__shared__ unsigned int dsSignalBase[numChannels];
	        for (unsigned int c = 0; c < numChannels; c++)
		{
		  const Channel *channel = 
	   	  static_cast<Channel *>(getChannel(c));
	  
	  	  dsSignalBase[c] = channel->directSignalReserve(0, 1);
		}

	      // data needs no compaction, so transfer it directly to
	      // the output queue for each channel, bypassing the
	      // channel buffer
	      //if(blockIdx.x == 0) {
	      for (unsigned int c = 0; c < numChannels; c++)
	      //if(tid < numChannels)
		{
		  const Channel *channel = 
		    static_cast<Channel *>(BaseType::getChannel(c));
		  
		  channel->directSignalWrite(0, s, dsSignalBase[c], 0); 
		}
		//printf("INSIDE TOTAL FIREABLE: %d, NUM PENDING: %d\n", totalFireable, numPending);

		//printf("PUSHED SIGNAL, %d, %d, blkIdx.x %d, thrdIdx.x %d, dsSignalBase: %d\n", (this->isThreadGroupLeader() ? 1 : 0), numChannels, blockIdx.x, threadIdx.x, dsSignalBase[tid]);
              }
		//}
		//printf("PUSHED SIGNAL, %d, %d, %d, dsSignalBase: %d\n", (this->isThreadGroupLeader() ? 1 : 0), numChannels, blockIdx.x, dsSignalBase[tid]);

    }
    
  private:

    using BaseType::getChannel;
    using BaseType::getFireableCount;
    
#ifdef INSTRUMENT_TIME
    using BaseType::gatherTimer;
    using BaseType::runTimer;
    using BaseType::scatterTimer;
#endif

#ifdef INSTRUMENT_FG_TIME
   using BaseType::fineGrainedTimer;
#endif

#ifdef INSTRUMENT_OCC
    using BaseType::occCounter;
#endif

#ifdef INSTRUMENT_COUNTS
    using BaseType::itemCounter;
#endif
    
    Source<T>* source;

    //
    // @brief number of inputs currently enqueued for this module
    //
    // @param instIdx index for which to check pending count (must be 0)
    //
    __device__
    virtual
    unsigned int numInputsPending(unsigned int instIdx) const
    { 
      assert(instIdx == 0);
      return numPending; 
    }
    
    //
    // @brief maximum number of inputs that could ever be enqueued for
    // this module at one time.  We never have more inputs pending than
    // can be requested from the source at once.
    //
    __device__
    virtual
    unsigned int maxPending() const
    { return REQ_SIZE; }
    
    
    // stimcheck: Need to fire for signal queues here as well (isTail Signal)
    //
    // @brief fire the module, consuming pending inputs if possible
    // and scattering the results to downstream queues.
    //
    // This version of fire() takes uses the input source directly,
    // rather than an input queue, and the "run" operation simply
    // moves the inputs directly to the downstream queue for each
    // channel, since there is no filtering behavior.
    //
    __device__
    virtual
    void fire()
    {
      // type of our downstream channels matchses our input type,
      // since the source module just copies its inputs downstream
      using Channel = typename BaseType::Channel<T>;
      
      //printf("SOURCE MODULE CALLED\n");

      int tid = threadIdx.x;
      
      MOD_TIMER_START(gather);
      
      // determine how many items we are going to consume
      unsigned int totalFireable = getFireableCount(0);

      __syncthreads();
      //if(IS_BOSS())
      //	printf("getFireableCount(0) = %d\n", totalFireable);
      //__syncthreads();
      
      assert(totalFireable > 0);
      assert(totalFireable <= numPending);
      
      MOD_OCC_COUNT(totalFireable);
      
      if (tid == 0)
	COUNT_ITEMS(totalFireable);

      MOD_TIMER_STOP(gather);
      MOD_TIMER_START(scatter);
      
      // Reserve space on all channels' downstream queues
      // for the data we are about to write, and get the
      // base pointer at which to write it for each queue.
      __shared__ unsigned int dsBase[numChannels];
      if (tid < numChannels)
	{
	  const Channel *channel = 
	    static_cast<Channel *>(getChannel(tid));
	  
	  dsBase[tid] = channel->directReserve(0, totalFireable);
	}
      __syncthreads(); // all threads must see dsBase[] values

      
      MOD_TIMER_STOP(scatter);
      MOD_TIMER_START(gather);
      
      //
      // move the data from the source to the downstream queues,
      // using all available threads.
      //
      
      for (unsigned int base = 0; 
	   base < totalFireable;
	   base += THREADS_PER_BLOCK)
	{
	  unsigned int idx = base + tid;
	  	  
	  // access input buffer only if we need an element, to avoid
	  // non-copy construction of a T.
	  T myData;
	  if (idx < totalFireable)
	    myData = source->get(pendingOffset + idx);
	  
	  MOD_TIMER_STOP(gather);
	  MOD_TIMER_START(scatter);

	  if (idx < totalFireable)
	    {
	      // data needs no compaction, so transfer it directly to
	      // the output queue for each channel, bypassing the
	      // channel buffer
	      for (unsigned int c = 0; c < numChannels; c++)
		{
		  const Channel *channel = 
		    static_cast<Channel *>(getChannel(c));
		  
		  channel->directWrite(0, myData, dsBase[c], idx); 
		}
	    }
	  
	  MOD_TIMER_STOP(scatter);
	  MOD_TIMER_START(gather);
	}
      
      __syncthreads(); // protect run from later changes to pending
      
      //
      // Decrement the available data from our current reservation.
      // If we've exhausted it, try to get more.
      //
      
/*
      __shared__ unsigned int dsSignalBase[numChannels];
      if (tid < numChannels)
	{
         const Channel *channel = 
	   static_cast<Channel *>(getChannel(tid));
	  
	    dsSignalBase[tid] = channel->directSignalReserve(0, 1);
	}
      __syncthreads(); // all threads must see dsBase[] values
*/
      if (IS_BOSS())
	{
	  numPending    -= totalFireable;
	  pendingOffset += totalFireable;

		//printf("TOTAL FIREABLE: %d, NUM PENDING: %d\n", totalFireable, (numPending == 0 ? 1 : 0));
	/*
	  if(this->isInTailInit()) {
			printf("IS IN TAIL INIT\n");
		}	
	else {
			printf("NOT IS IN TAIL INIT FOR BLOCK %d, numPending %d\n", blockIdx.x, numPending);
	}
	*/
	  if ((!this->isInTail() && numPending == 0) || this->isInTailInit())
	    {
	      numPending = source->reserve(REQ_SIZE, &pendingOffset);
		//printf("INSIDE TOTAL FIREABLE: %d, NUM PENDING: %d\n", totalFireable, numPending);
			//printf("SECOND BLOCK %d, numPending %d\n", blockIdx.x, numPending);
	      if (numPending == 0) { // no more input left to request!
		this->setInTail(true);
		//stimcheck:  Send tail Signal instead of setting for all downstream nodes immediately
		Signal s;
		s.setTag(Signal::SignalTag::Tail);
		//s.setCredit(totalFireable); 

		//stimcheck: Send the isTail Signal to all channels
		//unsigned int instOffset = 0;
		//unsigned int instIdx = 0;

		/*
		for(unsigned int c = 0; c < numChannels; ++c) {
			//this->pushSignal(s, instIdx, c);
			bool isHead = (tid == 0 || instOffset == 0);
			//getChannel(c)->scatterToSignalQueues(instIdx,
			//				     isHead,
			//				     this->isThreadGroupLeader());
		}
		*/

     		 // Reserve space on all channels' downstream queues
  		 // for the data we are about to write, and get the
      		// base pointer at which to write it for each queue.
      		__shared__ unsigned int dsSignalBase[numChannels];
      		//unsigned int dsSignalBase[numChannels];
	        //if(blockIdx.x == 0) {
      		//if (tid < numChannels)
	        for (unsigned int c = 0; c < numChannels; c++)
		{
		  const Channel *channel = 
	   	  static_cast<Channel *>(getChannel(c));

		  s.setCredit(channel->dsPendingOccupancy(tid));	  

	  	  dsSignalBase[c] = channel->directSignalReserve(0, 1);
		}
      		//__syncthreads(); // all threads must see dsBase[] values

	      // data needs no compaction, so transfer it directly to
	      // the output queue for each channel, bypassing the
	      // channel buffer
	      //if(blockIdx.x == 0) {
	      for (unsigned int c = 0; c < numChannels; c++)
	      //if(tid < numChannels)
		{
		  const Channel *channel = 
		    static_cast<Channel *>(getChannel(c));

		  //s.setCredit(channel->dsPendingOccupancy(tid));	  
		  
		  channel->directSignalWrite(0, s, dsSignalBase[c], 0); 
		  //printf("Wrote Tail Signal to channel %d [SOURCE, %d]\tpending occupancy: %d", c, blockIdx.x, s.getCredit());
		  //printf("Wrote Tail Signal to channel\n");
		}
		  //printf("Wrote Tail Signal to channel\n");
		//printf("Wrote Tail Signals to %d channels [SOURCE, %d]\tpending occupancy: %d", numChannels, blockIdx.x);
		//printf("INSIDE TOTAL FIREABLE: %d, NUM PENDING: %d\n", totalFireable, numPending);

		//printf("PUSHED SIGNAL, %d, %d, blkIdx.x %d, thrdIdx.x %d, dsSignalBase: %d\n", (this->isThreadGroupLeader() ? 1 : 0), numChannels, blockIdx.x, threadIdx.x, dsSignalBase[tid]);
              }
		//}
		//printf("PUSHED SIGNAL, %d, %d, %d, dsSignalBase: %d\n", (this->isThreadGroupLeader() ? 1 : 0), numChannels, blockIdx.x, dsSignalBase[tid]);
	    }
	}
    
      __syncthreads(); // make sure all can see tail status
    
      MOD_TIMER_STOP(gather);
    }

    
  private:
    
    size_t numPending;    // number of inputs left from last request
    size_t pendingOffset; // offset into source of next input to get
    
    size_t *tailPtr;
  };

}; // namespace Mercator

#endif
