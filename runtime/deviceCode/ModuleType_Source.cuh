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
    {
      numPending=UINT_MAX;
    }
    
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
      
      if (numPending == 0)
	this->setInTail(true);
    }
    
  private:

    using BaseType::dsQueueUtilAddress;
    using BaseType::dsActiveFlagAddress;

    using BaseType::getChannel;
    using BaseType::getFireableCount;
    using BaseType::maxOutputPerInput_AllChannels; 
 
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
    
    
    //
    // @brief fire the module, consuming pending inputs if possible
    // and scattering the results to downstream queues.
    //
    // This version of fire() takes uses the input source directly,
    // rather than an input queue, and the "run" operation simply
    // moves the inputs directly to the downstream queue for each
    // channel, since there is no filtering behavior.
    //

    #ifdef SCHEDULER_MINSWITCHES  
    //TODO:: simplify here 
    __device__
    virtual
    void fire()
    {
      int tid = threadIdx.x;
        #ifdef PRINTDBG
      int bid = blockIdx.x;
        #endif
      //if(tid==0) printf("source fireing --\n");
      
      // type of our downstream channels matchses our input type,
      // since the source module just copies its inputs downstream

      MOD_TIMER_START(gather);
      using Channel = typename BaseType::Channel<T>;
      const Channel* enumChannels[numChannels];      

      // enum channels once then use multiple
      for (unsigned int c = 0; c < numChannels; c++){
      enumChannels[c]= static_cast<Channel *>(getChannel(c));
      }
  

      unsigned int ew = this->ensembleWidth();
      __shared__ bool loopCont;     
  
      if(IS_BOSS()){
        loopCont=true;
      }
      
      __syncthreads(); // make sure all can see loop status

      while(loopCont){
        //if we fire (and not in tail) then it is implyed that we have enought for maxRunSize
        unsigned int totalFireable = min(ew, (unsigned int)numPending);

        MOD_OCC_COUNT(totalFireable);
        
        //count here DOES NOT matche the next next module... its greater
	//COUNT_ITEMS_INST(0, totalFireable);  // instrumentation

        MOD_TIMER_STOP(gather);
        MOD_TIMER_START(scatter);
        
        // Reserve space on all channels' downstream queues
        // for the data we are about to write, and get the
        // base pointer at which to write it for each queue.
        
        __shared__ unsigned int dsBase[numChannels];
        if (tid < numChannels){
          dsBase[tid] = enumChannels[tid]->directReserve(0, totalFireable);
        }
        __syncthreads(); // all threads must see dsBase[] values
        
        MOD_TIMER_STOP(scatter);
        MOD_TIMER_START(gather);
        
        //
        // move maxRunSize data items from the source to the downstream queues,
        // using all available threads.
        //
                
        // access input buffer only if we need an element, to avoid
        // non-copy construction of a T.
        //get no more tthen one ensamble width of daata
        T myData;
        if (tid < totalFireable)
          myData = source->get(pendingOffset + tid);

        #ifdef PRINTDBG
          if(tid==0)printf("\t%u: Source writing %u items downstream\n", bid, totalFireable);
        #endif

        MOD_TIMER_STOP(gather);
        MOD_TIMER_START(scatter);
        

        if (tid < totalFireable){
          // data needs no compaction, so transfer it directly to
          // the output queue for each channel, bypassing the
          // channel buffer
          if(IS_BOSS()){
	    COUNT_ITEMS_INST(0, totalFireable);  // instrumentation
          }
          for (unsigned int c = 0; c < numChannels; c++){
            enumChannels[c]->directWrite(0, myData, dsBase[c], tid); 
            //while we have this convient channel pointer, may as well check what the occupancy is like
            unsigned int dsQueue_rem = *dsQueueUtilAddress[0][c];
            //if there is not enough space to fire us again, activate DS 
            if(dsQueue_rem <= (ew*maxOutputPerInput_AllChannels)){ 
              *dsActiveFlagAddress[0][c]=1;
              //if(tid==0) printf("activating ds and breaking\n");
              loopCont =false;
              //maybe do this section in paraallel for all channels, then use 
            }
          }
        }
        
        MOD_TIMER_STOP(scatter);
        MOD_TIMER_START(gather);
    
        __syncthreads(); // protect run from later changes to pending and all see loop status
        
        //
        // Decrement the available data from our current reservation.
        // If we've exhausted it, try to get more.
        //
        if (IS_BOSS()){
          numPending    -= totalFireable;
          pendingOffset += totalFireable;

          if (!this->isInTail() && numPending == 0){
            numPending = source->reserve(REQ_SIZE, &pendingOffset);
            #ifdef PRINTDBG
              if(tid==0) printf("%u: source reserved %u\n", blockIdx.x, REQ_SIZE);
            #endif
            if (numPending == 0){ // no more input left to request!
              this->setInTail(true);
              //leave while loop
              #ifdef PRINTDBG
                if(tid==0) printf("%i: -----------------ENTERTING TAIL-----------------\n", bid);
              #endif
              loopCont =false;
            }
          }
        }
      
        __syncthreads(); // make sure all can see tail status
      
        MOD_TIMER_STOP(gather);
      }
      __syncthreads();
      // if(tid==0) printf("source done --\n");
      
    }

  #else

    __device__
    virtual
    void fire()
    {
      // type of our downstream channels matchses our input type,
      // since the source module just copies its inputs downstream
      using Channel = typename BaseType::Channel<T>;
      
      int tid = threadIdx.x;
      
      MOD_TIMER_START(gather);
      
      // obtain number of inputs that can be consumed by each instance,
      unsigned int totalFireable = getFireableCount(0);
      
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
      if (IS_BOSS())
	{
	  numPending    -= totalFireable;
	  pendingOffset += totalFireable;
	
	  if (!this->isInTail() && numPending == 0)
	    {
	      numPending = source->reserve(REQ_SIZE, &pendingOffset);
	      if (numPending == 0) // no more input left to request!
		this->setInTail(true);
	    }
	}
    
      __syncthreads(); // make sure all can see tail status
    
      MOD_TIMER_STOP(gather);

    }
  #endif
    
  private:
       
 
    size_t numPending;    // number of inputs left from last request
    size_t pendingOffset; // offset into source of next input to get
    
    size_t *tailPtr;
  };

}; // namespace Mercator

#endif
