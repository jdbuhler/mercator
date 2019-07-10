#ifndef __MODULE_TYPE_SINGLEITEM_CUH
#define __MODULE_TYPE_SINGLEITEM_CUH

//
// @file ModuleType_SingleItem.cuh
// @brief general MERCATOR module that assumes that each thread
//        group processes a single input per call to run()
//
// MERCATOR
// Copyright (C) 2018 Washington University in St. Louis; all rights reserved.
//

#include <cassert>

#include "ModuleType.cuh"

#include "module_options.cuh"

#include "mapqueues/gather.cuh"

#include "Queue.cuh"


namespace Mercator  {

  //
  // @class ModuleType_SingleItem
  // @brief MERCATOR module whose run() fcn takes one input per thread group
  // We use CRTP rather than virtual functions to derive subtypes of this
  // module, so that the run() function can be inlined in gatherAndRun().
  // The expected signature of run is
  //
  //   __device__ void run(const T &data, InstTagT tag)
  //
  // @tparam T type of input item to module
  // @tparam numInstances number of instances of module
  // @tparam numChannels  number of channels in module
  // @tparam runWithAllThreads call run with all threads, or just as many
  //           as have inputs?
  // @tparam DerivedModuleType subtype that defines the run() function
  template<typename T, 
	   unsigned int numInstances,
	   unsigned int numChannels,
	   unsigned int threadGroupSize,
	   unsigned int maxActiveThreads,
	   bool runWithAllThreads,
	   unsigned int THREADS_PER_BLOCK,
	   typename DerivedModuleType>
  class ModuleType_SingleItem
    : public ModuleType< ModuleTypeProperties<T, 
					      numInstances,
					      numChannels,
					      1, 
					      threadGroupSize,
					      maxActiveThreads,
					      runWithAllThreads,
					      THREADS_PER_BLOCK> > {
    
    typedef ModuleType< ModuleTypeProperties<T,
					     numInstances,
					     numChannels,
					     1,
					     threadGroupSize,
					     maxActiveThreads,
					     runWithAllThreads,
					     THREADS_PER_BLOCK> > BaseType;
    
  public:
    
    __device__
    ModuleType_SingleItem(const unsigned int *queueSizes)
      : BaseType(queueSizes)
    {}
    
  protected:

    using typename BaseType::InstTagT;
    using          BaseType::NULLTAG;
        
    using BaseType::getChannel;
    using BaseType::getFireableCount;
    
    using BaseType::maxRunSize; 
    
    // make these downwardly available to the user
    using BaseType::getNumInstances;
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
      // @brief fire a module, consuming as much from its queue as possible
      //
      __device__
      virtual
      void fire()
      {
        unsigned int tid = threadIdx.x;
        
        MOD_TIMER_START(gather);
        
        // obtain number of inputs that can be consumed by each instance, potentially taking into account the firing mask, set durring sched 
    #ifdef SCHEDULER_MINSWITCHES  
        unsigned int fireableCount = 
          (tid < numInstances ? getMaskedFireableCount(tid) : 0);
    #else
        unsigned int fireableCount = 
          (tid < numInstances ? getFireableCount(tid) : 0);
    #endif


      // compute progressive sums of items to be consumed in each instance,
      // and replicate these sums in each WARP as Ai.
      using Gather = QueueGather<numInstances>;
      
      unsigned int totalFireable;
      unsigned int Ai = Gather::loadExclSums(fireableCount, totalFireable);  
      
      assert(totalFireable > 0);
      
      MOD_OCC_COUNT(totalFireable);
      
      Queue<T> &queue = this->queue; 
      
      // Iterate over inputs to be run in block-sized chunks.
      // Do both gathering and execution of inputs in each iteration.
      // Every thread in a group receives the same input. 
      for (unsigned int base = 0;
	   base < totalFireable; 
	   base += maxRunSize)
	{
	  unsigned int groupId = tid / threadGroupSize;
	  unsigned int idx     = base + groupId;
	  InstTagT     instIdx = NULLTAG;
	  unsigned int instOffset;
	  
	  // activeWarps = ceil( max run size / WARP_SIZE )
	  unsigned int activeWarps = 
	    (maxRunSize + WARP_SIZE - 1)/WARP_SIZE;
	  
	  // only execute warps that need to pull at least one input value
	  if (tid / WARP_SIZE < activeWarps)
	    {
	      // Compute queue and offset values for each thread's input 
	      Gather::BlockComputeQueues(Ai, idx, instIdx, instOffset);
	    }
	  
	  const T &myData = 
	    (idx < totalFireable
	     ? queue.getElt(instIdx, instOffset)
	     : queue.getDummy()); // don't create a null reference
	  
	  MOD_TIMER_STOP(gather);
	  MOD_TIMER_START(run);
	  
	  DerivedModuleType *mod = static_cast<DerivedModuleType *>(this);
	  
	  if (runWithAllThreads || idx < totalFireable)
	    mod->run(myData, instIdx);
	  
	  __syncthreads(); // all threads must see active channel state
	  
	  MOD_TIMER_STOP(run);
	  MOD_TIMER_START(scatter);
	  
	  for (unsigned int c = 0; c < numChannels; c++)
	    {
	      // mark first thread writing to each instance
	      bool isHead = (tid == 0 || instOffset == 0);
	      
	      getChannel(c)->scatterToQueues(instIdx,
					     isHead,	
					     isThreadGroupLeader());
	    }
	  
	  __syncthreads(); // all threads must see reset channel state
	  
	  MOD_TIMER_STOP(scatter);
	  MOD_TIMER_START(gather);
	}
      
      // protect use of queue->getElt() from changes to head pointer due
      // to release.
      __syncthreads();
      
      // release any items that we consumed in this firing
      if (tid < numInstances)
	{
	  COUNT_ITEMS(fireableCount);
	  queue.release(tid, fireableCount);
	}
      
      // make sure caller sees updated queue state
      __syncthreads();
      
      MOD_TIMER_STOP(gather);
      //MOD_TIMER_START(activate);
    
      DerivedModuleType *mod = static_cast<DerivedModuleType *>(this);
      //update active/ inactive status here
          
      //1. node just fired, so is clearly active,so lets check if it should stay active by looking at 
      //  active -> inactive when input queue  has fewer than v_i inputs remaining.
      unsigned int remainingItems =  mod->numInputsPending(instIdx);
      if(remainingItems < WARP_SIZE){
        mod->flipActiveFlag(instIdx);
      }
      //2. we may have just activated the DS node
      //  inactive DSnode becomes active when its input queue  has fewer than vi−1gi−1 spaces remaining.

      //MOD_TIMER_STOP(activate);
    
      //make sure all threads see the new active/inactive status flags
      //__syncthreads();


    }
  };  // end ModuleType class
}  // end Mercator namespace

#endif
