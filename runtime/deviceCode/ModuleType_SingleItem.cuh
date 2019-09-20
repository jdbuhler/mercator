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

#include "ChannelBase.cuh"

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
    using BaseType::checkFiringMask;
    using BaseType::maxRunSize; 
    
    using BaseType::ensembleWidth;
    using BaseType::numInputsPending;
    using BaseType::isInTail;
    using BaseType::deactivate;
    
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
    #ifdef SCHEDULER_MINSWITCHES

      //called with all threads
      __device__
      virtual
      void fire()
      {
        unsigned int tid = threadIdx.x;
      
              #ifdef PRINTDBG
        int bid = blockIdx.x;
              #endif

        MOD_TIMER_START(gather);
        unsigned int ew = ensembleWidth();
        bool inTail = isInTail();
        unsigned int worstCaseMult = this->maxOutputPerInput_AllChannels; 
        Queue<T> &queue = this->queue; 
        DerivedModuleType *mod = static_cast<DerivedModuleType *>(this);

        //for each node in the module
        for(unsigned int node=0; node<numInstances;node++){
          //if it was suppose to fire
          if(checkFiringMask(node)){

            //this is true at selectioni time , ensure by scheduler
            unsigned int spaceAvail = ew * worstCaseMult; 
            unsigned int numFired = 0;
            unsigned int totalToFire = numInputsPending(node);

            //round down to nearest multiple of ew
            unsigned int overfill = totalToFire % ew;
            totalToFire-=overfill;
            //we are in tail, and are force to compute safe amount to fire
            //overwrite what we had before  
            if(inTail){
              totalToFire = getFireableCount(node);
            }

            //keep going until we have exaused our
            //input queue (in chunks of ensamble width)
            while( numFired<totalToFire ){
             
              //num to fire is the smaller of stuff remaining and ew
              unsigned int numToFire = ew;
              unsigned int leftToFire = totalToFire - numFired;
              //overwrite numToFire
              if( leftToFire < ew){
                numToFire  = leftToFire;
              }
              
              
              if (spaceAvail < numToFire * worstCaseMult){
                break;
              } 

              assert(numToFire> 0);

              #ifdef PRINTDBG
              if(IS_BOSS()){
                printf("%i: \tNode %u pulling %u from %u (num pending)\n",
                        bid, 
                        node,  
                        numToFire,  
                        numInputsPending(node));
              }
              #endif

              const T &myData = 
                (tid < numToFire
                 ? queue.getElt(node, tid+numFired)
                 : queue.getDummy()); // don't create a null reference

              MOD_TIMER_STOP(gather);

              MOD_TIMER_START(run);
              if (tid < numToFire)
                mod->run(myData, node);
              __syncthreads(); // all threads must see active channel state
              MOD_TIMER_STOP(run);


              MOD_TIMER_START(scatter);
              numFired+=numToFire;
              for (unsigned int c = 0; c < numChannels; c++){
                  //update if we should fire again also flips active flag on ds node
                  unsigned int DSspace = getChannel(c)->compressCopyToDSQueue(node, isThreadGroupLeader());
                  if(spaceAvail>DSspace){
                    spaceAvail = DSspace; 
                  }
              
              }

              __syncthreads(); // all threads must see reset channel state
              MOD_TIMER_STOP(scatter);
              MOD_TIMER_START(gather);
            }

            // release ALL items that were consumed during node firing 
            if(IS_BOSS()){ //call single threaded
              COUNT_ITEMS_INST(node, numFired);  // instrumentation
              queue.release(node, numFired);
              //if there is still downstream space, 
              //but we made it here, that means we are out of input and should deactivate
              if(spaceAvail>= ew * worstCaseMult){
                //this->deactivate(node);
                deactivate(node);
              }
              
            }

            MOD_TIMER_STOP(gather);

            #ifdef PRINTDBG
              if(IS_BOSS()) printf("%i: \tNode %u fired, has %u remaining in-queue\n", bid ,node, numInputsPending(node));
            #endif
            //is this required 
            __syncthreads();
          }
        }
      }
    #else

      __device__
      virtual
      void fire()
      {
        unsigned int tid = threadIdx.x;
        
        MOD_TIMER_START(gather);
        
        // obtain number of inputs that can be consumed by each instance
        unsigned int fireableCount = 
          (tid < numInstances ? getFireableCount(tid) : 0);


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
    }
    #endif
  };  // end ModuleType class
}  // end Mercator namespace

#endif
