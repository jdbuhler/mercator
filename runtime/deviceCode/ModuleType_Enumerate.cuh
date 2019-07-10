#ifndef __MODULE_TYPE_ENUMERATE_CUH
#define __MODULE_TYPE_ENUMERATE_CUH

//
// @file ModuleType_Enumerate.cuh
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
  // @class ModuleType_Enumerate
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
  class ModuleType_Enumerate
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
    ModuleType_Enumerate(const unsigned int *queueSizes)
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

    //Queue<T> *parentBuffer[numInstances]; 
    //
    //
    //
    __device__
    void run(T inputItem, InstTagT nodeIdx)
    {
	  DerivedModuleType *mod = static_cast<DerivedModuleType *>(this);
	  
	  //if (runWithAllThreads || idx < totalFireable)
	    mod->findCount(nodeIdx);
    }

    //
    // @brief fire a module, consuming as much from its queue as possible
    //
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

      //stimcheck:  If the scheduler determined that there were fireable data elements, fire them, otherwise fire no data, syncthreads, and process signals.
      if(totalFireable > 0) {

      assert(totalFireable > 0);
      
      MOD_OCC_COUNT(totalFireable);
      
      Queue<T> &queue = this->queue; 

      // Iterate over inputs to be run in block-sized chunks.
      // Do both gathering and execution of inputs in each iteration.
      // Every thread in a group receives the same input. 
      //for (unsigned int base = 0;
	//   base < totalFireable; 
	//   base += maxRunSize)
	//{
	
	  unsigned int base = 0;	//Dummy var to replace old loop	
	  //this->signalHandler();

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
	  
	  //const T &myData = 
	  //  (idx < totalFireable
	  //   ? queue.getElt(instIdx, instOffset)
	  //   : queue.getDummy()); // don't create a null reference
	  const T &myData = queue.getElt(instIdx, 0);
	  
	  MOD_TIMER_STOP(gather);
	  MOD_TIMER_START(run);
	  
	  DerivedModuleType *mod = static_cast<DerivedModuleType *>(this);
	  
	  //if (runWithAllThreads || idx < totalFireable)
	    mod->run(myData, instIdx);
	  
	  __syncthreads(); // all threads must see active channel state
	  
	  MOD_TIMER_STOP(run);
	  MOD_TIMER_START(scatter);
	  
	  //unsigned int numProduced[numChannels];
	  for (unsigned int c = 0; c < numChannels; c++)
	    {
	      // mark first thread writing to each instance
	      bool isHead = (tid == 0 || instOffset == 0);
	      
	      //numProduced[c] =
	      getChannel(c)->scatterToQueues(instIdx,
					     isHead,	
					     isThreadGroupLeader());

	      //stimcheck: Scatter to Signal Queues as well as data queues
	      //We know by this point which Signals will continue downstream
	      //getChannel(c)->scatterToQueues(instIdx,
		//			     isHead,	
		//			     isThreadGroupLeader());
	    }
	  
	  __syncthreads(); // all threads must see reset channel state
	  
	  MOD_TIMER_STOP(scatter);
	  MOD_TIMER_START(gather);
	//} //end for
      
      // protect use of queue->getElt() from changes to head pointer due
      // to release.
      __syncthreads();
      
      // release any items that we consumed in this firing
      if (tid < numInstances)
	{
	  COUNT_ITEMS(fireableCount);
	  //queue.release(tid, fireableCount);
	  queue.release(tid, 1);
	}


      } //end main if


      // make sure caller sees updated queue state
      __syncthreads();
      
      MOD_TIMER_STOP(gather);

      __syncthreads();

	//stimcheck: Only the boss thread needs to make the enumerate and aggregate signals
	
	if(threadIdx.x < numInstances) {
	unsigned int instIdx = threadIdx.x;
	//Create a new enum signal to send downstream
	Signal s;
	s.setTag(Signal::SignalTag::Enum);

	//Reserve space downstream for enum signal
	unsigned int dsSignalBase;
      	using Channel = typename BaseType::Channel<T>;
	
	//printf("\t\t\tNUM CHANNELS: %d\n", numChannels);
	for (unsigned int c = 0; c < numChannels; c++) {
		//const Channel *channel = static_cast<Channel *>(getChannel(c));
		//dsSignalBase[c] = channel->directSignalReserve(0, 1);
		//s.setCredit((channel->dsSignalQueueHasPending(tid)) ? channel->getNumItemsProduced(tid) : channel->dsPendingOccupancy(tid));

		  Channel *channel = 
		    static_cast<Channel *>(getChannel(c));
		//Set the credit for our new signal depending on if there are already signals downstream.
		if(channel->dsSignalQueueHasPending(instIdx)) {
			s.setCredit(channel->getNumItemsProduced(instIdx));
		}
		else {
			s.setCredit(channel->dsPendingOccupancy(instIdx));
		}

		//If the channel is NOT an aggregate channel, send a new enum signal downstream
		if(!(channel->isAggregate())) {
			dsSignalBase = channel->directSignalReserve(instIdx, 1);

			//Write enum signal to downstream node
			channel->directSignalWrite(instIdx, s, dsSignalBase, 0);
			channel->resetNumProduced(instIdx);
		}
	}
	}
	
	__syncthreads();

	//stimcheck: Decrement credit for the module here (if needed)
	if(tid < numInstances) {
		if(this->hasSignal[tid]) {
			//this->currentCredit[tid] -= fireableCount;
			this->currentCredit[tid] -= 1;
		}
	}
	__syncthreads();

	//if(IS_BOSS()) {
	//	printf("CALLING SIGNAL HANDLER ENUMERATE. . . \n");
	//}
	this->signalHandler();
	__syncthreads();
    }
  };  // end ModuleType class
}  // end Mercator namespace

#endif
