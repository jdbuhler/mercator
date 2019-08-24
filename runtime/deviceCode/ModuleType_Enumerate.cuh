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
      : BaseType(queueSizes), parentBuffer(numInstances, queueSizes), refCounts(numInstances, queueSizes)
    {
	for(unsigned int i = 0; i < numInstances; ++i)
	{
		dataCount[i] = 0;
		currentCount[i] = 0;
		//noFireFlag[i] = false;
    		allChannelOutCapacity[numInstances] = UINT_MAX;
    		allChannelOutSignalCapacity[numInstances] = UINT_MAX;
	}
	refCount = 1;
    }
    
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

    //stimcheck: Add base type for numInputsPending to call "super" equivalent
    using BaseType::numInputsPending;
    using BaseType::numInputsPendingSignal;
    using BaseType::setParent;
    
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

    //stimcheck: Stores the amount of data we are still waiting on being processed before
    //removing the head data item.
    unsigned int dataCount[numInstances];
    unsigned int currentCount[numInstances];
    unsigned int allChannelOutCapacity[numInstances];
    unsigned int allChannelOutSignalCapacity[numInstances];
    //bool noFireFlag[numInstances];

    //stimcheck: TODO will be set in the codegen, not here, but for now only testing ref counts of 1.
    unsigned int refCount;

    Queue<T> parentBuffer; 
    Queue<unsigned int> refCounts;


    //
    // @brief Sets the currentCount and dataCount as necessary when calling
    // computeNumFireable from the Scheduler.  Returns the minimum of the input
    // numFireable calculated as normal (using dsCapacicties) and the
    // (re)-calculated dataCount - currentCount.
    //
    // The function is only to be called by a SINGLE-THREADED function, and
    // should be called for all instances of a module concurrently.
    //
    // @param instIdx instance for which to evaluate the dataCount for
    // @param numFireable the current number of fireable elements calculated from dsCapacities
    //
    // @return unsigned int the minimum of numFireable and the calculated dataCount
    //
    __device__
    //unsigned int
    void
    //setCounts(unsigned int instIdx, unsigned int numFireable)
    setCounts(unsigned int instIdx)
    {
	if(currentCount[instIdx] == dataCount[instIdx]) {
		if(parentBuffer.getCapacity(instIdx) - parentBuffer.getOccupancy(instIdx) > 0) {
			//stimcheck: ADD TO PARENT BUFFER HERE
      	  		Queue<T> &queue = this->queue; 
			unsigned int base = parentBuffer.reserve(instIdx, 1);
			unsigned int refbase = refCounts.reserve(instIdx, 1);
			unsigned int offset = 0;
	  		const T &elt = queue.getElt(instIdx, offset);
	  		const unsigned int &refelt = refCount;
			parentBuffer.putElt(instIdx, base, offset, elt);
			refCounts.putElt(instIdx, refbase, offset, refelt);

			//stimcheck: Set the current parent here once we have it in the parent buffer
			//setParent(static_cast<void**>(parentBuffer.getTail(instIdx)));
			void* s;
			//s = static_cast<void*>(*elt);
			//s = static_cast<void*>(parentBuffer.getTail(instIdx));
			s = parentBuffer.getVoidTail(instIdx);
			setParent(s, instIdx);
			//currentParent = static_cast<void*>(parentBuffer.getTail(instIdx)));


			currentCount[instIdx] = 0;
			dataCount[instIdx] = this->findCount(instIdx);
			printf("[%d] IN HERE\t\trefCounts[here] = %d\t\trefelt = %d\t\tcurrentParent = %p\n", blockIdx.x, refCounts.getElt(instIdx, offset), refelt, s);
		}
	}
	//assert(dataCount[instIdx] == 3);
	//if(dataCount[instIdx] != 3) {
	//	printf("[%d] DC NOT SET CORRECTLY\n", blockIdx.x);
	//}
	//printf("[%d] CD\n", blockIdx.x);

	//stimcheck: Set the noFireFlag for the spcial case where we need to fire off an element that has 0 data items
	//if(dataCount[instIdx] == 0)
	//{
	//	noFireFlag[instIdx] = true;
	//	return 1;
	//}

	//return min(numFireable, dataCount[instIdx] - currentCount[instIdx]);
    }

    __device__
    unsigned int
    getDC() {
	return dataCount[0];
    }

    __device__
    unsigned int 
    computeNumFireable(unsigned int instIdx, bool numPendingSignals)
    {
      assert(instIdx < numInstances);
      
      // start at max fireable, then discover bottleneck
      //unsigned int numFireable = numInputsPending(instIdx);
      unsigned int nf = numInputsPending(instIdx);
      //unsigned int numFireable = UINT_MAX;
      unsigned int numFireable = numInputsPending(instIdx);
      //bool blockOnSigQueue = false;
      allChannelOutCapacity[instIdx] = UINT_MAX;
      allChannelOutSignalCapacity[instIdx] = UINT_MAX;
      
	#if PF_DEBUG
	printf("[%d] NUM INPUTS PENDING = %d\n", blockIdx.x, nf);
	#endif
      if (nf > 0)
	{
	  // for each channel
	  for (unsigned int c = 0; c < numChannels; ++c)
	    {
	      unsigned int dsCapacity = 
		getChannel(c)->dsCapacity(instIdx);
	      unsigned int dsSignalCapacity =
		getChannel(c)->dsSignalCapacity(instIdx);
	      
		allChannelOutCapacity[instIdx] = min(allChannelOutCapacity[instIdx], dsCapacity);
		allChannelOutSignalCapacity[instIdx] = min(allChannelOutSignalCapacity[instIdx], dsSignalCapacity);

	      //Check the setting of the credit for the total number of fireable items
		//assert(numFireable >= this->currentCredit[instIdx]);
		//TODO
		//stimcheck: THIS SHOULD BE ==0, BUT CAUSES FAILURE IN SIGNAL QUEUE RESERVATION CURRENTLY.
		if(dsSignalCapacity == 1) {
		  numFireable = 0;
		  allChannelOutCapacity[instIdx] = 0;
		  allChannelOutSignalCapacity[instIdx] = 0;
		  //blockOnSigQueue = true;
		  #if PF_DEBUG
		  printf("[%d] SNO SPACE DOWNSTREAM\n", blockIdx.x);
		  #endif
		  break;
		}
	     	numFireable = min(numFireable, dsCapacity);
	    }

	#if PF_DEBUG
	if(nf > 0 && numFireable == 0) {
		printf("[%d] NO SPACE DOWNSTREAM\n", blockIdx.x);
	}
	#endif

	//stimcheck: Special case for sinks, since they do not have downstream channels, but still need to keep track of credit
	//for correct signal handling.
	/*
	if(numPendingSignals) {
		//stimcheck: We do not want the credit to hold back execution while there is at least 1
		//data element queued up.  We execute all the sub-elements of the current parent, so
		//checking the credit is only applicable when there are no data elements on the queue.
		if(this->currentCredit[instIdx] == 0) {
			printf("[%d] HERE\n", blockIdx.x);
			numFireable = min(numFireable, this->currentCredit[instIdx]);
			//noFireFlag[instIdx] = false;
		}
	}
	*/
	//stimcheck: Compute numFirable normally, then perform findCount as needed.
	//numFireable = BaseType::computeNumFireable(instIdx, numPendingSignals);

	//printf("BEFORE SET COUNTS: %d, numFireable %d, currentCount %d, dataCount %d\n", instIdx, numFireable, currentCount[instIdx], dataCount[instIdx]);
	if(nf > 0) {
		//numFireable = setCounts(instIdx, numFireable);
		setCounts(instIdx);
	}
	//printf("AFTER SET COUNTS: %d, numFireable %d, currentCount %d, dataCount %d\n", instIdx, numFireable, currentCount[instIdx], dataCount[instIdx]);
	#if PF_DEBUG
	if(numFireable > 0) {
		printf("[%d] NM FIREABLE = %d\t\tDC = %d\t\tCC = %d\t\tCREDIT = %d\t\tnumPending = %d\n", blockIdx.x, numFireable, dataCount[instIdx], currentCount[instIdx], this->currentCredit[instIdx], nf);
	}
	#endif

	//stimcheck: Set numFireable to 0 again if the dataCount is 0 but there is no space downstream for signals.
	//This allows the scheduler to then schedule firing of downstream signals rather than crashing on no space being left
	//from after this module.


	//stimcheck: Special case for sinks, since they do not have downstream channels, but still need to keep track of credit
	//for correct signal handling.
	if(numPendingSignals) {
		//stimcheck: We do not want the credit to hold back execution while there is at least 1
		//data element queued up.  We execute all the sub-elements of the current parent, so
		//checking the credit is only applicable when there are no data elements on the queue.
		//if(this->currentCredit[instIdx] == 0) {
		//	printf("[%d] HERE\n", blockIdx.x);
			numFireable = min(numFireable, this->currentCredit[instIdx]);
			//noFireFlag[instIdx] = false;
		//}
	}
        return numFireable;
      }
      return 0;
    }

    __device__
    virtual
    unsigned int findCount(InstTagT nodeIdx) {
	assert(false && "FindCount base called.");
	return 0;
    }


    //
    //
    //
    __device__
    void run(T inputItem, InstTagT nodeIdx)
    {
	  //DerivedModuleType *mod = static_cast<DerivedModuleType *>(this);
	  
	  //if (runWithAllThreads || idx < totalFireable)
	    //mod->findCount(nodeIdx);

	  //stimcheck: Push the the current set of indices that we can downstream.
	  //push(currentCount[nodeIdx] + threadIdx.x, nodeIdx);
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
	//(tid < numInstances && (this->hasSignal[tid] && this->currentCredit[tid] > 0) ? min(dataCount[tid] - currentCount[tid], allChannelOutCapacity[tid]) : 0);
	(tid < numInstances ? min(dataCount[tid] - currentCount[tid], allChannelOutCapacity[tid]) : 0);

	if(tid < numInstances) {
		if(this->hasSignal[tid]) {
			if(this->currentCredit[tid] == 0) {
				fireableCount = 0;
			}
		}
		else if(numInputsPendingSignal(tid) > 0) {
			fireableCount = 0;
		}
	}
	//(tid < numInstances ? getFireableCount(tid) : 0);

	__syncthreads();
      //if(tid < numInstances) {
	//if(noFireFlag[tid])
	//	fireableCount = 0;
      //}
      
      // compute progressive sums of items to be consumed in each instance,
      // and replicate these sums in each WARP as Ai.
      using Gather = QueueGather<numInstances>;

      unsigned int totalFireable;
      unsigned int Ai = Gather::loadExclSums(fireableCount, totalFireable);  

      bool sendEnumSignal = false;

      // release any items from 
	/*
      if (tid < numInstances)
	{
	  COUNT_ITEMS(fireableCount);
      	  Queue<T> &queue = this->queue; 
	  if(noFireFlag[tid]) {
		assert(queue.getOccupancy(tid) > 0);
		#if PF_DEBUG
		printf("MADE IT2\t\tCURRENT COUNT = %d\t\tDATA COUNT = %d\t\tCURRENT CREDIT = %d\n", currentCount[tid], dataCount[tid], this->currentCredit[tid]);
		#endif
		sendEnumSignal = true;
	  	queue.release(tid, 1);
	  }
	}
	*/

      //stimcheck:  If the scheduler determined that there were fireable data elements, fire them, otherwise fire no data, syncthreads, and process signals.
      if(totalFireable > 0) {

      assert(totalFireable > 0);
      
	#if PF_DEBUG
	if(fireableCount > 0 && tid < numInstances)
	printf("FIREABLE COUNT = %d, CURRENT COUNT = %d, DATA COUNT = %d, MAX RUN SIZE = %d, TOTAL FIREABLE = %d\n", fireableCount, currentCount[tid], dataCount[tid], maxRunSize, totalFireable);
	#endif

      MOD_OCC_COUNT(totalFireable);

      Queue<T> &queue = this->queue; 

      // Iterate over inputs to be run in block-sized chunks.
      // Do both gathering and execution of inputs in each iteration.
      // Every thread in a group receives the same input. 
      for (unsigned int base = 0;
	   base < totalFireable; 
	   base += maxRunSize)
	{
	//for(unsigned int inst = 0;
	//	inst < numInstances;
	//	++inst)
	//{
	
	  //unsigned int base = 0;	//Dummy var to replace old loop	
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
	  
	  __syncthreads();
	  DerivedModuleType *mod = static_cast<DerivedModuleType *>(this);
	  
	  //if (runWithAllThreads || idx < totalFireable)
	  //if(idx < dataCount[instIdx])
	  //for(unsigned int c = 0; c < numChannels; ++c) {
	  //	if(currentCount[inst] + threadIdx.x < dataCount[inst] && currentCount[inst] + idx < totalFireable)
	   // 	  mod->run(myData, inst);
	  	//stimcheck: We work on 1 instance at a time, so increment currentCount with single thread
	  //	if(IS_BOSS()) {
	//		currentCount[inst] += getChannel(c)->dsCapacity(instIdx);
	  //	}
	  //}

	  //if (runWithAllThreads || idx + base < totalFireable)
	//	mod->push(idx + base + currentCount[instIdx], instIdx);

	  if (runWithAllThreads || tid + base < totalFireable)
		mod->push(tid + base + currentCount[instIdx], instIdx);
	  
	  __syncthreads(); // all threads must see active channel state

	  //stimcheck: We work on 1 instance at a time, so increment currentCount with single thread
	  //if(IS_BOSS()) {
		//currentCount[inst] += 
	  //}

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

	  //stimcheck: Add the amount of indices we just pushed to the currentCount
	//  if(tid < numInstances) {
	//	currentCount[inst] += 
	//  }
	  
	  __syncthreads(); // all threads must see reset channel state
	  
	  MOD_TIMER_STOP(scatter);
	  MOD_TIMER_START(gather);
	} //end for
      
      // protect use of queue->getElt() from changes to head pointer due
      // to release.
      __syncthreads();

      if(tid < numInstances) {
	currentCount[tid] += fireableCount;
      }

      __syncthreads();
      
      // release any items that we consumed in this firing
      if (tid < numInstances)
	{
	  COUNT_ITEMS(fireableCount);
	  //queue.release(tid, fireableCount);
	  //if(currentCount[tid] == dataCount[tid]) {
	  //	queue.release(tid, 1);
	//	currentCount[tid] = 0;
	 // }
	  if(currentCount[tid] == dataCount[tid]) {
		#if PF_DEBUG
		printf("MADE IT\t\tCURRENT COUNT = %d\t\tDATA COUNT = %d\t\tCURRENT CREDIT = %d\n", currentCount[tid], dataCount[tid], this->currentCredit[tid]);
		#endif
		sendEnumSignal = true;
	  	queue.release(tid, 1);
	  }
	}

      } //end main if
      else if(tid < numInstances) {
	if(getFireableCount(tid) > 0) {
        Queue<T> &queue = this->queue; 
      	// release any items that we consumed in this firing
      	//if (tid < numInstances)
	//{
	  COUNT_ITEMS(fireableCount);
	  //queue.release(tid, fireableCount);
	  //if(currentCount[tid] == dataCount[tid]) {
	  //	queue.release(tid, 1);
	//	currentCount[tid] = 0;
	 // }
	  if(dataCount[tid] == 0) {
		printf("RELEASED HERE\n");
		#if PF_DEBUG
		printf("MADE IT\t\tCURRENT COUNT = %d\t\tDATA COUNT = %d\t\tCURRENT CREDIT = %d\n", currentCount[tid], dataCount[tid], this->currentCredit[tid]);
		#endif
		sendEnumSignal = true;
	  	queue.release(tid, 1);
	  }
	//}
	}
      }

      // make sure caller sees updated queue state
      __syncthreads();

      //stimcheck: Set any loose sendEnumSignals that may have been missed in execution (if totalFireable == 0)

      __syncthreads();
      
      MOD_TIMER_STOP(gather);

      __syncthreads();

	//stimcheck: Only the boss thread needs to make the enumerate and aggregate signals
	
	if(threadIdx.x < numInstances) {
	if(sendEnumSignal) {
	#if PF_DEBUG
	printf("PUSHING ENUMERATE SIGNAL . . .\n");
	#endif
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

		printf("[%d] BEFORE SET PARENT\t%p\n", blockIdx.x, s.getParent());
		s.setParent(&parentBuffer.getModifiableTail(instIdx));
		printf("[%d] AFTER SET PARENT\t%p\n", blockIdx.x, s.getParent());

		//If the channel is NOT an aggregate channel, send a new enum signal downstream
		if(!(channel->isAggregate())) {
			assert(channel->dsSignalCapacity(instIdx) > 0);
			dsSignalBase = channel->directSignalReserve(instIdx, 1);

			//Write enum signal to downstream node
			channel->directSignalWrite(instIdx, s, dsSignalBase, 0);
			channel->resetNumProduced(instIdx);
		}
	}
	}
	}
	
	__syncthreads();

	//stimcheck: Decrement credit for the module here (if needed)
	if(tid < numInstances) {
		if(this->hasSignal[tid] && sendEnumSignal) {
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
