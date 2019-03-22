#ifndef __MODULE_TYPE_CUH
#define __MODULE_TYPE_CUH

//
// @file ModuleType.cuh
// @brief a MERCATOR module type that knows its input type
//
// MERCATOR
// Copyright (C) 2018 Washington University in St. Louis; all rights reserved.
//

#include <cstdio>
#include <cassert>
#include <climits>

#include "ModuleTypeBase.cuh"

#include "Queue.cuh"

#include "device_config.cuh"

#include "options.cuh"

namespace Mercator  {
 
  //
  // @class ModuleTypeProperties
  // @brief properties of a MERCATOR module known at compile time
  //
  // @tparam T type of input to module
  // @tparam numInstances number of instances of module
  // @tparam numChannels  number of channels in module
  // @tparam numEltsPerGroup number of input elements/thread
  // @tparam threadGroupSize  number of threads in a thread group
  // @tparam maxActiveThreads max # of live threads in any call to run()
  // @tparam runWithAllThreads call run with all threads, or just as many
  //           as have inputs?
  //
  template <typename _T, 
	    unsigned int _numInstances, 
	    unsigned int _numChannels,
	    unsigned int _numEltsPerGroup,
	    unsigned int _threadGroupSize,
	    unsigned int _maxActiveThreads,
	    bool _runWithAllThreads,
	    unsigned int _THREADS_PER_BLOCK>
  struct ModuleTypeProperties {
    typedef _T T;
    static const unsigned int numInstances     = _numInstances;
    static const unsigned int numChannels      = _numChannels;
    static const unsigned int numEltsPerGroup  = _numEltsPerGroup;
    static const unsigned int threadGroupSize  = _threadGroupSize;
    static const unsigned int maxActiveThreads = _maxActiveThreads;
    static const bool runWithAllThreads        = _runWithAllThreads;
    static const unsigned int THREADS_PER_BLOCK= _THREADS_PER_BLOCK;  
  };
  
  
  //
  // @class ModuleType
  // @brief MERCATOR most general module type
  //
  // This class implements most of the interface in ModuleTypeBase,
  // but it leaves the fire() function to subclasses (and hence is
  // still pure virtual).
  //
  // @tparam Props properties structure for module
  //
  template<typename Props>
  class ModuleType : public ModuleTypeBase {
    
    using                                    T = typename Props::T;
    static const unsigned int numInstances     = Props::numInstances;
    static const unsigned int numChannels      = Props::numChannels;
    static const unsigned int numEltsPerGroup  = Props::numEltsPerGroup;
    static const unsigned int threadGroupSize  = Props::threadGroupSize;
    static const unsigned int maxActiveThreads = Props::maxActiveThreads;
    static const bool runWithAllThreads        = Props::runWithAllThreads;
    
    // actual maximum # of possible active threads in this block
    static const unsigned int deviceMaxActiveThreads =
      (maxActiveThreads > Props::THREADS_PER_BLOCK 
       ? Props::THREADS_PER_BLOCK 
       : maxActiveThreads);
    
    // number of thread groups (no partial groups allowed!)
    static const unsigned int numThreadGroups = 
      deviceMaxActiveThreads / threadGroupSize;
    
    // max # of active threads assumes we only run full groups
    static const unsigned int numActiveThreads =
      numThreadGroups * threadGroupSize;
    
  protected:
    
    // maximum number of inputs that can be processed in a single 
    // call to the module's run() function
    static const unsigned int maxRunSize =
      numThreadGroups * numEltsPerGroup;

    // forward-declare channel class
    
    class ChannelBase;
    
    template <typename T>
    class Channel;
    
  public:

    typedef uint8_t InstTagT;                  // instance tag type for run
    static const InstTagT NULLTAG = UCHAR_MAX; // null tag type
    
    //
    // @brief Constructor
    //
    // @param maxActiveThreads max allowed # of active threads per run
    // @param iqueueSizes array per node of requested queue sizes
    //
    __device__
    ModuleType(const unsigned int *queueSizes)
      : queue(numInstances, queueSizes), signalQueue(numInstances, queueSizes)
    {
      // init channels array
      for(unsigned int c = 0; c < numChannels; ++c)
	channels[c] = nullptr;
      
#ifdef INSTRUMENT_OCC
      occCounter.setMaxRunSize(maxRunSize);
#endif
    }
    
    
    //
    // @brief Destructor
    //
    __device__
    virtual
    ~ModuleType()
    {
      for (unsigned int c = 0; c < numChannels; ++c)
	{
	  ChannelBase *channel = channels[c];
	  if (channel)
	    delete channel;
	}
    }
    
    
    //
    // @brief Create and initialize an output channel.
    //
    // @param c index of channel to initialize
    //
    // @param outputsPerInput Num outputs/input for the channel
    // @param reservedSlots array of reserved slot counts for each 
    //          inst's downstream queue
    //
    template<typename DST>
    __device__
    void initChannel(unsigned int c, 
		     unsigned int outputsPerInput,
		     const unsigned int *reservedSlots)
    {
      assert(c < numChannels);
      assert(outputsPerInput > 0);
      
      // init the output channel -- should only happen once!
      assert(channels[c] == nullptr);
      
      channels[c] = new Channel<DST>(outputsPerInput, 
				     reservedSlots);
      
      // make sure alloc succeeded
      if (channels[c] == nullptr)
	{
	  printf("ERROR: failed to allocate channel object [block %d]\n",
		 blockIdx.x);
	  
	  crash();
	}
    }
    
    
    // stimcheck: Add setting of the downstream Signal edge
    //
    // @brief Set the edge from channel channelIdx of node usInstIdx of
    // this module type to point to node dsInstIdx of module type dsModule.
    // We specify the downstream module's queue, which is all that the
    // channel needs to know.
    //
    // @param channelIdx upstream channel
    // @param usInstIdx upstream node
    // @param dsModule downstream module
    // @param dsInstIdx downstream node
    //
    template <typename DSP>
    __device__
    void setDSEdge(unsigned int channelIdx,
		   unsigned int usInstIdx,
		   ModuleType<DSP> *dsModule, 
		   unsigned int dsInstIdx)
    { 
      Channel<typename DSP::T> *channel = 
	static_cast<Channel<typename DSP::T> *>(channels[channelIdx]);
      
      channel->setDSEdge(usInstIdx, dsModule->getQueue(), dsInstIdx);
      channel->setDSSignalEdge(usInstIdx, dsModule->getSignalQueue(), dsInstIdx);
    }
    
    //
    // @brief return our queue (needed for setDSEdge(), since downstream
    // module is not necessarily of same type as us).
    //
    __device__
    Queue<T> *getQueue()
    { return &queue; }

    // stimcheck: Return the Signal Queue of this module
    //
    // @brief return our queue (needed for setDSEdge(), since downstream
    // module is not necessarily of same type as us).
    //
    __device__
    Queue<Signal> *getSignalQueue()
    { return &signalQueue; }
    
    ///////////////////////////////////////////////////////////////////
    // FIREABLE COUNTS FOR SCHEDULING
    ///////////////////////////////////////////////////////////////////
    
    
    //
    // @brief Compute the number of inputs that can safely be consumed
    // from one instance's input queue in one firing of this module.
    // Of course, we cannot consume more inputs than are actually
    // queued, but we are also limited by the amount of free space in
    // each channel's downstream queue.
    //
    // This function is SINGLE-THREADED and may be called for all
    // instances of a module concurrently.
    //
    // @param instIdx instance for which to compute fireable count.
    //
    __device__
    unsigned int 
    //computeNumFireable(unsigned int instIdx, unsigned int numPendingSignals) const
    computeNumFireable(unsigned int instIdx, bool numPendingSignals) const
    {
      assert(instIdx < numInstances);
      
      // start at max fireable, then discover bottleneck
      unsigned int numFireable = numInputsPending(instIdx);
      //bool madeIt = false;

      //printf("NUM PEND SIG = %d\n", numPendingSignals);
      
      //if (numFireable > 0)
	//{
	  // for each channel
	  for (unsigned int c = 0; c < numChannels; ++c)
	    {
	      unsigned int dsCapacity = 
		channels[c]->dsCapacity(instIdx);
	      unsigned int dsSignalCapacity =
		channels[c]->dsSignalCapacity(instIdx);
	      
	      //Check the setting of the credit for the total number of fireable items
		//printf((numPendingSignals > 0 ? "EARLY %d\t" : "LATE %d\t"), numPendingSignals);
		//printf("numPendingSignals = %d, fn = %d\n", numPendingSignals, numInputsPendingSignal(instIdx));
	      //if(numInputsPendingSignal(instIdx) > 0) {
	      //if(numPendingSignals > 0) {
	      if(numPendingSignals) {
	      //if(hasSignal[instIdx]) {
		//madeIt = true;
		printf("[%d, %d] NUM FIREABLE = %d, CURRENT CREDIT = %d\n", blockIdx.x, instIdx, numFireable, this->currentCredit[instIdx]);
		assert(numFireable >= this->currentCredit[instIdx]);
	      	numFireable = min(numFireable, dsCapacity);
	      	numFireable = min(numFireable, this->currentCredit[instIdx]);
		//numFireable = this->currentCredit[instIdx];
		printf("MADE IT [%d, %d] cc = %d, dsCap = %d\t", instIdx, blockIdx.x, numFireable, dsCapacity);
	      }
	      else {
		 assert(this->currentCredit[instIdx] == 0);
		 assert(numPendingSignals == 0);
	     	 numFireable = min(numFireable, dsCapacity);
		 printf("IT MADE[%d, %d] cc = %d\t", instIdx, blockIdx.x, numFireable);
	      }

	      //No downstream signal space available and signal available in queue, cannot proceed
	      //if(dsSignalCapacity == 0 && numPendingSignals) {
		//numFireable = 0;
	     // }
	    }
	//}

	//stimcheck: Special case for sinks, since they do not have downstream channels, but still need to keep track of credit
	//for correct signal handling.
	if(numChannels == 0 && numPendingSignals) {
		numFireable = min(numFireable, this->currentCredit[instIdx]);
	}
      
      if(numPendingSignals) {
	//printf("NUM PENDING SIGNALS = %d\n", numPendingSignals);
	assert(this->currentCredit[instIdx] <= numFireable);
	//if(this->currentCredit[instIdx] <= numFireable && !madeIt) {
	//	printf("BIG PROBLEM %d\n", numChannels);
	//}
      }
      return numFireable;
    }
    

    //
    // @brief Compute the number of inputs that can safely be consumed
    // in one firing of all instances of this module.  We sum 
    // the fireable counts from each instance and then round down to
    // a full number of run ensembles.
    //
    // In addition to computing the result, cache the per-instance
    // fireable counts computed here, so that they can be looked up
    // when we actually fire.
    //
    // This function must be called with ALL THREADS in the block.
    //
    // @param instIdx instance for which to compute fireable count.
    //
    __device__
    unsigned int 
    computeNumFireableTotal(bool enforceFullEnsembles)
    {
      int tid = threadIdx.x;

      __shared__ unsigned int tf;
      __shared__ bool hasPendingSignal;  //This doesn't need to be shared, right?

      //bool nhc = !(hasCredit());
      unsigned int numFireable = 0;
      unsigned int numInputsPending = 0;
      __syncthreads(); //

      if(tid < numInstances && tid < WARP_SIZE) {
      	numInputsPending = numInputsPendingSignal(tid);
      }

      __syncthreads(); //

      if(tid < numInstances && tid < WARP_SIZE) {
	if(numInputsPending > 0) {
	  hasPendingSignal = true;
	}
      }

      //if(tid < WARP_SIZE) {
      //	printf("PENDING SIGNALS[tid < WARP_SIZE %d] = %d\t", (tid < WARP_SIZE ? 1 : 0),  numInputsPending);
      //}

      __syncthreads(); //
      
      // The number of instances is <= the architectural warp size, so
      // just do the computation in a single warp, and then propagate
      // the final result out to all threads in the block.
      if (tid < WARP_SIZE)
	{
	  //unsigned int numFireable = 0;
	  if (tid < numInstances)
	    //numFireable = computeNumFireable(tid, numInputsPending);
	    numFireable = computeNumFireable(tid, hasPendingSignal);
	}

      __syncthreads(); //

      if(tid < WARP_SIZE) {
	  if(numFireable > 0) {
	  	printf("NUM FIREABLE[%d, %d] = %d\n", tid, blockIdx.x, numFireable);
	  }
        }

      __syncthreads(); //

      if (tid < WARP_SIZE) {
	  unsigned int totalFireable;
	  
	  using Scan = WarpScan<unsigned int, WARP_SIZE>;
	  unsigned int sf = Scan::exclusiveSum(numFireable, totalFireable);
	  
	  //if (enforceFullEnsembles && !hasCredit())
	  //if (enforceFullEnsembles)
	  //if (enforceFullEnsembles && numInputsPending == 0)
	    if (enforceFullEnsembles && !hasPendingSignal)
	    {
		//if(IS_BOSS()) {
		//	printf("ENFORCING FULL ENSEMBLES hasPendingSignal = %d\n", (hasPendingSignal ? 1 : 0));
		//}
	      // round total fireable count down to a full multiple of # of
	      // elements that can be consumed by one call to run()
	      totalFireable = 
		(totalFireable / maxRunSize) * maxRunSize;
	      
	      // adjust individual fireable counts to match reduced total 
	      if (numFireable + sf > totalFireable)
		numFireable = (sf > totalFireable ? 0 : totalFireable - sf);
	    }
	/*
	  else 
		if(IS_BOSS()) {
			printf("Computed %d items to fire\n", totalFireable);
		}
	  }
	*/
	  
	  // cache results of per-instance fireable calculation for later
	  // use by module's firing function
	  if (tid < numInstances)
	    lastFireableCount[tid] = numFireable;
	  
	  if (tid == 0)
	    tf = totalFireable;
	}
      
      __syncthreads();
      
      return tf;
    }

    //stimcheck: Function for checking if we have credit currently in the module anywhere
    __device__
    bool
    hasCredit() {
	//bool c = false;
	//if(IS_BOSS()) {
		for(unsigned int i = 0; i < numInstances; ++i) {
			if(currentCredit[i] > 0) {
				return true;
			}
		}
	//}
	//printf("HERE %d\t", (c ? 1 : 0));
	//__syncthreads();
	//return c;
	return false;
    }
    
    
    __device__
    unsigned int
    getTotalCredit() {
	__shared__ unsigned int c;
	if(IS_BOSS()) {
		c = 0;
		for(unsigned int i = 0; i < numInstances; ++i) {
			c += currentCredit[i];
		}
	}
	//printf("HERE %d\t", (c ? 1 : 0));
	__syncthreads();
	return c;
    }
    
    //stimcheck:  Compute the number of fireable signals, used for clearing queues that still have signals
    //
    // @brief Compute the number of inputs that can safely be consumed
    // from one instance's input queue in one firing of this module.
    // Of course, we cannot consume more inputs than are actually
    // queued, but we are also limited by the amount of free space in
    // each channel's downstream queue.
    //
    // This function is SINGLE-THREADED and may be called for all
    // instances of a module concurrently.
    //
    // @param instIdx instance for which to compute fireable count.
    //
    __device__
    unsigned int 
    computeNumSignalFireable(unsigned int instIdx) const
    {
      assert(instIdx < numInstances);
      
      // start at max fireable, then discover bottleneck
      unsigned int numFireable = numInputsPendingSignal(instIdx);
      
      if (numFireable > 0)
	{
	  // for each channel
	  for (unsigned int c = 0; c < numChannels; ++c)
	    {
	      unsigned int dsCapacity = 
		channels[c]->dsSignalCapacity(instIdx);
	      
	      numFireable = min(numFireable, dsCapacity);
	    }
	}
      
      return numFireable;
    }
    
    //stimcheck:  Compute num fireable signals, used for clearing out signal queues before next app run
    //
    // @brief Compute the number of inputs that can safely be consumed
    // in one firing of all instances of this module.  We sum 
    // the fireable counts from each instance and then round down to
    // a full number of run ensembles.
    //
    // In addition to computing the result, cache the per-instance
    // fireable counts computed here, so that they can be looked up
    // when we actually fire.
    //
    // This function must be called with ALL THREADS in the block.
    //
    // @param instIdx instance for which to compute fireable count.
    //
    __device__
    unsigned int 
    computeNumSignalFireableTotal(bool enforceFullEnsembles)
    {
      //enforceFullEnsembles = false;
      int tid = threadIdx.x;
      
      __shared__ unsigned int tf;
      
      // The number of instances is <= the architectural warp size, so
      // just do the computation in a single warp, and then propagate
      // the final result out to all threads in the block.
      if (tid < WARP_SIZE)
	{
	  unsigned int numFireable = 0;
	  if (tid < numInstances)
	    numFireable = computeNumSignalFireable(tid);
	  
	  unsigned int totalFireable;
	  
	  using Scan = WarpScan<unsigned int, WARP_SIZE>;
	  unsigned int sf = Scan::exclusiveSum(numFireable, totalFireable);
	  
	//stimcheck: No reason to ever enforce full ensembles for signals currently
	/*
	  if (enforceFullEnsembles)
	    {
	      // round total fireable count down to a full multiple of # of
	      // elements that can be consumed by one call to run()
	      totalFireable = 
		(totalFireable / maxRunSize) * maxRunSize;
	      
	      // adjust individual fireable counts to match reduced total 
	      if (numFireable + sf > totalFireable)
		numFireable = (sf > totalFireable ? 0 : totalFireable - sf);
	    }
	  */
	  // cache results of per-instance fireable calculation for later
	  // use by module's firing function
	  //stimcheck: DO NOT OVERWRITE CACHED FIREABLE COUNT HERE.  This refers to DATA not SIGNALS.
	  //if (tid < numInstances)
	    //lastFireableCount[tid] = numFireable;
	  
	  if (tid == 0)
	    tf = totalFireable;
	}
      
      __syncthreads();
      //printf("TOTAL SIGNALS FIREABLE FOUND: %d\n", tf);
      return tf;
    }
    
    //
    // @brief compute the total number of inputs queued in the input
    // queues for all instances of this module.
    //
    __device__
    unsigned int 
    computeNumPendingTotal() const
    {
      int tid = threadIdx.x;
      
      __shared__ unsigned int tp;
      
      // number of instances is <= architectural warp size, so just do
      // the computation in a single warp, and then propagate the
      // final result out to all threads in the block.
      if (tid < WARP_SIZE)
	{
	  unsigned int numPending = 0;
	  if (tid < numInstances)
	    numPending = numInputsPending(tid);
	 
	  using Sum = WarpReduce<unsigned int, WARP_SIZE>;
	  unsigned int totalPending = Sum::sum(numPending, numInstances);
	  
	  if (tid == 0)
	    tp = totalPending;
	}
      
      __syncthreads();
      
      return tp;
    }

    //stimcheck:  Signal debug equivalent
    //
    // @brief compute the total number of inputs queued in the input
    // queues for all instances of this module.
    //
    __device__
    unsigned int 
    computeNumPendingTotalSignal() const
    {
      int tid = threadIdx.x;
      
      __shared__ unsigned int tp;
      
      // number of instances is <= architectural warp size, so just do
      // the computation in a single warp, and then propagate the
      // final result out to all threads in the block.
      if (tid < WARP_SIZE)
	{
	  unsigned int numPending = 0;
	  if (tid < numInstances)
	    numPending = numInputsPendingSignal(tid);
	 
	  using Sum = WarpReduce<unsigned int, WARP_SIZE>;
	  unsigned int totalPending = Sum::sum(numPending, numInstances);
	  
	  if (tid == 0)
	    tp = totalPending;
	}
      
      __syncthreads();
      
      return tp;
    }
        
    ///////////////////////////////////////////////////////////////////
    // OUTPUT CODE FOR INSTRUMENTATION
    ///////////////////////////////////////////////////////////////////
    
#ifdef INSTRUMENT_TIME
    //
    // @brief print the contents of the module's timers
    // @param moduleId a numerical identifier to print along with the
    //    output
    //
    __device__
    virtual
    void printTimersCSV(unsigned int moduleId) const
    {
      assert(IS_BOSS());
      
      DeviceTimer::DevClockT gatherTime  = gatherTimer.getTotalTime();
      DeviceTimer::DevClockT runTime     = runTimer.getTotalTime();
      DeviceTimer::DevClockT scatterTime = scatterTimer.getTotalTime();
      
      printf("%d,%u,%llu,%llu,%llu\n",
	     blockIdx.x, moduleId, gatherTime, runTime, scatterTime);
    }

#endif

#ifdef INSTRUMENT_OCC
    //
    // @brief print the contents of the module's occupancy counter
    // @param moduleId a numerical identifier to print along with the
    //    output
    //
    __device__
    virtual
    void printOccupancyCSV(unsigned int moduleId) const
    {
      assert(IS_BOSS());
      printf("%d,%u,%u,%llu,%llu,%llu\n",
	     blockIdx.x, moduleId,
	     occCounter.sizePerRun,
	     occCounter.totalInputs,
	     occCounter.totalRuns,
	     occCounter.totalFullRuns);
    }
#endif

#ifdef INSTRUMENT_COUNTS
    //
    // @brief print the contents of the module's item counters
    // @param moduleId a module identifier to print along with the
    //    output
    //
    __device__
    virtual
    void printCountsCSV(unsigned int moduleId) const
    {
      assert(IS_BOSS());
      
      printCountsSingle(itemCounter, moduleId, -1);
      
      for (unsigned int c = 0; c < numChannels; c++)
	printCountsSingle(channels[c]->itemCounter, moduleId, c);
    }
    
    //
    // @brief print the contents of one item counter
    // @param counter the counter to print
    // @param moduleId a module identifier to print along with the
    //         output
    // @param channelId a channel identifier to print along with the 
    //         output
    //
    __device__
    void printCountsSingle(const ItemCounter<numInstances> &counter,
			   unsigned int moduleId, int channelId) const
    {
      // print counts for every instance of this channel
      for (unsigned int i = 0; i < numInstances; i++)
	printf("%d,%u,%d,%u,%llu\n",
	       blockIdx.x, moduleId, channelId, i, counter.counts[i]);
    }
    
#endif
    
  protected:

    ChannelBase* channels[numChannels];  // module's output channels
    
    Queue<T> queue;                     // module's input queue
    Queue<Signal> signalQueue;                     // module's input queue
    
    // most recently computed count of # fireable in each instance
    unsigned int lastFireableCount[numInstances];
    
    // stimcheck: Current credit available to each node
    int currentCredit[numInstances];
    bool hasSignal[numInstances];

    // stimcheck: Current number of items the module has worked on (up till now)
    // This is used to determine the amount of credit we need to give signals that
    // are produced by the module (if any).  Can basically be ignored if we don't
    // have any signals that require this knowledge.  This also allows us to not
    // have to modify the Signals in the signal queue when decrementing credit.
    unsigned int numDataProduced[numInstances];

#ifdef INSTRUMENT_TIME
    DeviceTimer gatherTimer;
    DeviceTimer runTimer;
    DeviceTimer scatterTimer;
#endif
    
#ifdef INSTRUMENT_OCC
    OccCounter occCounter;
#endif

#ifdef INSTRUMENT_COUNTS
    ItemCounter<numInstances> itemCounter; // counts inputs to module
#endif
    
    //
    // @brief inspector for the channels array (for subclasses)
    // @param c index of channel to get
    //
    __device__
    ChannelBase *getChannel(unsigned int c) const 
    { 
      assert(c < numChannels);
      return channels[c]; 
    }

    //
    // @brief number of inputs currently enqueued for a particular
    // instance of this module.
    //
    // @param instIdx index for which to check pending count
    //
    __device__
    virtual
    unsigned int numInputsPending(unsigned int instIdx) const
    {
      assert(instIdx < numInstances);
      
      return queue.getOccupancy(instIdx);
    }

    //stimcheck:  Signal debug equivalent
    //
    // @brief number of inputs currently enqueued for a particular
    // instance of this module.
    //
    // @param instIdx index for which to check pending count
    //
    __device__
    virtual
    unsigned int numInputsPendingSignal(unsigned int instIdx) const
    {
      assert(instIdx < numInstances);
      //printf("IN HERE\t%d\t", signalQueue.getOccupancy(instIdx));
      return signalQueue.getOccupancy(instIdx);
    }
    
    //
    // @brief maximum number of inputs that could ever be enqueued for
    // this module, over all its instances, at one time.
    //
    __device__
    virtual
    unsigned int maxPending() const
    { return queue.getTotalCapacity(); }
    
    
    //
    // @brief get last cached count of number of inputs fireable for
    // a particular instance of this module
    //
    __device__
    unsigned int getFireableCount(unsigned int instIdx) const
    { 
      assert(instIdx < numInstances);
      
      return (lastFireableCount[instIdx]);
    }

    //stimcheck: Signal handler function, preforms actions based on signals
    //that can currently be processed.
    __device__
    void signalHandler() {
	//printf("%d\n", blockIdx.x);
	//Perform actions based on signals in each instance's signal queue
	unsigned int tid = threadIdx.x;
	unsigned int instIdx = tid;
	unsigned int i = 0;
	unsigned int sigQueueOcc = 0;
	Signal s;
	__syncthreads();

	//testing
	if(tid < numInstances) {
		if(signalQueue.getOccupancy(tid) > 0) {
			//signalQueue.getModifiableHead(tid).setStart(true);
			//signalQueue.getModifiableTail(tid).setEnd(true);
		}
	}
	__syncthreads();

	if(instIdx < numInstances) {
		sigQueueOcc = signalQueue.getOccupancy(instIdx);
	}

	__syncthreads();

	if(instIdx < numInstances) {
		//printf("Signal Queue Occupancy: %d\n", signalQueue.getOccupancy(instIdx));
		//if(signalQueue.getOccupancy(instIdx) > 0) {
		//	printf("Signal Queue Occupancy: %d\n", signalQueue.getOccupancy(instIdx));
		//}
		//s = signalQueue.getElt(instIdx, i);
		//while(s != nullptr) {
		//while(&(s) != nullptr && i < 3) {
		//while(signalQueue.getOccupancy(instIdx) && i < 3) {

		//unsigned int numBefore[numInstances];

		//while(signalQueue.getOccupancy(instIdx) > i) {
		while(sigQueueOcc > i) {
			s = signalQueue.getElt(instIdx, i);
			//Base case: we have credit to wait on

			assert(currentCredit[instIdx] >= 0);

			//Currently must be true, since we are only working with signals that use the current downstream space
			if(currentCredit[instIdx] > 0) {
				assert(currentCredit[instIdx] <= queue.getOccupancy(instIdx));
			}

			//Signal has Credit
			// AND
			//Signal credit has not already been taken
			if(s.getCredit() > 0 && !(hasSignal[instIdx])) {
				//printf("Credit Signal Processed\n");
				currentCredit[instIdx] = s.getCredit();
				hasSignal[instIdx] = true;
				printf("\nCredit Signal Processed\ncurrentCredit[%d, %d]: %d\n\n", instIdx, blockIdx.x, currentCredit[instIdx]);
			}

			//Base case: we have credit to wait on
			//If the current credit has reached 0, then we can consume signal
			if(currentCredit[instIdx] > 0) {
				printf("currentCredit[%d, %d]: %d\t\tqueueContents = %d\n", instIdx, blockIdx.x, currentCredit[instIdx], queue.getOccupancy(instIdx));
				break;
			}
			else {
				hasSignal[instIdx] = false;
			}

			assert(!(hasSignal[instIdx]));

			///////////////////
			//Signal type cases
			///////////////////

			//Start Signal
			if(s.getStart()) {
				printf("Start Signal Processed\n");
			}

			//End Signal
			if(s.getEnd()) {
				printf("End Signal Processed\n");
			}

			//Enumerate Signal
			if(s.getEnum()) {
				//Actual enumeration functionality happens in the module,
				//nothing needs to be set by the signal handler, other than
				//to begin a new aggregate if needed, otherwise continue
				//to pass the signal downstream.
				if(!(this->isAgg())) {		
					//Create a new tail signal to send downstream
					Signal s;
					s.setEnum(true);
					//s.setCredit(

					//Reserve space downstream for tail signal
					unsigned int dsSignalBase;
	        			for (unsigned int c = 0; c < numChannels; c++) {
						const Channel<int> *channel = static_cast<Channel<int> *>(getChannel(c));
						//dsSignalBase[c] = channel->directSignalReserve(0, 1);
						dsSignalBase = channel->directSignalReserve(instIdx, 1);

						//Write tail signal to downstream node
						channel->directSignalWrite(instIdx, s, dsSignalBase, 0);
					}
				}
				printf("Enumerate Signal Processed\n");
			}

			//Aggregate Signal
			if(s.getAgg()) {
				printf("Aggregate Signal Processed\n");
			}

			//Tail Signal
			if(s.getTail()) {
				//printf("Tail Signal Processed\n");
				//using Channel = typename BaseType::Channel<int>;

				//////////////////////////////////////////////////////////////this->setInTail(true);

				//Create a new tail signal to send downstream
				Signal s;
				s.setTail(true);	

				//Reserve space downstream for tail signal
				unsigned int dsSignalBase;
	        		for (unsigned int c = 0; c < numChannels; c++) {
					const Channel<int> *channel = static_cast<Channel<int> *>(getChannel(c));
					//dsSignalBase[c] = channel->directSignalReserve(0, 1);
					//s.setCredit((channel->dsSignalQueueHasPending(tid)) ? channel->getNumItemsProduced(tid) : channel->dsPendingOccupancy(tid));
					//printf("pending occupancy: %d\n", channel->dsPendingOccupancy(tid));
					if(channel->dsSignalQueueHasPending(instIdx)) {
						s.setCredit(channel->getNumItemsProduced(instIdx));
					}
					else {
						s.setCredit(channel->dsPendingOccupancy(instIdx));
						
						//s.setCredit(0);
						printf("New Tail Signal Propagated[%d, %d]\tpending occupancy: %d, Capacity: %d\n", instIdx, blockIdx.x, channel->dsPendingOccupancy(instIdx), channel->dsCapacity(instIdx));
					}
					dsSignalBase = channel->directSignalReserve(instIdx, 1);

					//Write tail signal to downstream node
					channel->directSignalWrite(instIdx, s, dsSignalBase, 0);

					//printf("New Tail Signal Propagated[%d, %d]: Credit = %d\n", instIdx, blockIdx.x, channel->dsPendingOccupancy(instIdx));
				}
			}

			//Reset number of items produced if we have processed a signal
			if(i == 0) {
				for(unsigned int c = 0; c < numChannels; c++) {
					Channel<int> *channel = static_cast<Channel<int> *>(getChannel(c));
					channel->resetNumProduced(instIdx);
				}
			}

			//Increment counter to next index in signal queue
			++i;
			//s = signalQueue.getElt(instIdx, i);
		}
	}
	__syncthreads();	//Bring all threads together at end
	
	//Release all signals that were processed
	if(instIdx < numInstances && i > 0) {
		signalQueue.release(instIdx, i);
	}

	__syncthreads();
    }

    __device__
    void decrementCredit(unsigned int c, unsigned int instIdx) {
	if(instIdx < numInstances)
		currentCredit[instIdx] -= c;
    }
    
    
    ///////////////////////////////////////////////////////////////////
    // RUN-FACING FUNCTIONS 
    // These functions expose documented properties and behavior of the 
    // module to the user's run(), init(), and cleanup() functions.
    ///////////////////////////////////////////////////////////////////
    
    //
    // @brief get the number of instances
    //
    __device__
    unsigned int getNumInstances() const
    { return numInstances; }

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
    // @param instTag tag of node that is writing item
    // @param channelIdx channel to which to write the item
    //
    template<typename DST>
    __device__
    void push(const DST &item, 
	      InstTagT instTag, 
	      unsigned int channelIdx = 0) const
    {
      Channel<DST>* channel = 
	static_cast<Channel<DST> *>(channels[channelIdx]);
      
      channel->push(item, isThreadGroupLeader());
    }

    // stimcheck: Push Signal to downstream channel
    // Uses void* for channel type (Since the type does not matter)
    //
    // @brief Write an output signal to the indicated channel.
    //
    // @param item Item to be written
    // @param instTag tag of node that is writing item
    // @param channelIdx channel to which to write the item
    //
    __device__
    void pushSignal(const Signal &item, 
	      InstTagT instTag, 
	      unsigned int channelIdx = 0) const
    {
      Channel<void*>* channel = 
	static_cast<Channel<void*> *>(channels[channelIdx]);
      
      channel->pushSignal(item, isThreadGroupLeader());
    }
  };  // end ModuleType class
}  // end Mercator namespace

#include "Channel.cuh"

#endif
