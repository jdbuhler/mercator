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
      : queue(numInstances, queueSizes)
    {
      // init channels array
      for(unsigned int c = 0; c < numChannels; ++c)
	channels[c] = nullptr;
      
#ifdef INSTRUMENT_OCC
      occCounter.setMaxRunSize(maxRunSize);
#endif

      //init activeFlag to always false in all nodes except source
      for(unsigned int j = 0; j < numInstances; ++j)
        activeFlag[j] = false;

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
    }
    
    //
    // @brief return our queue (needed for setDSEdge(), since downstream
    // module is not necessarily of same type as us).
    //
    __device__
    Queue<T> *getQueue()
    { return &queue; }

    ///////////////////////////////////////////////////////////////////
    //NEW INTERFACE FOR SCHEDULER_MINSWITCHES
    ///////////////////////////////////////////////////////////////////

    //called multithreaded
    __device__
    bool getActiveFlag(unsigned int instIdx)
    {
      assert(instIdx < numInstances);
      return activeFlag[instIdx];
    } 


    //called multithreaded
    __device__
    void flipActiveFlag(unsigned int instIdx)
    { 
      assert(instIdx < numInstances);
      activeFlag[instIdx] = !activeFlag[instIdx];
    } 

    //called multithread
    __device__
    bool computeIsFirable() const{
      int tid = threadIdx.x;
      //if this node is active and DS is inactive or were in tail
        //return true
      //else it is not firable
      return false;
    }
    
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
    computeNumFireable(unsigned int instIdx) const
    {
      assert(instIdx < numInstances);
      
      // start at max fireable, then discover bottleneck
      unsigned int numFireable = numInputsPending(instIdx);
      
      if (numFireable > 0)
	{
	  // for each channel
	  for (unsigned int c = 0; c < numChannels; ++c)
	    {
	      unsigned int dsCapacity = 
		channels[c]->dsCapacity(instIdx);
	      
	      numFireable = min(numFireable, dsCapacity);
	    }
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
      
      // The number of instances is <= the architectural warp size, so
      // just do the computation in a single warp, and then propagate
      // the final result out to all threads in the block.
      if (tid < WARP_SIZE)
	{
	  unsigned int numFireable = 0;
	  if (tid < numInstances)
	    numFireable = computeNumFireable(tid);
	  
	  unsigned int totalFireable;
	  
	  using Scan = WarpScan<unsigned int, WARP_SIZE>;
	  unsigned int sf = Scan::exclusiveSum(numFireable, totalFireable);
	  
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

    bool activeFlag[numInstances]; //is this node active 

    ChannelBase* channels[numChannels];  // module's output channels
    
    Queue<T> queue;                     // module's input queue
    
    // most recently computed count of # fireable in each instance
    unsigned int lastFireableCount[numInstances];
    
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

  };  // end ModuleType class
}  // end Mercator namespace

#include "Channel.cuh"

#endif
