#ifndef __NODE_CUH
#define __NODE_CUH

//
// @file Node.cuh
// @brief a MERCATOR node that knows its input type
//
// MERCATOR
// Copyright (C) 2020 Washington University in St. Louis; all rights reserved.
//

#include <cstdio>
#include <cassert>

#include "NodeBase.cuh"

#include "Queue.cuh"

#include "Signal.cuh"

#include "device_config.cuh"

#include "options.cuh"

namespace Mercator  {

  //
  // @class NodeProperties
  // @brief properties of a MERCATOR node known at compile time
  //
  // @tparam T type of input
  // @tparam numChannels  number of channels
  // @tparam numEltsPerGroup number of input elements/thread
  // @tparam threadGroupSize  number of threads in a thread group
  // @tparam maxActiveThreads max # of live threads in any call to run()
  // @tparam runWithAllThreads call run() with all threads, or just as many
  //           as have inputs?
  //
  template <typename _T, 
	    unsigned int _numChannels,
	    unsigned int _numEltsPerGroup,
	    unsigned int _threadGroupSize,
	    unsigned int _maxActiveThreads,
	    bool _runWithAllThreads,
	    unsigned int _THREADS_PER_BLOCK>
  struct NodeProperties {
    typedef _T T;
    static const unsigned int numChannels      = _numChannels;
    static const unsigned int numEltsPerGroup  = _numEltsPerGroup;
    static const unsigned int threadGroupSize  = _threadGroupSize;
    static const unsigned int maxActiveThreads = _maxActiveThreads;
    static const bool runWithAllThreads        = _runWithAllThreads;
    static const unsigned int THREADS_PER_BLOCK= _THREADS_PER_BLOCK;  
  };


  //
  // @class Node
  // @brief MERCATOR most general node type
  //
  // This class implements most of the interface in NodeBase,
  // but it leaves the fire() function to subclasses (and hence is
  // still pure virtual).
  //
  // @tparam Props properties structure for node
  //
  template<typename Props>
  class Node : public NodeBase {

    using                                    T = typename Props::T;
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
    // call to the node's run() function
    static const unsigned int maxRunSize =
      numThreadGroups * numEltsPerGroup;

    // forward-declare channel class

    class ChannelBase;
    
    template <typename T>
    class Channel;

  public:

    //
    // @brief Constructor
    //
    // @param maxActiveThreads max allowed # of active threads per run
    // @param queueSize requested size for node's input queue
    //
    __device__
    Node(const unsigned int queueSize,
	 Scheduler *scheduler, unsigned int region)
      : NodeBase(scheduler, region),
	queue(queueSize),
        signalQueue(queueSize) // could be smaller?
    {
      // init channels array
      for (unsigned int c = 0; c < numChannels; ++c)
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
    ~Node()
    {
      for (unsigned int c = 0; c < numChannels; ++c)
	{
	  ChannelBase *channel = channels[c];
	  if (channel)
	    delete channel;
	}
    }



    //
    // @brief Construct the edge between this node and a downstream
    // neighbor on a particular channel.
    //
    // @param channelIdx channel that holds edge
    // @param dsNode node at downstream end of edge
    // @param reservedSlots reserved slot count for edge's queue
    //
    template <typename DSP>
    __device__
    void setDSEdge(unsigned int channelIdx,
		   Node<DSP> *dsNode,
		   unsigned int reservedSlots) 
    { 
      Channel<typename DSP::T> *channel = 
	static_cast<Channel<typename DSP::T> *>(channels[channelIdx]);

      dsNode->setParentNode(this);
      channel->setDSEdge(dsNode, 
			 dsNode->getQueue(), 
			 dsNode->getSignalQueue(),
			 reservedSlots);
    }
    
    //
    // @brief return our queue (needed for setDSEdge()).
    //
    __device__
    Queue<T> *getQueue()
    { 
      return &queue; 
    }
    
    //
    // @brief return our signal queue (needed for setDSEdge()).
    //
    __device__
    Queue<Signal> *getSignalQueue()
    { 
      return &signalQueue; 
    }
    
    //
    // @brief is any input queued for this node?
    // (Only used for debugging.)
    //
    __device__
    bool hasPending() const
    {
      return (!queue.empty() || !signalQueue.empty());
    }

  //Begin and end stubs for enumeration and aggregation 
  public: 
    __device__
    virtual
    void begin() {}

    __device__
    virtual
    void end() {}
    
    ///////////////////////////////////////////////////////////////////
    // OUTPUT CODE FOR INSTRUMENTATION
    ///////////////////////////////////////////////////////////////////
    
#ifdef INSTRUMENT_TIME
    //
    // @brief print the contents of the node's timers
    // @param nodeId a numerical identifier to print along with the
    //    output
    //
    __device__
    void printTimersCSV(unsigned int nodeId) const
    {
      assert(IS_BOSS());
    
      DeviceTimer::DevClockT inputTime  = inputTimer.getTotalTime();
      DeviceTimer::DevClockT runTime    = runTimer.getTotalTime();
      DeviceTimer::DevClockT outputTime = outputTimer.getTotalTime();
    
      printf("%d,%u,%llu,%llu,%llu\n",
	     blockIdx.x, nodeId, inputTime, runTime, outputTime);
    }
  
#endif
  
#ifdef INSTRUMENT_OCC
    //
    // @brief print the contents of the node's occupancy counter
    // @param nodeId a numerical identifier to print along with the
    //    output
    //
    __device__
    void printOccupancyCSV(unsigned int nodeId) const
    {
      assert(IS_BOSS());
      printf("%d,%u,%u,%llu,%llu,%llu\n",
	     blockIdx.x, nodeId,
	     occCounter.sizePerRun,
	     occCounter.totalInputs,
	     occCounter.totalRuns,
	     occCounter.totalFullRuns);
    }
#endif
  
#ifdef INSTRUMENT_COUNTS
    //
    // @brief print the contents of the node's item counters
    // @param nodeId a node identifier to print along with the
    //    output
    // @param inputOnly print only the input counts, not the channel
    //    counts
    __device__
    void printCountsCSV(unsigned int nodeId, bool inputOnly) const
    {
      assert(IS_BOSS());
    
      printCountsSingle(itemCounter, nodeId, -1);
    
      if (!inputOnly)
	for (unsigned int c = 0; c < numChannels; c++)
	  printCountsSingle(channels[c]->itemCounter, nodeId, c);
    }
    
    //
    // @brief print the contents of one item counter
    // @param counter the counter to print
    // @param nodeId a node identifier to print along with the
    //         output
    // @param channelId a channel identifier to print along with the 
    //         output
    //
    __device__
    void printCountsSingle(const ItemCounter &counter,
			   unsigned int nodeId, int channelId) const
    {
      printf("%d,%u,%d,%llu\n",
	     blockIdx.x, nodeId, channelId, counter.count);
    }
  
#endif

  protected:
    
    Queue<T> queue;                     // node's input queue
    Queue<Signal> signalQueue;          // node's input signal queue
    ChannelBase* channels[numChannels]; // node's output channels
    
    RefCountedArena::Handle parentHandle; // handle to current parent object

#ifdef INSTRUMENT_TIME
    DeviceTimer inputTimer;
    DeviceTimer runTimer;
    DeviceTimer outputTimer;
#endif
  
#ifdef INSTRUMENT_OCC
    OccCounter occCounter;
#endif
  
#ifdef INSTRUMENT_COUNTS
    ItemCounter itemCounter; // counts inputs to node
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
    // @brief number of inputs currently enqueued for this node.
    //
    __device__
    virtual
    unsigned int numInputsPending() const
    {
      return queue.getOccupancy();
    }
    
    //
    // @brief Create and initialize an output channel.
    //
    // @param c index of channel to initialize
    // @param outputsPerInput Num outputs/input for the channel
    //
    template<typename DST>
    __device__
    void initChannel(unsigned int c, 
		     unsigned int outputsPerInput,
		     bool isAgg = false)
    {
      assert(c < numChannels);
      assert(outputsPerInput > 0);
      
      // init the output channel -- should only happen once!
      assert(channels[c] == nullptr);
      
      channels[c] = new Channel<DST>(outputsPerInput, isAgg);

      // make sure alloc succeeded
      if (channels[c] == nullptr)
	{
	  printf("ERROR: failed to allocate channel object [block %d]\n",
		 blockIdx.x);

	  crash();
	}
    }

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
    void push(const DST &item, 
	      unsigned int channelIdx = 0) const
    {
      Channel<DST>* channel = 
	static_cast<Channel<DST> *>(channels[channelIdx]);
      
      channel->push(item, isThreadGroupLeader());
    }


    // 
    // @brief The main signal handler function for a node.  Read
    // the sigIdx-th signal in the queue and perform whatever action
    // it demands (which may generate additional downstream signals).
    //
    // Return the credit associated with the signal at index sigIdx+1
    // in the queue, if any exists; otherwise, return 0.
    //
    __device__
    unsigned int signalHandler(unsigned int sigIdx)
    {
      const Queue<Signal> &signalQueue = this->signalQueue;
      
      /////////////////////////////
      // SIGNAL HANDLING SWITCH
      /////////////////////////////
      
      const Signal &s = signalQueue.getElt(sigIdx);
      
      switch (s.tag)
	{
	case Signal::Enum:
	  {
	    if (IS_BOSS())
	      {
		// set the parent object for this node (specified
		// as a handle to an object in the ParentBuffer)
		parentHandle = s.handle;
		
		//Reserve space downstream for the new signal
		for (unsigned int c = 0; c < numChannels; ++c)
		  {
		    ChannelBase *channel = getChannel(c);
		    
		    // propagate the signal unless we are at region
		    // frontier
		    if (!channel->isAggregate())
		      channel->pushSignal(s);
		  }
	      }
	    
	    //Call the begin stub of this node
	    this->begin();
	    
	    break;
	  }
	  
	case Signal::Agg:
	  {
	    //Call the end stub of this node
	    this->end();
	    
	    if (IS_BOSS())
	      {
		//Reserve space downstream for the new signal
		for(unsigned int c = 0; c < numChannels; ++c)
		  {
		    ChannelBase *channel = getChannel(c);
		    
		    // if we're not at a region frontier, propagate
		    // the signal; if we are, we can remove our reference
		    // to the parent object.
		    if(!channel->isAggregate())
		      channel->pushSignal(s);
		    else
		      parentHandle.unref();
		  }
	      }
	    break;
	  }
	  
	default:
	  {
	    assert(false && "Invalid signal type detected");
	  }
	}
      
      // return credit from next signal if there is one
      return (sigIdx + 1 < signalQueue.getOccupancy()
	      ? signalQueue.getElt(sigIdx + 1).credit
	      : 0);
    }

  };  // end Node class
}  // end Mercator namespace

#include "Channel.cuh"

#endif
