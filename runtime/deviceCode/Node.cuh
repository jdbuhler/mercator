#ifndef __NODE_CUH
#define __NODE_CUH

//
// @file Node.cuh
// @brief a MERCATOR node that knows its input type
//
// MERCATOR
// Copyright (C) 2019 Washington University in St. Louis; all rights reserved.
//

#include <cstdio>
#include <cassert>
#include <climits>

#include "NodeBase.cuh"

#include "Signal.cuh"

#include "Queue.cuh"

#include "Scheduler.cuh"

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
	 Scheduler *ischeduler)
      : queue(queueSize),
        signalQueue(queueSize),	//stimcheck: Currently use data queue size for signal queue size.
	scheduler(ischeduler),
	parent(nullptr),
	isActive(false),
	nDSActive(0),
	isFlushing(false),
	enumId(0),
	currentCreditCounter(0),
	currentParent(nullptr),
	writeThruId(0)
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

      channels[c] = new Channel<DST>(outputsPerInput);

      // make sure alloc succeeded
      if (channels[c] == nullptr)
	{
	  printf("ERROR: failed to allocate channel object [block %d]\n",
		 blockIdx.x);

	  crash();
	}

      if(isAgg)
	channels[c]->setAggregate();
    }


    //
    // @brief Construc tthe edge between this node and a downstream
    // neighbor on a partrcular channel.
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

      dsNode->setParent(this);
      channel->setDSEdge(dsNode, dsNode->getQueue(), reservedSlots, dsNode->getSignalQueue());
    }


    //
    // @brief return our queue (needed for setDSEdge().
    //
    __device__
    Queue<T> *getQueue()
    { 
      return &queue; 
    }

    //
    // @brief return our signal queue (needed for setDSEdge().
    //
    __device__
    Queue<Signal> *getSignalQueue()
    { 
      return &signalQueue; 
    }

    //
    // @brief set the parent of this node (the node at the upstream
    // end of is incoming edge).
    //
    // @param iparent parent node
    ///
    __device__
    void setParent(NodeBase *iparent)
    { 
      parent = iparent;
    }

    //
    // @brief set the enumeration region Id of the node
    //
    // @param e The enumeration region id to set the node to
    //
    __device__
    void setEnumId(unsigned int e)
    {
      enumId = e;
    }

    //
    // @brief get the enumeration region Id of the node
    //
    // @return unsigned int The enumeration region Id of the node
    //
    __device__
    unsigned int getEnumId()
    {
      return enumId;
    }

    //
    // @brief set the write through enumeration region Id of the node
    //
    // @param e The enumeration region id to set the write through of
    // the node to
    //
    __device__
    void setWriteThruId(unsigned int w)
    {
      writeThruId = w;
    }

    //
    // @brief get the write through enumeration region Id of the node
    //
    // @return unsigned int The enumeration region id to set the write
    // through of the node to
    //
    __device__
    unsigned int getWriteThruId()
    {
      return writeThruId;
    }

    //
    // @brief indicate that node is in flush mode
    //
    __device__
    void setFlushing()
    {
      isFlushing = true;
    }

    //
    // @brief set node to be active for scheduling purposes;
    // if this makes node fireable, schedule it for execution.
    //
    __device__
    void activate()
    {
      assert(IS_BOSS());

      // do not reschedule already-active nodes -- we can activate
      // an active node when we put it into flush mode
      if (!isActive)
	{
	  isActive = true;
	  if (nDSActive == 0) // node is eligible for firing
	    scheduler->addFireableNode(this);
	}   
    }

    //
    // @brief set node to be inactive for scheduling purposes;
    //
    __device__
    void deactivate()
    {
      assert(IS_BOSS());
      
      isActive = false;
      if (parent)  // source has no parent
	parent->decrDSActive();
    }
    
    //
    // @brief decrement node's count of active downstream children;
    // if this makes the node fireable, schedule it for execution.
    //
    __device__
    void decrDSActive()
    {
      assert(IS_BOSS());
      
      nDSActive--;
      if (nDSActive == 0 && isActive) // node is eligible for firing
	scheduler->addFireableNode(this);
    }
    
    //
    // @brief return number of data items queued for this node.
    // (Only used for debugging right now.)
    //
    __device__
    unsigned int numPending()
    {
      return queue.getOccupancy();
    }


    /*
    // 
    // @brief set the current credit counter for the node.
    // 
    // @param ccc The value to set the current credit counter to
    // 
    __device__
    void setCurrentCreditCounter(unsigned int ccc)
    {
	currentCreditCounter = ccc;
    }
	  
    // 
    // @brief set the current credit counter for the node.
    // 
    // @return unsigned int The current credit counter value of the node
    // 
    __device__
    unsigned int getCurrentCreditCounter()
    {
	return currentCreditCounter;
    }
    */

    // 
    // @brief The main signal handler function for nodes.  Perform signal
    // actions and create new signals here.  This IS multithreaded.
    // 
    __device__
    bool signalHandler()
    {
	while(this->numSignalsPending() > 0)
	{
		/////////////////////////////
		// FULL SIGNAL QUEUE CHECK
		/////////////////////////////
		bool hasFullDSSignalQueue = false;
		for(unsigned int c = 0; c < numChannels; ++c)
		{
			//Downstream signal queue is full, return true to indicate as such,
			//and activate the respective downstream nodes.
			if(getChannel(c)->dsSignalCapacity() <= 2)
			{
				if(IS_BOSS())
					getChannel(c)->getDSNode()->activate();
				hasFullDSSignalQueue = true;
			}
		}
		
		//Short circut here when we have a full signal queue.
		if(hasFullDSSignalQueue)
		{
			return true;
		}


		/////////////////////////////
		// CREDIT ASSIGNMENT & CHECK
		/////////////////////////////
		Queue<Signal> &signalQueue = this->signalQueue;

		const Signal& s = signalQueue.getElt(0);	//0th element is always the head of signal queue
		if(!hasSignal)
		{
			currentCreditCounter = s.getCredit();
			hasSignal = true;
		}

		if(hasSignal && currentCreditCounter > 0)
		{
			//Nothing to do here, we still have credit available
			return false;
		}


		/////////////////////////////
		// SIGNAL HANDLING SWITCH
		/////////////////////////////
		const Signal::SignalTag t = signalQueue.getElt(0).getTag();	//0th element is always the head of signal queue

		switch(t)
		{
			case Signal::SignalTag::Enum:
			{
				//Call the begin stub of this node
				this->begin();

				if(IS_BOSS())
				{
					//Create new Enum signal to send downstream
					Signal s_new;
					s_new.setTag(Signal::SignalTag::Enum);
					s_new.setParent(s.getParent());	//Set the parent for the new signal
					setCurrentParent(s.getParent());	//Set the new parent for the node
	
					//Reserve space downstream for the new signal
					for(unsigned int c = 0; c < numChannels; ++c)
					{
						Channel<void*> *channel = static_cast<Channel<void*> *>(getChannel(c));
	
						//If the channel is NOT an aggregate channel, send the new signal downstream
						if(!(channel->isAggregate()))
						{
							if(channel->dsSignalQueueHasPending())
							{
								s_new.setCredit(channel->getNumItemsProduced());
							}
							else
							{
								s_new.setCredit(channel->dsPendingOccupancy());
							}
							pushSignal(s_new, channel);
						}
					}
				}
				break;
			}

			case Signal::SignalTag::Agg:
			{
				//Call the end stub of this node
				this->end();

				if(IS_BOSS())
				{
					//Create new Enum signal to send downstream
					Signal s_new;
					s_new.setTag(Signal::SignalTag::Agg);
					s_new.setRefCount(s.getRefCount());	//Set the parent for the new signal
	
					//Reserve space downstream for the new signal
					for(unsigned int c = 0; c < numChannels; ++c)
					{
						Channel<void*> *channel = static_cast<Channel<void*> *>(getChannel(c));
	
						//If the channel is NOT an aggregate channel, send the new signal downstream
						if(!(channel->isAggregate()))
						{
							if(channel->dsSignalQueueHasPending())
							{
								s_new.setCredit(channel->getNumItemsProduced());
							}
							else
							{
								s_new.setCredit(channel->dsPendingOccupancy());
							}
							pushSignal(s_new, channel);
						}
						else
						{
							//Subtract from the current node's enumeration region ID's reference count
							s.getRefCount()[0] -= 1;
						}
					}
				}
				break;
			}

			default:
			{
				assert(false && "Signal without tag found, aborting. . .");
			}
		}
		__syncthreads();	//Make sure we are done with the current signal . . .
		if(IS_BOSS())
			signalQueue.release(1);	//Release the one signal we just processed.
		__syncthreads();	//Make sure everyone sees the updated signal queue . . .

		hasSignal = false;	//Cannot get here unless we processed a signal.
	}
	__syncthreads();	//Make sure everyone got to the end of signal handling . . .
	return false;		//No more signals to process, can break here
    }

    //
    //
    //
    __device__
    void pushSignal(Signal& s,
	            Channel<void*>* channel) const
    {
	unsigned int dsSignalBase = channel->directSignalReserve(1);

	channel->directSignalWrite(s, dsSignalBase, 0);

	channel->resetNumItemsProduced();
    }

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
    virtual
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
    virtual
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
    virtual
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

  //stimcheck: Begin and end stubs for enumeration and aggregation 
  public: 
    __device__
    virtual
    void begin() {}

    __device__
    virtual
    void end() {}

  protected:

    Queue<T> queue;                     // node's input queue
    Queue<Signal> signalQueue;          // node's input signal queue
    ChannelBase* channels[numChannels]; // node's output channels

    Scheduler *scheduler;      // scheduler used to enqueue fireable nodes
    
    NodeBase *parent;          // parent of this node in dataflow graph
    
    bool isActive;             // is node in active
    unsigned int nDSActive;    // # of active downstream children of node
    bool isFlushing;           // is node in flushing mode?

    int currentCreditCounter;  // the current credit available to a node
    bool hasSignal;  	       // whether or not the current credit is valid (has a signal)
    void* currentParent;       // the current parent object of the current enumeration if applicable
    unsigned int enumId;       // the enumeration region Id of the node, needed for flushing state

    unsigned int writeThruId;	// highest priority localFlush tag seen of this enumeration region Id

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
    // @brief number of signals currently enqueued for this node.
    //
    __device__
    virtual
    unsigned int numSignalsPending() const
    {
      return signalQueue.getOccupancy();
    }
  
    //
    // @brief Sets the currentParent pointer to a specified pointer, which should only be from a Signal.
    //
    // @param v The pointer of the new parent object
    //
    __device__
    virtual
    void setCurrentParent(void* v)
    {
      currentParent = v;
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

  };  // end Node class
}  // end Mercator namespace

#include "Channel.cuh"

#endif
