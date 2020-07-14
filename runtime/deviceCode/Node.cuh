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

#include "Channel.cuh"

#include "Queue.cuh"
#include "Signal.cuh"
#include "ParentBuffer.cuh"

#include "device_config.cuh"

#include "options.cuh"

namespace Mercator  {

  //
  // @class Node
  // @brief most general typed node
  //
  // @tparam T type of input
  // @tparam numChannels  number of channels
  // @tparam numEltsPerGroup number of input elements/thread
  // @tparam threadGroupSize  number of threads in a thread group
  // @tparam maxActiveThreads max # of live threads in any call to run()
  // @tparam runWithAllThreads call run() with all threads, or just as many
  //           as have inputs?
  //
  template <typename T, 
	    unsigned int numChannels,
	    unsigned int numEltsPerGroup,
	    unsigned int threadGroupSize,
	    unsigned int maxActiveThreads,
	    bool runWithAllThreads,
	    unsigned int THREADS_PER_BLOCK>
  class Node : public NodeBase {
    
    // actual maximum # of possible active threads in this block
    static const unsigned int deviceMaxActiveThreads =
      (maxActiveThreads > THREADS_PER_BLOCK 
       ? THREADS_PER_BLOCK 
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
        signalQueue(queueSize), // could be smaller?
	parentArena(nullptr)
    {
      // init channels array
      for (unsigned int c = 0; c < numChannels; ++c)
	{
	  channels[c] = nullptr;
	  dsNodes[c] = nullptr;
	}
      
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
    // @brief set the generic channel object and downstream
    // node pts for a given channel.
    //
    // @param channelIdx channel that holds edge
    // @param dsNode node at downstream end of edge
    // @param channel channel object for downstream channel
    //
    __device__
    void setDSEdge(unsigned int channelIdx,
		   NodeBase *dsNode,
		   QueueBase *dsQueue,
		   Queue<Signal> *dsSignalQueue)

    { 
      dsNodes[channelIdx] = dsNode;
      dsNode->setParentNode(this);
      
      channels[channelIdx]->setDSEdge(dsQueue, dsSignalQueue);
    }
    
    __device__
    void setParentArena(RefCountedArena *a)
    {
      parentArena = a;
    }
    
    //
    // @brief return our queue (needed for channel's setDSEdge()).
    //
    __device__
    Queue<T> *getQueue()
    { 
      return &queue; 
    }
    
    //
    // @brief return our signal queue (needed for channel's setDSEdge()).
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

    // begin and end stubs for enumeration and aggregation 
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
    NodeBase*    dsNodes[numChannels];  // node's downstream neighbors
    
    // state for nodes in enumerated regions
    RefCountedArena *parentArena;       // ptr to any associated parent buffer
    unsigned int parentIdx;             // index of parent obj in buffer

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
      
      channels[c] = new Channel<DST,THREADS_PER_BLOCK>(outputsPerInput, 
						       numThreadGroups,
						       threadGroupSize,
						       numEltsPerGroup,
						       isAgg);

      // make sure alloc succeeded
      if (channels[c] == nullptr)
	{
	  printf("ERROR: failed to allocate channel object [block %d]\n",
		 blockIdx.x);
	  
	  crash();
	}
    }
    
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

    __device__
    NodeBase *getDSNode(unsigned int c) const
    {
      assert(c < numChannels);
      return dsNodes[c];
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
    void push(const DST &item, unsigned int channelIdx = 0) const
    {
      Channel<DST,THREADS_PER_BLOCK>* channel = 
	static_cast<Channel<DST,THREADS_PER_BLOCK> *>(channels[channelIdx]);
      
      channel->push(item);
    }

    
    ////////////////////////////////////////////////////////////////////
    // SIGNAL HANDLING LOGIC
    //
    // As new signal types Foo are added in Signal.h, we need to add a
    // handler here named handleFoo() and update the switch stmt in
    // handleSignal() accordingly.  Handlers are always VIRTUAL so
    // that subclasses may override them; if a generic Node needs no
    // handler for a signal, it should be given an empty function
    // here, and the subclass should provide a new version.
    ///////////////////////////////////////////////////////////////////

    // 
    // @brief Signal handling dispatch for a node.  Consume the
    // sigIdx-th signal in the queue and perform whatever action it
    // demands (which may generate additional downstream signals).
    //
    // Return the credit associated with the signal at index sigIdx+1
    // in the queue, if any exists; otherwise, return 0.
    //
    __device__
    unsigned int handleSignal(unsigned int sigIdx)
    {
      const Queue<Signal> &signalQueue = this->signalQueue;
      
      /////////////////////////////
      // SIGNAL HANDLING SWITCH
      /////////////////////////////
      
      const Signal &s = signalQueue.getElt(sigIdx);
      
      switch (s.tag)
	{
	case Signal::Enum:
	  handleEnum(s);
	  break;
	  
	case Signal::Agg:
	  handleAgg(s);
	  break;
	  
	default:
	  if (IS_BOSS())
	    printf("ERROR: unhandled signal type %d detected\n", s.tag);
	  assert(false);
	}
      
      // return credit from next signal if there is one
      return (sigIdx + 1 < signalQueue.getOccupancy()
	      ? signalQueue.getElt(sigIdx + 1).credit
	      : 0);
    }
    
  protected:    

    __device__
    virtual
    void handleEnum(const Signal &s)
    {
      if (IS_BOSS())
	{
	  // set the parent object for this node (specified
	  // as an index into its parent arena)
	  
	  parentIdx = s.parentIdx;
	  parentArena->ref(parentIdx);
	  
	  
	  //Reserve space downstream for the new signal
	  for (unsigned int c = 0; c < numChannels; ++c)
	    {
	      ChannelBase *channel = getChannel(c);
	      
	      // propagate the signal unless we are at region frontier
	      if (!channel->isAggregate())
		{
		  // add reference for newly created signal
		  parentArena->ref(parentIdx); 
		  channel->pushSignal(s);
		}
	    }
	  
	  parentArena->unref(parentIdx); // signal is destroyed
	}
      
      //Call the begin stub of this node
      this->begin();
    }
    
    
    __device__
    virtual
    void handleAgg(const Signal &s)
    {
      //Call the end stub of this node
      this->end();
      
      if (IS_BOSS())
	{
	  //Reserve space downstream for the new signal
	  for (unsigned int c = 0; c < numChannels; ++c)
	    {
	      ChannelBase *channel = getChannel(c);
	      
	      // if we're not at a region frontier, propagate
	      // the signal; if we are, we can remove our reference
	      // to the parent object.
	      if(!channel->isAggregate())
		channel->pushSignal(s);	    
	    }
	  
	  parentArena->unref(parentIdx); // remove this node's reference
	}
    }
    
  };  // end Node class
}  // end Mercator namespace

#endif
