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

#include "Channel.cuh"

#include "Queue.cuh"

#include "device_config.cuh"

#include "options.cuh"

namespace Mercator  {

  //
  // @class Node
  // @brief MERCATOR most general node type
  //
  // @tparam numChannels  number of channels
  // @tparam numEltsPerGroup number of input elements/thread
  // @tparam threadGroupSize  number of threads in a thread group
  // @tparam maxActiveThreads max # of live threads in any call to run()
  // @tparam runWithAllThreads call run() with all threads, or just as many
  //           as have inputs?
  // @tparam T type of input
  //
  template <unsigned int numChannels,
	    unsigned int numEltsPerGroup,
	    unsigned int threadGroupSize,
	    unsigned int maxActiveThreads,
	    bool runWithAllThreads,
	    unsigned int THREADS_PER_BLOCK,
	    typename... Ts>
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
	 Scheduler *scheduler)
      : NodeBase(scheduler),
	queue(queueSize)
    {
      // init channels array
      for(unsigned int c = 0; c < numChannels; ++c)
	{
	  channels[c] = nullptr;
	  dsNodes[c] = nullptr;
	}
      
#ifdef INSTRUMENT_OCC
      occCounter.setMaxRunSize(maxRunSize);
#endif
    }
    
    
    //
    // @brief set the generic channel object and downstream node ptrs
    // for a given channel.
    //
    __device__
    void setDSEdgeGeneric(unsigned int c,
			  NodeBase *dsNode,
			  ChannelBase *channel)
    {
      dsNode->setParent(this);
      dsNodes[c] = dsNode;
      channels[c] = channel;
    }
    
		   
    //
    // @brief return our queue (needed for setDSEdge()).
    //
    __device__
    Queue<Ts...> *getQueue()
    { 
      return &queue; 
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
  
  protected:

    Queue<Ts...> queue;                  // node's input queue
    
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
    // @brief inspector for the downstream nodes array (for subclasses)
    // @param c index of downstream node to get
    //
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
    
  private:
    
    ChannelBase* channels[numChannels];  // node's output channels
    NodeBase *dsNodes[numChannels];      // node's downstream neighbors
    
  };  // end Node class
}  // end Mercator namespace

#endif
