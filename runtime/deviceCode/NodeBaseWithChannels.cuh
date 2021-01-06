#ifndef __NODEBASEWITHCHANNELS_CUH
#define __NODEBASEWITHCHANNELS_CUH

//
// @file NodeBaseWithChannels.cuh
// @brief a MERCATOR node that knows its input type
//
// MERCATOR
// Copyright (C) 2020 Washington University in St. Louis; all rights reserved.
//

#include <cassert>

#include "NodeBase.cuh"

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
  template <unsigned int numChannels>
  class NodeBaseWithChannels : public NodeBase {
    
  public:

    ///////////////////////////////////////////////////////
    // INIT/CLEANUP KERNEL FUNCIIONS
    ///////////////////////////////////////////////////////

    //
    // @brief Constructor
    //
    __device__
    NodeBaseWithChannels(Scheduler *scheduler, 
			 unsigned int region,
			 NodeBase *usNode)
      : NodeBase(scheduler, region, usNode)
    {
      // init channels array
      for (unsigned int c = 0; c < numChannels; ++c)
	{
	  channels[c] = nullptr;
	  dsNodes[c] = nullptr;
	}
    }
    
    //
    // @brief Destructor
    //
    __device__
    ~NodeBaseWithChannels()
    {
      for (unsigned int c = 0; c < numChannels; ++c)
	{
	  ChannelBase *channel = channels[c];
	  if (channel)
	    delete channel;
	}
    }
    
    //
    // @brief associate a downstream edge with a channel
    //
    __device__
    void setDSEdge(unsigned int channelIdx,
		   NodeBase *dsNode,
		   QueueBase *queue,
		   Queue<Signal> *signalQueue)
    {
      assert(IS_BOSS());
      assert(channelIdx < numChannels);
      
      channels[channelIdx]->setDSQueues(queue, signalQueue);
      
      dsNodes[channelIdx] = dsNode;
    }
    
    
  protected:
    
    //
    // @brief set a channel entry for this node
    // (called from node constructors)
    //
    __device__
    void setChannel(unsigned int c, ChannelBase *channel)
    {
      assert(IS_BOSS());
      
      assert(c < numChannels);
      channels[c] = channel;
    }
    
    ///////////////////////////////////////////////////////////
    
  protected:
    
    //
    // @brief inspector for the dsNodes array (for subclasses)
    // @param c index of downstream node to get
    //
    __device__
    NodeBase *getDSNode(unsigned int c) const
    {
      assert(c < numChannels);
      return dsNodes[c];
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
    
  private:
    
    ChannelBase *channels[numChannels]; // node's output channels
    NodeBase    *dsNodes[numChannels];  // node's downstream neighbors
    
  };  // end NodeBaseWithChannels class
}  // end Mercator namespace

#endif
