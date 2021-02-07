#ifndef __NODEBASEWITHCHANNELS_CUH
#define __NODEBASEWITHCHANNELS_CUH

//
// @file NodeBaseWithChannels.cuh
// @brief a node base class that supplies a channel and dsNode array
//
// MERCATOR
// Copyright (C) 2021 Washington University in St. Louis; all rights reserved.
//

#include <cassert>

#include "NodeBase.cuh"

#include "ChannelBase.cuh"

#include "device_config.cuh"

#include "options.cuh"

namespace Mercator  {

  //
  // @class NodeBaseWithChannels
  // @brief base class to store/provide channel and dsNode information
  //
  // @tparam numChannels  number of channels
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
	channels[c] = nullptr;
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

  public:
    
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
    
  };  // end NodeBaseWithChannels class
}  // end Mercator namespace

#endif
