#ifndef __BUFFEREDCHANNELBASE_CUH
#define __BUFFEREDCHANNELBASE_CUH

//
// @file BufferedChannelBase.cuh
// @brief MERCATOR channel object with a buffer to allow push() with 
// a subset of threads
//
// MERCATOR
// Copyright (C) 2020 Washington University in St. Louis; all rights reserved.
//

#include "ChannelBase.cuh"

#include "options.cuh"

namespace Mercator  {
    
  //
  // @class BufferedChannelBase
  // @brief Holds all data associated with an output stream from a node.
  //
  class BufferedChannelBase : public ChannelBase {
    
  public:

    ///////////////////////////////////////////////////////
    // INIT/CLEANUP KERNEL FUNCIIONS
    ///////////////////////////////////////////////////////
    
    //
    // @brief Constructor (called single-threaded)
    //
    // @param ioutputsPerInput Outputs per input for this channel
    //
    __device__
    BufferedChannelBase(unsigned int ioutputsPerInput, bool isAgg,
			unsigned int inumThreadGroups,
			unsigned int ithreadGroupSize,
			unsigned int numEltsPerGroup)
      : ChannelBase(inumThreadGroups * numEltsPerGroup * ioutputsPerInput, 
		    isAgg),
	outputsPerInput(ioutputsPerInput),
	numThreadGroups(inumThreadGroups),
	threadGroupSize(ithreadGroupSize)
    {
      nextSlot = new unsigned char [numThreadGroups];
      for (unsigned int j = 0; j < numThreadGroups; j++)
	nextSlot[j] = 0;
    }
    
    __device__
    virtual
    ~BufferedChannelBase()
    { 
      delete [] nextSlot; 
    }
    
    /////////////////////////////////////////////////////////////
    
    //
    // @brief move any pushed data from output buffer to downstream queue
    // MUST BE CALLED WITH ALL THREADS
    //
    __device__
    virtual
    void completePush() = 0;
  
  protected:
    
    const unsigned int outputsPerInput;
    const unsigned int numThreadGroups;
    const unsigned int threadGroupSize;
    
    unsigned char *nextSlot;
  }; // end BufferedChannelBase class
}  // end Mercator namespace

#endif
