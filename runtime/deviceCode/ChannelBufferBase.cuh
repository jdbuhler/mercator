#ifndef __CHANNELBUFFERBASE_CUH
#define __CHANNELBUFFERBASE_CUH

//
// @file ChannelBufferBase.cuh
// @brief MERCATOR channel buffer object to allow push() with 
// a subset of threads
//
// MERCATOR
// Copyright (C) 2021 Washington University in St. Louis; all rights reserved.
//

#include "ChannelBase.cuh"

#include "options.cuh"

namespace Mercator  {
    
  //
  // @class ChannelBufferBase
  // @brief Buffers data sent to an output channel during a single run
  //
  class ChannelBufferBase {
    
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
    ChannelBufferBase(unsigned int numThreadGroups)
    {
      nextSlot = new unsigned char [numThreadGroups];
      for (unsigned int j = 0; j < numThreadGroups; j++)
	nextSlot[j] = 0;
    }
    
    __device__
    virtual ~ChannelBufferBase()
    { 
      delete [] nextSlot; 
    }
    
    /////////////////////////////////////////////////////////////
    
    __device__
    virtual
    void finishWrite(ChannelBase *cb) = 0;
    
  protected:
    
    unsigned char *nextSlot;
  }; // end ChannelBufferBase class
}  // end Mercator namespace

#endif
