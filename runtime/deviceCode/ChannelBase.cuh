#ifndef __CHANNELBASE_CUH
#define __CHANNELBASE_CUH

//
// @file ChannelBase.cuh
// @brief MERCATOR channel base object
//
// MERCATOR
// Copyright (C) 2020 Washington University in St. Louis; all rights reserved.
//

#include <cassert>
#include <cstdio>

#include "Queue.cuh"

#include "Signal.cuh"

#include "options.cuh"

namespace Mercator  {

  //
  // @class ChannelBase
  // @brief Holds all data associated with an output stream from a node.
  //
  class ChannelBase {
    
    enum Flags { FLAG_ISAGGREGATE = 0x01 };
    
  public:
    
    //
    // @brief Constructor (called single-threaded)
    //
    // @param ioutputsPerInput Outputs per input for this channel
    //
    __device__
    ChannelBase(unsigned int ioutputsPerInput, bool isAgg)
      : outputsPerInput(ioutputsPerInput),
	propFlags(isAgg ? FLAG_ISAGGREGATE : 0),
	numItemsWritten(0),
	dsQueue(nullptr),
	dsSignalQueue(nullptr)
    {}
    
    __device__
    virtual
    ~ChannelBase()
    {}
    
    //
    // @brief Set the downstream target of the edge for
    // this channel.
    //
    // @param idsNode downstream node
    //
    __device__
    void setDSEdge(QueueBase     *idsQueue,
		   Queue<Signal> *idsSignalQueue)
    {
      dsQueue = idsQueue;
      dsSignalQueue = idsSignalQueue;
    }
    
    //
    // @brief determine if this channel is aggregating
    //
    __device__
    bool isAggregate() const
    {
      return (propFlags & FLAG_ISAGGREGATE);
    }
    
    //
    // @brief get the number of inputs whose output could
    // be safely written to this channel's downstream queue.
    //
    __device__
    unsigned int dsCapacity() const
    {
      return dsQueue->getFreeSpace() / outputsPerInput;
    }
     
    //
    // @brief get the number of signals whose output could
    // be safely written to this channel's downstream signal queue.
    //
    __device__
    unsigned int dsSignalCapacity() const
    {
      return dsSignalQueue->getFreeSpace();
    }
      
    
    //
    // If we've managed to fill the downstream queue, activate its
    // target node. Let our caller know if we activated the ds node.
    //
    // @param maxRunSize maximum # of inputs that can be emitted in
    //        a single run
    __device__
    bool checkDSFull(unsigned int maxRunSize) const
    {
      return (dsQueue->getFreeSpace() < maxRunSize * outputsPerInput ||
	      dsSignalQueue->getFreeSpace() < MAX_SIGNALS_PER_RUN);
    }
    
    
    __device__
    virtual 
    void completePush() = 0;
    
    //
    // @brief push a signal to a specified channel, and reset the number
    // of items produced on that channel. This function is SINGLE THREADED.
    //
    // @param s the signal being sent downstream
    // @param channel the channel on which the signal is being sent
    //
    __device__
    void pushSignal(const Signal& s)
    {
      assert(dsSignalQueue->getFreeSpace() > 0);
      
      unsigned int credit = 
	(dsSignalQueue->empty()
	 ? dsSignalQueue->getOccupancy()
	 : numItemsWritten);
      
      Signal &sNew = dsSignalQueue->enqueue(s);
      sNew.credit = credit;
      
      numItemsWritten = 0;
    }
    
  protected:
    
    const unsigned int outputsPerInput;  // max # outputs per input to node
    const unsigned int propFlags;        // Signal propagation flags
    
    unsigned int numItemsWritten;        // # items produced since last signal
    
    //
    // target (edge) for writing items downstream
    //
    
    QueueBase *dsQueue;
    Queue<Signal> *dsSignalQueue;
    
    
  }; // end ChannelBase class
}  // end Mercator namespace

#endif
