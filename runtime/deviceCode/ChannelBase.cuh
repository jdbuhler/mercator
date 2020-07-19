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
    
    ///////////////////////////////////////////////////////
    // INIT/CLEANUP KERNEL FUNCIIONS
    ///////////////////////////////////////////////////////

    //
    // @brief Constructor (called single-threaded)
    //
    // @param minFreeSpace minimum space required for ds queue to be non-full
    //
    __device__
    ChannelBase(unsigned int iminFreeSpace, bool isAgg)
      : minFreeSpace(iminFreeSpace),
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
    // @param idsQueue downstream data queue
    // @param idsSignalQueue downstream signal queue
    //
    __device__
    void setDSQueues(QueueBase     *idsQueue,
		     Queue<Signal> *idsSignalQueue)
    {
      assert(IS_BOSS());
      
      dsQueue = idsQueue;
      dsSignalQueue = idsSignalQueue;
    }
    
    ///////////////////////////////////////////////////////
    
    //
    // @brief determine if this channel is aggregating
    //
    __device__
    bool isAggregate() const
    {
      return (propFlags & FLAG_ISAGGREGATE);
    }
    
    //
    // @brief get free space of the downstream data queue
    //
    __device__
    unsigned int dsCapacity() const
    {
      return dsQueue->getFreeSpace();
    }
    
    //
    // If we've managed to fill the downstream queue, activate its
    // target node. Let our caller know if we activated the ds node.
    //
    __device__
    bool checkDSFull() const
    {
      return (dsQueue->getFreeSpace() < minFreeSpace ||
	      dsSignalQueue->getFreeSpace() < MAX_SIGNALS_PER_RUN);
    }
    
    //
    // @brief push a signal to a specified channel, and reset the number
    // of items produced on that channel. 
    //
    // @param s the signal being sent downstream
    //
    __device__
    void pushSignal(const Signal& s)
    {
      assert(IS_BOSS());
      assert(dsSignalQueue->getFreeSpace() > 0);
      
      unsigned int credit = 
	(dsSignalQueue->empty()
	 ? dsQueue->getOccupancy()
	 : numItemsWritten);
      
      Signal &sNew = dsSignalQueue->enqueue(s);
      sNew.credit = credit;
      
      numItemsWritten = 0;
    }
    
  protected:
    
    const unsigned int minFreeSpace;  // min space for queue not to be full
    const unsigned int propFlags;     // Signal propagation flags
    
    unsigned int numItemsWritten;     // # items produced since last signal
    
    //
    // target (edge) for writing items downstream
    //
    
    QueueBase *dsQueue;
    Queue<Signal> *dsSignalQueue;
    
    //
    // @brief prepare for a direct write to the downstream queue(s)
    // by reserving space for the items to write.
    //
    // @param number of slots to reserve for next write
    // @return starting index of reserved segment.
    //
    __device__
    unsigned int dsReserve(unsigned int nToWrite) const
    {
      assert(IS_BOSS());
      return dsQueue->reserve(nToWrite);
    }
  }; // end ChannelBase class
}  // end Mercator namespace

#endif
