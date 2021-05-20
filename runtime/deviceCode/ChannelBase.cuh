#ifndef __CHANNELBASE_CUH
#define __CHANNELBASE_CUH

//
// @file ChannelBase.cuh
// @brief MERCATOR channel base object
//
// MERCATOR
// Copyright (C) 2021 Washington University in St. Louis; all rights reserved.
//

#include <cassert>

#include "NodeBase.cuh"

#include "Queue.cuh"

#include "Signal.cuh"

#include "options.cuh"

#include "instrumentation/out_dist_counter.cuh"
#include "instrumentation/maxvectorgain_dist_counter.cuh"

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
    ChannelBase(unsigned int iminFreeSpace, bool isAgg,
		NodeBase *dsNode,
		QueueBase *dsQueue, 
		Queue<Signal> *dsSignalQueue)
      : minFreeSpace(iminFreeSpace),
	propFlags(isAgg ? FLAG_ISAGGREGATE : 0),
	numItemsWritten(0),
	dsNode(dsNode),
	dsQueue(dsQueue),
	dsSignalQueue(dsSignalQueue)
    {}
    
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
    // @brief get total size of the downstream data queue
    //
    __device__
    unsigned int dsSize() const
    {
      return dsQueue->getCapacity();
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
    // @brief prepare for a direct write to the downstream queue(s)
    // by reserving space for the items to write.
    //
    // @param number of slots to reserve for next write
    // @return starting index of reserved segment.
    //
    __device__
    size_t dsReserve(unsigned int nToWrite)
    {
      assert(IS_BOSS());
      assert(dsQueue->getFreeSpace() >= nToWrite);
      
      if (dsQueue->getFreeSpace() - nToWrite < minFreeSpace)
	dsNode->activate();
      
      numItemsWritten += nToWrite;
      
      return dsQueue->reserve(nToWrite);
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
      
      unsigned int credit;
      
      if (dsSignalQueue->empty())
	credit = dsQueue->getOccupancy();
      else
	{
	  if (dsSignalQueue->getFreeSpace() <= MAX_SIGNALS_PER_RUN)
	    dsNode->activate();
	  
	  credit = numItemsWritten;
	}
      
      Signal &sNew = dsSignalQueue->enqueue(s);
      sNew.credit = credit;
      
      numItemsWritten = 0;
    }
    
    //
    // @brief pass a flush request to our downstream node
    //
    __device__
    void flush(unsigned int flushStatus) const
    { NodeBase::flush(dsNode, flushStatus); }
      

    //////////////////////////////////////
    // OUTPUT DISTRIBUTION INSTRUMENTATION
    //////////////////////////////////////

#ifdef INSTRUMENT_OUT_DIST
    OutDistCounter outDistCounter;

    /*__device__
    unsigned long long* getOutDist()
    {
      return outDistCounter.distribution;
    }*/
#endif

#ifdef INSTRUMENT_MAXVECTORGAIN_DIST
    MaxVectorGainCounter maxVectorGainDistCounter;

    /*__device__
    unsigned long long* getOutDist()
    {
      return outDistCounter.distribution;
    }*/
#endif

  protected:
    
    const unsigned int minFreeSpace;  // min space for queue not to be full
    const unsigned int propFlags;     // Signal propagation flags
  
    //
    // target (edge) for writing items downstream
    //
    
    NodeBase* const dsNode;
    QueueBase* const dsQueue;
    Queue<Signal>* const dsSignalQueue;
    
    unsigned int numItemsWritten;     // # items produced since last signal
    
  }; // end ChannelBase class
}  // end Mercator namespace

#endif
