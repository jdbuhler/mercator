#ifndef __NODE_BASE_CUH
#define __NODE_BASE_CUH

//
// @file NodeBase.cuh
// @brief Base class of MERCATOR node object,  used to
//   access different nodes uniformly for scheduling,
//   initialization, finalization, and instrumentation printing.
//
// MERCATOR
// Copyright (C) 2019 Washington University in St. Louis; all rights reserved.
//

#include <cassert>

#include "Scheduler.cuh"

#include "options.cuh"

namespace Mercator  {
  
  //
  // @class NodeBase
  // @brief A base class for nodes
  //
  
  class NodeBase {
  
  public:
    
    static const unsigned int NO_FLUSH = (unsigned int) -1;
    
    __device__
    NodeBase(Scheduler *ischeduler, unsigned int iregion)
      : scheduler(ischeduler),
	parent(nullptr),
	isActive(false),
	isBlocked(false),
	nDSActive(0),
	flushStatus(NO_FLUSH),
	region(iregion)
    {}
    
    __device__
    virtual 
    ~NodeBase() {}
    
    
    //
    // @brief set the parent of this node (the node at the upstream
    // end of is incoming edge).
    //
    // @param iparent parent node
    ///
    __device__
    void setParentNode(NodeBase *iparent)
    { 
      assert(IS_BOSS());
      
      parent = iparent;
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
	  if (parent) //source has no parent
	    parent->incrDSActive();
	  
	  if (nDSActive == 0) // node is eligible for firing
	    scheduler->addFireableNode(this);
	}   
    }
    
    //
    // @brief set node to be inactive for scheduling purposes
    //
    __device__
    void deactivate()
    {
      assert(IS_BOSS());
      assert(isActive);
      if (isActive)
	{
	  isActive = false;
	  if (parent)  // source has no parent
	    parent->decrDSActive();
	}
    }
    
    __device__
    void forceReschedule()
    {
      assert(IS_BOSS());
      assert(isActive && nDSActive == 0);
      
      scheduler->addFireableNode(this);
    }
    
    //
    // @brief increment node's count of active downstream children;
    //
    __device__
    void incrDSActive()
    {
      assert(IS_BOSS());
      
      nDSActive++;
    }
    
    //
    // @brief decrement node's count of active downstream children;
    // if this makes the node fireable, AND it is not blocked from
    // firing for some reason, schedule it for execution.
    __device__
    void decrDSActive()
    {
      assert(IS_BOSS());
      
      assert(nDSActive > 0);

      nDSActive--;
      if (nDSActive == 0 && isActive && !isBlocked) // node eligible for firing
	scheduler->addFireableNode(this);
    }

    __device__
    bool isFlushing() const
    {
      return (flushStatus <= region);
    }

    __device__
    void initiateFlush(NodeBase *dsNode)
    {
      assert(IS_BOSS());
      
      dsNode->flushStatus = min(dsNode->flushStatus, region);
    }
    
    __device__
    void propagateFlush(NodeBase *dsNode)
    {
      assert(IS_BOSS());
      
      dsNode->flushStatus = min(dsNode->flushStatus, flushStatus);
    }

    __device__
    void clearFlush()
    {
      assert(IS_BOSS());
      
      flushStatus = NO_FLUSH;
    }
        
    __device__
    void block()
    {
      assert(IS_BOSS());
      
      isBlocked = true;
    }
    
    __device__
    void unblock()
    {
      assert(IS_BOSS());
      
      if (isBlocked)
	{
	  isBlocked = false;
	  forceReschedule();
	}
    }
    
    __device__
    virtual
    unsigned int numPending() const = 0;

    __device__
    virtual
    void fire() = 0;

    __device__
    virtual
    void init() {}
    
    __device__
    virtual
    void cleanup() {}
    
    //////////////////////////////////////////////////////////////
    // INSTRUMENTATION PRINTING (see Node.h for details)
    //////////////////////////////////////////////////////////////
    
#ifdef INSTRUMENT_TIME
    __device__
    virtual
    void printTimersCSV(unsigned int nodeId) const = 0;
#endif
    
#ifdef INSTRUMENT_OCC
    __device__
    virtual
    void printOccupancyCSV(unsigned int nodeId) const = 0;
#endif
    
#ifdef INSTRUMENT_COUNTS
    __device__
    virtual
    void printCountsCSV(unsigned int nodeId, bool inputOnly = false) const = 0;
#endif
    
    
  private:
    
    Scheduler *scheduler;      // scheduler used to enqueue fireable nodes
    
    NodeBase *parent;          // parent of this node in dataflow graph
    
    bool isActive;             // is node in active
    unsigned int nDSActive;    // # of active downstream children of node
    unsigned int flushStatus; // is node flushing? If so, how far?
    
    bool isBlocked;
    
    unsigned int region;
    
  };    // end class NodeBase
}   // end Mercator namespace

#endif
