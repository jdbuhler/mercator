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

#include "Scheduler.cuh"

#include "options.cuh"

namespace Mercator  {
  
  //
  // @class NodeBase
  // @brief An untyped base class for nodes
  //
  
  class NodeBase {
  
  public:
    
    __device__
    NodeBase(Scheduler *ischeduler)
      : scheduler(ischeduler),
	parent(nullptr),
	isActive(false),
	nDSActive(0),
	_isFlushing(false)
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
    void setParent(NodeBase *iparent)
    { 
      parent = iparent;
    }
    
    __device__
    bool isFlushing() const
    {
      return _isFlushing;
    }
    
    //
    // @brief indicate that node is in flush mode
    //
    __device__
    void setFlushing()
    {
      _isFlushing = true;
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

	  if (parent)
	    parent->incrDSActive();
	  
	  if (nDSActive == 0) // node is eligible for firing
	    scheduler->addFireableNode(this);
	}   
    }
    
    //
    // @brief set node to be inactive for scheduling purposes;
    //
    __device__
    void deactivate()
    {
      assert(IS_BOSS());
      
      isActive = false;
      
      if (parent)  // source has no parent
	parent->decrDSActive();
    }

    // for debugging only
    __device__
    virtual
    unsigned int numPending() = 0;

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
    bool _isFlushing;          // is node in flushing mode?


    //
    // @brief increment node's count of active downstream children.
    //
    __device__
    void incrDSActive()
    {
      assert(IS_BOSS());
      
      nDSActive++;
    }
    
    //
    // @brief decrement node's count of active downstream children;
    // if this makes the node fireable, schedule it for execution.
    //
    __device__
    void decrDSActive()
    {
      assert(IS_BOSS());
      
      nDSActive--;
      if (nDSActive == 0 && isActive) // node is eligible for firing
	scheduler->addFireableNode(this);
    }
    
  };    // end class NodeBase
}   // end Mercator namespace

#endif
