#ifndef __NODEBASE_CUH
#define __NODEBASE_CUH

//
// @file NodeBase.cuh
// @brief Base class of MERCATOR node object, used to
//   access different nodes uniformly for scheduling,
//   initialization, finalization, and instrumentation printing.
//
// MERCATOR
// Copyright (C) 2020 Washington University in St. Louis; all rights reserved.
//

#include <climits>
#include <cassert>

#include "Scheduler.cuh"

#include "options.cuh"

namespace Mercator  {
  
  //
  // @class NodeBase
  // @brief A base class for nodes
  //
  
  class NodeBase {
    
    // largest possible flushing status -- any flush will override it
    static const unsigned int NO_FLUSH = UINT_MAX;
  
  public:
    
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
    // end of its incoming edge).
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
    // @brief is any input queued for this node?
    // (Only used for debugging.)
    //
    __device__
    virtual
    bool hasPending() const = 0;

    
    //
    // @brief method to fire a node, consuming input and
    // possibly producing output.  This is implemented
    // appropriately by different subclasses of Node.
    //
    __device__
    virtual
    void fire() = 0;
    
    
    ///////////////////////////////////////////////////////////
    // SCHEDULING PROTOCOL API
    // We use the AFIE (active-full/inactive-empty) protocol
    // described in Plano & Buhler 2020.  The protocol is
    // slightly extended here to allow acive nodes to block
    // waiting for resources other than their input and output
    // queues, e.g., waiting for space to free up in an internal
    // buffer due to the activity of downstream nodes.
    ///////////////////////////////////////////////////////////

    //
    // @brief set node to be active for scheduling purposes.
    // If this makes node fireable, schedule it for execution.
    //
    __device__
    void activate()
    {
      assert(IS_BOSS());

      // do not reschedule already-active nodes -- we can activate
      // an active node when we put it into flush mode.
      if (!isActive)
	{
	  isActive = true;
	  if (parent) //source has no parent
	    parent->incrDSActive();
	  
	  if (nDSActive == 0) // inactive nodes cannot be blocked
	    scheduler->addFireableNode(this);
	}   
    }
    
    //
    // @brief set a node to be inactive for scheduling purposes
    //
    __device__
    void deactivate()
    {
      assert(IS_BOSS());
      
      if (isActive) // we never actually call deactivate on an inactive node
	{
	  isActive = false;
	  if (parent)  // source has no parent
	    parent->decrDSActive();
	}
    }
    
    //
    // @brief Schedule ourself for firing.  This is needed in case
    // the node did not empty its input or fill its output when it
    // last became active (possible only due to blocking on another
    // resource or insufficient input to the Source).
    //
    __device__
    void forceReschedule()
    {
      assert(IS_BOSS());
      assert(isActive && nDSActive == 0 && !isBlocked);
      
      scheduler->addFireableNode(this);
    }
    
    
    //
    // @brief set this node to blocking status
    //
    __device__
    void block()
    {
      assert(IS_BOSS());
      
      isBlocked = true;
    }
    

    //
    // @brief remove the block on this node and force it to
    // reschedule.  Because it blocked while active (and hence had
    // just been taken off the firable worklist), it is active
    // and still not on the worklist.  If it did not fill its
    // output queues before blocking, it can now safely be fired,
    //
    __device__
    void unblock()
    {
      assert(IS_BOSS());
      
      if (isBlocked)
	{
	  isBlocked = false;
	  if (nDSActive == 0)
	    scheduler->addFireableNode(this);
	}
    }
        
    ////////////////////////////////////////////////////////////////
    // INIT and CLEANUP stubs, called at the beginning and end of app
    // execution, respectively.  These may be reimplemented by the
    // user for any user-defined node with state, and system-defined
    // nodes can use them too.
    ////////////////////////////////////////////////////////////////
    
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
    
  protected:
    
    ///////////////////////////////////////////////////////////////
    // FLUSHING API
    // Flushing is used to ensure that all input to a node
    // is consumed, even if the node would not ordinarily
    // continue to fire due to having only a small amount of input
    // available.  The source uses flushing to clear the pipeline
    // when it sees the end of its input stream, while enumerating
    // nodes use it to free up space in a full parent buffer by
    // forcing open parents to finish processing.
    //
    //
    // Each node is assigned a nonnegative integer "region" for
    // flushing purposes.  A flush issued by node N propagates to all
    // connected nodes with EQUAL OR GREATER region numbers.  The
    // source is always region 0, so its flushes reach the entire
    // graph.  Regions associated with an enumeration receive higher
    // numbers and so can have flushes restricted to the nodes
    // associated with one enumeration.
    //
    // The MERCATOR compiler must ensure that if there is an
    // edge directly from enumerated region A to nonoverlapping
    // enumerated region B, B's region has a lower number than A's (so
    // that flushes issued by A do not propagate to B.)  Eventually,
    // when we support nested regions, if a region B lies wholly 
    // inside region A, B must hav ea higher region number than A.
    // By fixing these properties at compile time, we can avoid
    // having to explicitly check at runtime for region boundaries.
    //
    // At any time, a node has an integer FLUSHING STATUS.  If the
    // node's status is > its region ID, it is not flushing.
    // Otherwise, it *is* flushing, and the status indicates the
    // region that caused the flush.  When a flushing node consumes
    // all its input, it propagates its flushing status downstream and
    // then clears its flushing status; the rule of propagation given
    // above determines whether propagation succeeds.
    //////////////////////////////////////////////////////////////
    
    //
    // @brief true iff the current node is flushing
    //
    __device__
    bool isFlushing() const
    {
      return (flushStatus <= region);
    }
    
    //
    // @brief ask a downstream neighboring node to start a new
    // flush associated with our region.
    //
    // @param dsNode the downstream node to start flushing
    //
    __device__
    void initiateFlush(NodeBase *dsNode)
    {
      assert(IS_BOSS());
      
      dsNode->flushStatus = min(dsNode->flushStatus, region);
    }

    //
    // @brief propagate our flushing status to a downstream neighbor.
    //
    // @param dsNode the downstream node that receives the propagation.
    //
    __device__
    void propagateFlush(NodeBase *dsNode)
    {
      assert(IS_BOSS());
      
      dsNode->flushStatus = min(dsNode->flushStatus, flushStatus);
    }
    
    //
    // @brief clear our flushing status
    //
    __device__
    void clearFlush()
    {
      assert(IS_BOSS());
      
      flushStatus = NO_FLUSH;
    }

  private:
    
    Scheduler *scheduler;      // scheduler used to enqueue fireable nodes
    
    NodeBase *parent;          // parent of this node in dataflow graph
    
    bool isActive;             // is node active?
    bool isBlocked;            // is node blocked from execution?
    
    unsigned int nDSActive;    // # of active downstream children of node
    
    unsigned int region;       // region identifier for flushing
    unsigned int flushStatus;  // is node flushing? If so, how far?

    
    ////////////////////////////////////////////////////////////////
    // PRIVATE PORTION OF SCHEDULING INTERFACE
    ////////////////////////////////////////////////////////////////

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
    // @brief decrement node's count of active downstream children.
    // If this makes the node fireable, AND it is not blocked from
    // firing for some reason, schedule it for execution.
    //
    __device__
    void decrDSActive()
    {
      assert(IS_BOSS());
      
      assert(nDSActive > 0);

      nDSActive--;
      if (nDSActive == 0 && isActive && !isBlocked)
	scheduler->addFireableNode(this);
    }
        
  };    // end class NodeBase
}   // end Mercator namespace

#endif
