#ifndef __NODEBASE_CUH
#define __NODEBASE_CUH

//
// @file NodeBase.cuh
// @brief Base class of MERCATOR node object, used to
//   access different nodes uniformly for scheduling,
//   initialization, finalization, and instrumentation printing.
//
// MERCATOR
// Copyright (C) 2021 Washington University in St. Louis; all rights reserved.
//

#include <climits>
#include <cassert>

#include "Scheduler.cuh"

#include "options.cuh"

#include "instrumentation/occ_counter.cuh"

namespace Mercator  {
  
  //
  // @class NodeBase
  // @brief A base class for nodes
  //
  
  class NodeBase {
    
    // largest possible flushing status -- any flush will override it
    static const unsigned int NO_FLUSH = UINT_MAX;
    
  public:

    ///////////////////////////////////////////////////////
    // INIT/CLEANUP KERNEL FUNCIIONS
    ///////////////////////////////////////////////////////
    
    __device__
    NodeBase(Scheduler *ischeduler, 
	     unsigned int iregion, 
	     NodeBase *iusNode)
      : region(iregion),
	scheduler(ischeduler),
	usNode(iusNode),
	active(false), blocked(false), nDSActive(0),
	flushStatus(NO_FLUSH)
    {}
    
    __device__
    virtual ~NodeBase() {}
    
    /////////////////////////////////////////////////////////
    
    //
    // @brief is any input queued for this node?
    // (Only used for debugging.)
    //
    __device__
    virtual bool hasPending() const = 0;
    
    
    //
    // @brief method to fire a node, consuming input and
    // possibly producing output.  This is implemented
    // appropriately by different subclasses of Node.
    //
    __device__
    virtual void fire() = 0;
    
    //
    // @brief initialization code run each time an app starts.
    // Subclasses supply their own implementations if desired
    //
    __device__
    virtual void init() {}


    //
    // @brief initialization code run each time an app finishes.
    // Subclasses supply their own implementations if desired
    //
    __device__
    virtual void cleanup() {}
    
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
      if (!isActive())
	{
	  active = true;
	  if (usNode) //source has no upstream neighbor
	    usNode->incrDSActive();
	  
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
      
      if (isActive()) // we never actually call deactivate on an inactive node
	{
	  active = false;
	  if (usNode)  // source has no upstream neighbor
	    usNode->decrDSActive();
	}
    }
    
    //
    // @brief does this node have any active downstream neighbors?
    //
    __device__
    bool isDSActive() const
    { return nDSActive > 0; }
    
    //
    // @brief is ths node currently blocked?
    //
    __device__
    bool isBlocked() const
    { return blocked; }
    
    //
    // @brief set this node to blocking status
    //
    __device__
    void block()
    {
      assert(IS_BOSS());
      
      blocked = true;
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
      
      if (isBlocked())
	{
	  blocked = false;
	  if (nDSActive == 0)
	    scheduler->addFireableNode(this);
	}
    }
    
    ///////////////////////////////////////////////////////////////////
    // OUTPUT CODE FOR INSTRUMENTATION
    ///////////////////////////////////////////////////////////////////
    
#ifdef INSTRUMENT_TIME
    
    //
    // @brief print the contents of the node's timers
    // @param nodeId a numerical identifier to print along with the
    //    output
    //
    __device__
    void printTimersCSV(unsigned int nodeId) const
    {
      assert(IS_BOSS());
      
      DeviceTimer::DevClockT overheadTime = overheadTimer.getTotalTime();
      DeviceTimer::DevClockT userTime     = userTimer.getTotalTime();
      DeviceTimer::DevClockT pushTime     = pushTimer.getTotalTime();
      
      printf("%d,%u,%llu,%llu,%llu\n",
	     blockIdx.x, nodeId, userTime, pushTime, overheadTime);
    }
  
#endif
  
#ifdef INSTRUMENT_OCC
    //
    // @brief print the contents of the node's occupancy counter
    // @param nodeId a numerical identifier to print along with the
    //    output
    //
    __device__
    void printOccupancyCSV(unsigned int nodeId) const
    {
      assert(IS_BOSS());
      
      printf("%d,%u,%llu,%llu,%llu\n",
	     blockIdx.x, nodeId,
	     occCounter.totalInputs,
	     occCounter.totalRuns,
	     occCounter.totalFullRuns);
    }
#endif
    
#ifdef INSTRUMENT_TIME
    DeviceTimer overheadTimer;
    DeviceTimer userTimer;
    DeviceTimer pushTimer;
#endif
  
#ifdef INSTRUMENT_OCC
    OccCounter occCounter;
#endif
    
    
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

  public:

    //
    // @brief true iff the node is flushing
    //
    __device__
    bool isFlushing() const
    {
      return (flushStatus <= region);
    }
    
    //
    // @brief ask a downstream neighboring node to start a new
    // flush associated with a given region
    //
    // @param dsNode the downstream node to start flushing
    // @param flushRegion the region ID associated with the flush
    //
    __device__
    static void flush(NodeBase *dsNode, unsigned int flushRegion)
    {
      assert(IS_BOSS());
      
      dsNode->flushStatus = min(dsNode->flushStatus, flushRegion);
      
      if (dsNode->flushStatus == flushRegion)
	dsNode->activate();
    }
    
  protected:
    
    //
    // @brief get our current flush status
    //
    __device__
    unsigned int getFlushStatus() const
    { return flushStatus; }
    
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

    const unsigned int region;   // region identifier for flushing    
    Scheduler* const scheduler;  // scheduler used to enqueue fireable nodes
    NodeBase* const usNode;      // upstream neighbor in dataflow graph
    
    bool active;                 // is the node active?
    bool blocked;                // is the node blocking?
    unsigned int nDSActive;      // # of active downstream children of node
    unsigned int flushStatus;    // is node flushing? If so, how far?

    
    ////////////////////////////////////////////////////////////////
    // PRIVATE PORTION OF SCHEDULING INTERFACE
    ////////////////////////////////////////////////////////////////
    
    __device__
    bool isActive() const
    { return active; }
    
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
      if (nDSActive == 0 && isActive() && !isBlocked())
	scheduler->addFireableNode(this);
    }
        
  };    // end class NodeBase
}   // end Mercator namespace

#endif
