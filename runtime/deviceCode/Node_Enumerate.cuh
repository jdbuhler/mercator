#ifndef __NODE_ENUMERATE_CUH
#define __NODE_ENUMERATE_CUH

#include <cassert>

#include "Node.cuh"

#include "Channel.cuh"

#include "ParentBuffer.cuh"

namespace Mercator {

  //
  // @class Node_Enumerate
  // @brief MERCATOR node that enumerates the contents of a composite
  // object.  The node has a single output channel on which it emits
  // a stream of consecutive integers for each item enumerated, equal
  // in length to the number of elements in the item according to the
  // user-supplied findCount() function.
  //
  // @tparam T type of input item
  //
  template<typename T, 
	   unsigned int THREADS_PER_BLOCK>
  class Node_Enumerate
    : public Node<T,
		  1,             // one output channel
		  THREADS_PER_BLOCK> {
    
    using BaseType = Node<T,
			  1, 
			  THREADS_PER_BLOCK>;
    
  private:
    
    using BaseType::getChannel;
    using BaseType::getDSNode;
    
    using BaseType::parentIdx;
    
#ifdef INSTRUMENT_OCC
    using BaseType::occCounter;
#endif

  public:
    
    ///////////////////////////////////////////////////////
    // INIT/CLEANUP KERNEL FUNCIIONS
    ///////////////////////////////////////////////////////

    __device__
    Node_Enumerate(unsigned int queueSize,
		   Scheduler *scheduler,
		   unsigned int region,
		   RefCountedArena *parentArena,
		   unsigned int ienumId)
      : BaseType(queueSize, scheduler, region, parentArena),
	enumId(ienumId),
	parentBuffer(10 /*queueSize*/, this), // FIXME: for stress test
	dataCount(0),
	currentCount(0)
    {
#ifdef INSTRUMENT_OCC
      occCounter.setMaxRunSize(1);
#endif
    }
    
    //
    // @brief get the parent buffer associated with this node 
    // (to pass to nodes in this region)
    //
    __device__
    RefCountedArena *getParentArena()
    { return &parentBuffer; }
    
    ///////////////////////////////////////////////////////

  private:

    // ID of node's enumeration region (used for flushing)
    const unsigned int enumId;
    
    // Where parent objects of the enumerate node are stored.  Size is
    // set to the same as data queue currently
    ParentBuffer<T> parentBuffer;

    // total number of items in currently enumerating object
    unsigned int dataCount;
    
    // number of items so far in currently enumerating object
    unsigned int currentCount;
    
    //
    // @brief find the number of data items that need to be enumerated
    // from the current parent object.
    //
    // Like the run function, this is filled out by the user in the
    // generated code.  This function is called SINGLE-THREADED.
    //
    __device__
    virtual
    unsigned int findCount(const T &item) = 0;

    //
    // @brief doRun() processes inputs one at a time
    //
    __device__
    unsigned int inputSizeHint() const
    { return 1; }
    
    
    //
    // @brief expand next input to its elements, issuing signals
    // each time a new parent input starts.
    //
    __device__
    unsigned int doRun(const Queue<T> &queue, 
		       unsigned int start,
		       unsigned int limit)
    {
      unsigned int tid = threadIdx.x;
      
      using Channel = Channel<int>;
      Channel *channel = static_cast<Channel*>(getChannel(0));
      
      unsigned int nFinished = 0;
      if (limit > 0)
	{
	  // recover state of partially emitted parent, if any
	  unsigned int myDataCount    = dataCount;
	  unsigned int myCurrentCount = currentCount;
	  
	  // begin a new parent if it's time.  IF we cannot
	  // (due to full parent buffer), indicate that we've read nothing 
	  if (myCurrentCount == myDataCount)
	    {
	      if (!startItem(queue.getElt(start), &myDataCount))
		return 0;
	      
	      NODE_OCC_COUNT(1);
	      
	      myCurrentCount = 0;
	    }
	  
	  // push as many elements as we can from the current
	  // item to the DS node
	  
	  unsigned int nEltsToWrite = 
	    min(myDataCount - myCurrentCount, channel->dsCapacity());
	  
	  __syncthreads(); // protect read of dsCapacity from updates below
	  
	  for (unsigned int base = 0; 
	       base < nEltsToWrite; 
	       base += THREADS_PER_BLOCK)
	    {
	      unsigned int vecSize = 
		min(THREADS_PER_BLOCK, nEltsToWrite - base);
	      
	      unsigned int v = myCurrentCount + base + tid;
	      
	      channel->pushCount(v, vecSize);
	    }
	  
	  myCurrentCount += nEltsToWrite;
	  
	  if (myCurrentCount == myDataCount)
	    {
	      finishItem();
	      nFinished++;
	    }
	  
	  // save any partial parent state
	  dataCount    = myDataCount;
	  currentCount = myCurrentCount;
	}
      
      return nFinished;
    }
    
    //
    // @brief begin enumeration of a new parent object.  If we cannot
    // begin an object becaues the parent buffer is full, block the
    // node and flush its region to make space. Otherwise, add the
    // object to the parent buffer, store its index in the buffer in
    // the node's state, and pass the index downstream as a signal to
    // the rest of the region.
    //
    // MUST BE CALLED WITH ALL THREADS
    //
    // @param item new parent object
    // @param count output parameter; holds result of findCount(item)
    //
    // @return true iff we were able to start enumerating the item.
    // 
    __device__
    bool startItem(const T &item, unsigned int *count)
    {
      // buffer is full -- initiate DS flushing to clear it out, then
      // block.  We'll be rescheduled to execute once the buffer
      // is no longer full and we can unblock.
      if (parentBuffer.isFull())
	{
	  if (IS_BOSS())
	    {
	      NodeBase *dsNode = getDSNode(0);
	      
	      if (this->initiateFlush(dsNode, enumId))
		dsNode->activate();
	      
	      this->block();
	    }
	  
	  return false;
	}
      
      __syncthreads(); // protect read of parentBuffer from write below
      
      // set new current parent and issue enumerate signal
      __shared__ unsigned int eltCount;
      if (IS_BOSS())
	{
	  // new parent object already has refcount == 1
	  parentIdx = parentBuffer.alloc(item);
	  
	  // Create new Enum signal to send downstream
	  Signal s_new(Signal::Enum);	
	  s_new.parentIdx = parentIdx;
	  
	  parentBuffer.ref(parentIdx); // for use in signal
	  getChannel(0)->pushSignal(s_new);
	  
	  eltCount = this->findCount(item);
	}
      
      __syncthreads();
      
      *count = eltCount;
      return true;
    }
    
    
    //
    // @brief finish processing a parent item by dropping our reference
    // to it.
    //
    // SHOULD BE CALLED WITH ALL THREADS
    //
    __device__
    void finishItem()
    {
      if (IS_BOSS())
	{
	  parentBuffer.unref(parentIdx); // drop reference to parent item
	}
    }
    
        
    //
    // @brief if we have emptied our inputs in response to a flush,
    // signal our downstream neighbors so that, when they are flushing,
    // they can finish off any open parent.
    //
    __device__
    void flushComplete()
    {
      assert(IS_BOSS());
      
      if (parentIdx != RefCountedArena::NONE)
	{
	  // item was already unreferenced when we finished it
	  
	  parentIdx = RefCountedArena::NONE;
	  
	  // push a signal to force downstream nodes to finish off
	  // previous parent
	  Signal s_new(Signal::Enum);	
	  s_new.parentIdx = RefCountedArena::NONE;
      
	  getChannel(0)->pushSignal(s_new);
	}
    }

  };
  
}  // end Mercator namespace

#endif
