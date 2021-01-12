#ifndef __NODEFUNCTION_ENUMERATE_CUH
#define __NODEFUNCTION_ENUMERATE_CUH

#include <cassert>

#include "NodeFunction.cuh"

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
  class NodeFunction_Enumerate : public NodeFunction<1> {
    
    using BaseType = NodeFunction<1>;
    
    using BaseType::node;
    
  public:
    
    ///////////////////////////////////////////////////////
    // INIT/CLEANUP KERNEL FUNCIIONS
    ///////////////////////////////////////////////////////
    
    __device__
    NodeFunction_Enumerate(RefCountedArena *parentArena,
			   unsigned int ienumId)
      : BaseType(parentArena),
	enumId(ienumId),
	parentBuffer(10), // for stress test -- increase to queue size?
        dataCount(0),
        currentCount(0),
	activeParent(RefCountedArena::NONE)
    {}
    
    // override basic NodeFunction::SetNode to ensure that we
    // also associate our node with the parent buffer, in addition
    // to whatever our superclass does.
    __device__
    void setNode(NodeType *node)
    {
      BaseType::setNode(node);
      parentBuffer.setBlockingNode(node);
    }
    
    //
    // @brief get the parent buffer associated with this node 
    // (to pass to nodes in this region)
    //
    __device__
    RefCountedArena *getArena()
    { return &parentBuffer; }
    
    ///////////////////////////////////////////////////////
    
    //
    // doRun() processes inputs one at a time
    //
    static const unsigned int inputSizeHint = 1;
    
    //
    // @brief function to execute code specific to this node.  This
    // function does NOT remove data from the queue.
    //
    // @param queue data queue containing items to be consumed
    // @param start index of first item in queue to consume
    // @param limit max number of items that this call may consume
    // @return number of items ACTUALLY consumed (may be 0).
    //
    __device__
    unsigned int doRun(const Queue<T> &queue, 
		       unsigned int start,
		       unsigned int limit)
    {
      unsigned int tid = threadIdx.x;
      
      using Channel = Channel<int>;
      Channel *channel = static_cast<Channel*>(node->getChannel(0));
      
      unsigned int nFinished = 0;
      
      // recover state of partially emitted parent, if any
      unsigned int myDataCount    = dataCount;
      unsigned int myCurrentCount = currentCount;
      
      // begin a new parent if it's time.  IF we cannot (due to
      // full parent buffer), indicate that we've read nothing
      if (myCurrentCount == myDataCount)
	{
	  const T &item = queue.getElt(start);
	  
	  // BEGIN WRITE blocking status, activeParent,
	  // ds signal queue ptr in startItem()
	  __syncthreads();
	  
	  if (IS_BOSS())
	    {
	      if (parentBuffer.isFull())
		{
		  node->block();
		  
		  // initiate DS flushing to clear it out, then
		  // block.  We'll be rescheduled to execute once
		  // the buffer is no longer full and we can
		  // unblock.
		  
		  NodeBase *dsNode = node->getDSNode(0);
		  
		  if (node->initiateFlush(dsNode, enumId))
		    dsNode->activate();
		}
	      else
		{
		  activeParent = startItem(item);
		}
	    }
	  
	  // END WRITE blocking status, activeParent,
	  // ds signal queue ptr in startItem()
	  __syncthreads();
	  
	  if (node->isBlocked())
	    return 0;
	  
	  NODE_OCC_COUNT(1);
	  
	  myDataCount = findCount(item);
	  myCurrentCount = 0;
	}
      
      // push as many elements as we can from the current
      // item to the DS node
      
      unsigned int nEltsToWrite = 
	min(myDataCount - myCurrentCount, channel->dsCapacity());
      
      __syncthreads(); // BEGIN WRITE basePtr, ds queue tail
      
      __shared__ unsigned int basePtr;
      if (IS_BOSS())
	basePtr = channel->dsReserve(nEltsToWrite);
      
      __syncthreads(); // END WRITE basePtr, ds queue tail
      
      for (unsigned int base = 0; 
	   base < nEltsToWrite; 
	   base += THREADS_PER_BLOCK)
	{
	  unsigned int srcIdx = base + tid;
	  
	  if (srcIdx < nEltsToWrite)
	    {
	      unsigned int v = myCurrentCount + srcIdx;
	      channel->dsWrite(basePtr, srcIdx, v);
	    }
	}
      
      myCurrentCount += nEltsToWrite;
      
      if (myCurrentCount == myDataCount)
	{
	  if (IS_BOSS())
	    {
	      // finished with this parent item -- drop reference to it
	      parentBuffer.unref(activeParent);
	    }				
	  
	  nFinished++;
	}
      
      __syncthreads(); // BEGIN WRITE dataCount, currentCount
      
      if (IS_BOSS())
	{
	  // save any partial parent state
	  dataCount    = myDataCount;
	  currentCount = myCurrentCount;
	}
      
      __syncthreads(); // END WRITE dataCount, currentCount
      
      return nFinished;
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
      
      if (activeParent != RefCountedArena::NONE)
	{
	  // item was already unreferenced when we finished it
	  
	  activeParent = RefCountedArena::NONE;
	  
	  // push a signal to force downstream nodes to finish off
	  // previous parent
	  Signal s_new(Signal::Enum);	
	  s_new.parentIdx = RefCountedArena::NONE;
	  
	  node->getChannel(0)->pushSignal(s_new);
	}
    }
    
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

    // the active parent object for enumeration
    unsigned int activeParent;
    
    //
    // @brief find the number of data items that need to be enumerated
    // from the current parent object.
    //
    // Like the run function, this is filled out by the user in the
    // generated code.  This function is called SINGLE-THREADED.
    //
    __device__
    virtual
    unsigned int findCount(const T &item) const = 0;
    
    //
    // @brief begin enumeration of a new parent object. Add the object
    // to the parent buffer, store its index in the buffer in the
    // node's state, and pass the index downstream as a signal to the
    // rest of the region.
    //
    // @param item new parent object
    // @return index of new item in parent buffer
    //
    __device__
    unsigned int startItem(const T &item)
    {
      assert(IS_BOSS());
      
      // new parent object already has refcount == 1
      unsigned int pIdx = parentBuffer.alloc(item);
      
      // Create new Enum signal to send downstream
      Signal s_new(Signal::Enum);	
      s_new.parentIdx = pIdx;
      parentBuffer.ref(pIdx); // for use in signal
      
      node->getChannel(0)->pushSignal(s_new);
      
      return pIdx;
    }
  };
  
}  // end Mercator namespace

#endif
