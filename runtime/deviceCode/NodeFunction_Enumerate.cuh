#ifndef __NODEFUNCTION_ENUMERATE_CUH
#define __NODEFUNCTION_ENUMERATE_CUH

//
// @file Node_Enumerate.cuh
// @brief MERCATOR node function that enumerates the elements of a
// composite item and passes a stream of element indices to its downstream
// child.
//
// MERCATOR
// Copyright (C) 2021 Washington University in St. Louis; all rights reserved.
//

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
  // @tparam InputView type of input view passed to doRun()
  // @tparam THREADS_PER_BLOCK constant giving thread block size
  // @tparam DerivedNodeFnKind subtype that defines the findCount() function
  // 
  template<typename T, 
	   typename InputView,
	   unsigned int THREADS_PER_BLOCK,
	   template <typename View> typename DerivedNodeFnKind>
  class NodeFunction_Enumerate : public NodeFunction<1> {
    
    using BaseType = NodeFunction<1>;
    using DerivedNodeFnType = DerivedNodeFnKind<InputView>;
    
    using BaseType::node;
    
  public:
    
    ///////////////////////////////////////////////////////
    // INIT/CLEANUP KERNEL FUNCIIONS
    ///////////////////////////////////////////////////////

    // @param parentArena parent object arena for region *containing* us
    // @param enumId identifier for region of which we are the head
    // @param nTerminalNodes number of terminal nodes in the region for
    //        which we are the head
    __device__
    NodeFunction_Enumerate(RefCountedArena *parentArena,
			   unsigned int enumId,
			   unsigned int nTerminalNodes)
      : BaseType(parentArena),
	enumId(enumId),
	nTerminalNodes(nTerminalNodes),
	parentBuffer(128), // FIXME: what size should it be?
        dataCount(0),
        currentCount(0)
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
    static const unsigned int maxInputs = 1;
    
    //
    // @brief function to execute code specific to this node.  This
    // function does NOT remove data from the queue.
    //
    // This function assumes that we process either 0 or 1 inputs in
    // one call to doRun().  We could loop to process more than 1
    // input in a call if there is downstream space, but doRun is
    // inlined in the node, so the benefits of adding an additional
    // level of loop to avoid returning to the caller are negligible.
    //
    // @param queue data queue containing items to be consumed
    // @param start index of first item in queue to consume
    // @param limit max number of items that this call may consume
    // @return number of items *FULLY* enumerated in this call (0 or 1)
    //
    __device__
    unsigned int doRun(const InputView &view,
		       size_t start,
		       unsigned int limit)
    {
      DerivedNodeFnType* const nf = static_cast<DerivedNodeFnType *>(this);
	      
      using Channel = Channel<unsigned int>;
      Channel* const channel = static_cast<Channel*>(node->getChannel(0));
      
      // recover state of last partially emitted parent, if any
      unsigned int myDataCount    = dataCount;
      unsigned int myCurrentCount = currentCount;

      unsigned int tid = threadIdx.x;
      unsigned int nFinished = 0;
      
      do
	{
	  // begin a new parent if it's time.  IF we cannot (due to
	  // full parent buffer), indicate that we've read nothing
	  if (myCurrentCount == myDataCount)
	    {
	      const typename InputView::EltT item =
		view.get(start + nFinished);
	      
	      // BEGIN WRITE blocking status, ds signal queue
	      __syncthreads();
	    
	      if (IS_BOSS())
		{
		  if (parentBuffer.isFull())
		    {
		      // initiate DS flushing to ensure that downstream
		      // clears out some space, then block.  We'll be 
		      // rescheduled to execute once the buffer is no 
		      // longer full and we can unblock.
		      channel->flush(enumId);
		      node->block();
		    }
		  else
		    {
		      // save new item to the parent buffer
		      unsigned int pIdx = 
			parentBuffer.alloc(item, nTerminalNodes);
      
		      // send new Enum signal downstream
		      Signal s(Signal::Enum);	
		      s.parentIdx = pIdx;
		      
		      channel->pushSignal(s);
		    }
		}
	      
	      // END WRITE blocking status, ds signal queue
	      __syncthreads();
	      
	      if (node->isBlocked())
		break;
	      
	      myDataCount = nf->findCount(item);
	      myCurrentCount = 0;
	    }
	  
	  // push as many elements as we can from the current
	  // item to the DS node
	
	  unsigned int nEltsToWrite = 
	    min(myDataCount - myCurrentCount, channel->dsCapacity());
	
	  __syncthreads(); // BEGIN WRITE basePtr, ds queue, dsActive status
	
	  __shared__ size_t basePtr;
	  if (IS_BOSS())
	    basePtr = channel->dsReserve(nEltsToWrite);
	
	  __syncthreads(); // END WRITE basePtr, ds queue, dsActive status
	
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
	      // We count an item as finished only when all of its elements
	      // are enumerated; otherwise, the node might think it is 
	      // all done and leave us with a partially enumerated last item.
	      nFinished++;
	    
	      NODE_OCC_COUNT(1, 1);		  
	    }
	  else
	    break; // had to stop without finishing item -- ds queue is full
	}
      while (nFinished < limit);
      
      __syncthreads(); // BEGIN WRITE dataCount, currentCount
      
      if (IS_BOSS())
	{
	  // save any partial parent state
	  dataCount    = myDataCount;
	  currentCount = myCurrentCount;
	  
	  // if we consumed all available input and our node is
	  // flushing, send an extra signal so that downstream nodes
	  // can also finish their open parents.
	  if (nFinished == limit && node->isFlushing())
	    {

	      Signal s_new(Signal::Enum);	
	      s_new.parentIdx = RefCountedArena::NONE;
	      
	      channel->pushSignal(s_new);
	    }
	}
      
      __syncthreads(); // END WRITE dataCount, currentCount
      
      return nFinished;
    }
    
  private:
    
    // ID of node's enumeration region (used for flushing)
    const unsigned int enumId;
    
    // # of terminal nodes for region -- used for parent refcount
    const unsigned int nTerminalNodes;
    
    // Where parent objects of the enumerate node are stored.  Size is
    // set to the same as data queue currently
    ParentBuffer<T> parentBuffer;
    
    // total number of items in currently enumerating object
    unsigned int dataCount;
    
    // number of items so far in currently enumerating object
    unsigned int currentCount;
  };
  
}  // end Mercator namespace

#endif
