#ifndef __NODE_ENUMERATE_CUH
#define __NODE_ENUMERATE_CUH

#include <cassert>

#include "Node.cuh"
#include "ChannelBase.cuh"

#include "ParentBuffer.cuh"

#include "timing_options.cuh"

namespace Mercator  {

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
    : public Node< NodeProperties<T, 
				  1,             // one output channel
				  1, 1,          // no run/scatter functions
				  THREADS_PER_BLOCK, // use all threads
				  true,
				  THREADS_PER_BLOCK> > {
    
    typedef Node< NodeProperties<T,
				 1, 
				 1, 1,
				 THREADS_PER_BLOCK,
				 true,
				 THREADS_PER_BLOCK> > BaseType;
    
    
  private:

    using BaseType::getChannel;
    using BaseType::maxRunSize; 
    
    using BaseType::parentHandle;
    
#ifdef INSTRUMENT_TIME
    using BaseType::inputTimer;
    using BaseType::runTimer;
    using BaseType::outputTimer;
#endif

#ifdef INSTRUMENT_OCC
    using BaseType::occCounter;
#endif

#ifdef INSTRUMENT_COUNTS
    using BaseType::itemCounter;
#endif

  public:
    
    __device__
    Node_Enumerate(unsigned int queueSize,
		   Scheduler *scheduler,
		   unsigned int region)
      : BaseType(queueSize, scheduler, region),
	nFrontierNodes(1),	// FIXME: get this from compiler
	dataCount(0),
	currentCount(0),
	parentBuffer(10 /*queueSize*/, this) // FIXME: for stress test
    {}
    
    //
    // @brief fire a node, consuming as much input 
    // from its queue as possible
    //
    // PRECONDITION: node is active (hence either flushing or has at
    // least maxRunSize inputs in its queue), and all its downstream
    // nodes are inactive (hence have at least enough space to hold
    // outputs from maxRunSize inputs in their queues).
    //
    // called with all threads
    
    __device__
    void fire()
    {
      unsigned int tid = threadIdx.x;
      
      const unsigned int maxInputSize = 1; // we consume one parent at a time
      
      TIMER_START(input);
      
      Queue<T> &queue = this->queue; 
      Queue<Signal> &signalQueue = this->signalQueue; 
      
      using Channel = typename BaseType::Channel<unsigned int>;
      Channel *channel = static_cast<Channel *>(getChannel(0));
      
      // # of items available to consume from queue
      unsigned int nDataToConsume = queue.getOccupancy();
      unsigned int nSignalsToConsume = signalQueue.getOccupancy();
      
      unsigned int nCredits = (nSignalsToConsume == 0
			       ? 0
			       : signalQueue.getHead().credit);
            
      // # of items already consumed from queue
      unsigned int nDataConsumed = 0;
      unsigned int nSignalsConsumed = 0;
      
      unsigned int inputLB = (this->isFlushing() ? 1 : maxInputSize);
      
      bool anyDSActive = false;
      
      while ((nDataToConsume - nDataConsumed >= inputLB || 
	      nSignalsConsumed < nSignalsToConsume) &&
	     !anyDSActive)
	{
#if 0
	  if (IS_BOSS())
	    printf("%d %p %d %d %d %d %d\n", 
		   blockIdx.x, this, 
		   nDataConsumed, nDataToConsume,  
		   nSignalsConsumed, nSignalsToConsume,
		   nCredits);
#endif
	  
	  unsigned int limit =
	    (nSignalsConsumed < nSignalsToConsume
	     ? nCredits 
	     : nDataToConsume - nDataConsumed);
	  
	  unsigned int nFinished = 0;
	  if (limit > 0)
	    {
	      TIMER_STOP(input);
	      
	      TIMER_START(run);
	      
	      // recover state of partially emitted parent, if any
	      unsigned int myDataCount = dataCount;
	      unsigned int myCurrentCount = currentCount;
	      
	      // begin a new parent if it's time.  IF we cannot
	      // (due to full parent buffer), terminate the main loop.
	      if (myCurrentCount == myDataCount)
		{
		  if (!startItem(queue.getElt(nDataConsumed), &myDataCount))
		    break;
		  
		  myCurrentCount = 0;
		}
	      
	      // push as many elements as we can from the current
	      // item to the DS node
	      
	      unsigned int nEltsToWrite = 
		min(myDataCount - myCurrentCount, channel->dsCapacity());
	      
	      __syncthreads(); // protect read of dsCapacity from updates below
	      
	      for (unsigned int base = 0; 
		   base < nEltsToWrite; 
		   base += maxRunSize)
		{
		  unsigned int v = base + tid;
		  
		  if (v < nEltsToWrite)
		    this->push(v);
		  
		  __syncthreads();
		  
		  // move these items downstream immediately.
		  // We know we won't overflow the dsqueue
		  channel->moveOutputToDSQueue();
		}
	      
	      myCurrentCount += nEltsToWrite;
	      
	      if (myCurrentCount == myDataCount)
		{
		  finishItem();
		  nFinished = 1;
		}
	      
	      // save any partial parent state
	      dataCount = myDataCount;
	      currentCount = myCurrentCount;
	    }
	  nDataConsumed += nFinished;
	  
	  if (nSignalsToConsume > 0)
	    {
	      //
	      // Track credit to next signal, and consume if needed.
	      //
	      nCredits -= nFinished;
	      
	      if (nCredits == 0)
		{
		  nCredits = this->signalHandler(nSignalsConsumed);
		  nSignalsConsumed++;
		}
	    }
	  
	  TIMER_STOP(run);
	  
	  TIMER_START(output);
	  
	  __syncthreads();
	  
	  //
	  // Check whether child has been activated by filling a queue
	  //
	  anyDSActive |= channel->checkDSFull();
	  
	  TIMER_STOP(output);
	  
	  TIMER_START(input);
	}
      
      // protect code above from queue changes below
      __syncthreads();
      
      if (IS_BOSS())
	{
	  COUNT_ITEMS(nDataConsumed);  // instrumentation
	  
	  queue.release(nDataConsumed);
	  signalQueue.release(nSignalsConsumed);
	  
	  if (!signalQueue.empty())
	    signalQueue.getHead().credit = nCredits;
	  
	  if (nDataToConsume - nDataConsumed < inputLB &&
	      nSignalsConsumed == nSignalsToConsume)
	  {
	    // less than a full ensemble remains, or 0 if flushing
	    this->deactivate(); 
	    
	    if (this->isFlushing())
	      {
		// no more inputs to read -- force downstream nodes
		// into flushing mode and activate them (if not
		// already active).  Even if they have no input,
		// they must fire once to propagate flush mode and
		// activate *their* downstream nodes.
		NodeBase *dsNode = channel->getDSNode();
		this->propagateFlush(dsNode);
		dsNode->activate();
		
		this->clearFlush(); // disable
	      }

	  }
	}
      
      TIMER_STOP(input);
    }

  private:

    // # of nodes in this enumeration region's frontier -- used
    // for reference-counting signals for region
    const unsigned int nFrontierNodes;  
    
    // total number of items in currently enumerating object
    unsigned int dataCount;
    
    // number of items so far in currently enumerating object
    unsigned int currentCount;
    
    // Where parent objects of the enumerate node are stored.  Size is
    // set to the same as data queue currently
    ParentBuffer<T> parentBuffer;
    
    //
    // @brief find the number of data items that need to be enumerated
    // from the current parent object.
    //
    // Like the run function, this is filled out by the user in the
    // generated code.  This function is called SINGLE THREADED.
    //
    __device__
    virtual
    unsigned int findCount(const T &item) = 0;
        
    
    //
    // @brief begin enumeration of a new parent object.  If we
    // cannot begin an object becaues the parent buffer is full,
    // block the node and flush its region to make space. Otherwise,
    // add the object to the parent buffer, store a handle to
    // it in the node's state, and pass the handle downstream as
    // a signal to the rest of the region.
    //
    // @param item new parent object
    // @param count output parameter; holds result of findCount(item)
    //
    // Returns true iff we were able to start enumerating item.
    // 
    __device__
    bool startItem(const T &item, unsigned int *count)
    {
      __syncthreads();

      // buffer is full -- initiate DS flushing to clear it out,
      // then terminate.  We'll reschedule ourselves to execute
      // once the buffer is no longer full.
      if (parentBuffer.isFull())
	{
	  if (IS_BOSS())
	    {
	      NodeBase *dsNode = getChannel(0)->getDSNode();
	      
	      this->initiateFlush(dsNode);
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
	  parentHandle = parentBuffer.alloc(item, nFrontierNodes);
	  
	  eltCount = this->findCount(item);
	  
	  // Create new Enum signal to send downstream
	  Signal s_new(Signal::Enum);	
	  s_new.handle = parentHandle;
	  
	  getChannel(0)->pushSignal(s_new);
	}
      __syncthreads();
      
      *count = eltCount;
      return true;
    }

    
    //
    // @brief finish processing a parent item.  We just pass on the
    // end-of-item boundary to our region, which knows what to do
    // with it.
    //
    __device__
    void finishItem()
    {
      if (IS_BOSS())
	{
	  Signal s_new(Signal::Agg);
	  
	  getChannel(0)->pushSignal(s_new);
	}
    }

  };
  
}  // end Mercator namespace

#endif
