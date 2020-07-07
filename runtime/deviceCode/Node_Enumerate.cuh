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
    
  public:
    
    __device__
    Node_Enumerate(unsigned int queueSize,
		   Scheduler *scheduler)
      : BaseType(queueSize, scheduler),
	nFrontierNodes(1),	// FIXME: get from compiler
	dataCount(0),
	currentCount(0),
	parentBuffer(queueSize)
    {}
    
  protected:

    using BaseType::getChannel;
    using BaseType::maxRunSize; 
    
    using BaseType::isFlushing;
    
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
    
    
    __device__
    bool startItem(const T &item, unsigned int *count)
    {
      __syncthreads();
      
      if (parentBuffer.isFull())
	{
	  // buffer is actually full -- FIXME: initiate DS flushing
	  getChannel(0)->getDSNode()->activate();
	  
	  // re-enqueue ourselves
	  this->deactivate();
	  this->activate();
	  
	  return false;
	}
      
      // set new current parent and issue enumerate signal
      __shared__ unsigned int eltCount;
      if (IS_BOSS())
	{
	  parentHandle = parentBuffer.alloc(item, nFrontierNodes);

	  eltCount = this->findCount(item);
	  
	  // Create new Enum signal to send downstream
	  Signal s_new(Signal::Enum);	
	  s_new.setHandle(parentHandle);
	  
	  getChannel(0)->pushSignal(s_new);
	}
      __syncthreads();
      
      *count = eltCount;
      return true;
    }

    __device__
    void finishItem()
    {
      if (IS_BOSS())
	{
	  Signal s_new(Signal::Agg);
	  
	  getChannel(0)->pushSignal(s_new);
	}
    }

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
    virtual
    void fire()
    {
      unsigned int tid = threadIdx.x;
      
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
			       : signalQueue.getHead().getCredit());
            
      // # of items already consumed from queue
      unsigned int nDataConsumed = 0;
      unsigned int nSignalsConsumed = 0;

      
      // amount of space free on downstream data queue
      unsigned int dsSpace = channel->dsCapacity();
      
      // state of partially emitted item, if any
      unsigned int myDataCount = dataCount;
      unsigned int myCurrentCount = currentCount;

      __syncthreads(); // protect ds channel capacity
      
      bool anyDSActive = false;
      
      while ((nDataConsumed < nDataToConsume ||
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
	  
	  unsigned int nItems = min(limit, 1); // only enum one input at a time
	  
	  TIMER_STOP(input);
	  
	  TIMER_START(run);
	  
	  if (nItems > 0)
	    {
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
		min(myDataCount - myCurrentCount, dsSpace);
	      
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
	      dsSpace -= nEltsToWrite;
	      
	      if (myCurrentCount == myDataCount)
		{
		  finishItem();
		  nDataConsumed++;
		}
	    }
	  
	  // only consume a signal if we are not mid-item
	  if (nSignalsToConsume > 0 && myCurrentCount == myDataCount)
	    {
	      //
	      // Track credit to next signal, and consume if needed.
	      //
	      nCredits -= nItems;
	      
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
	  anyDSActive |= channel->checkDSFull(this->getWriteThruId());
	  
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
	    signalQueue.getHead().setCredit(nCredits);
	  
	  dataCount = myDataCount;
	  currentCount = myCurrentCount;
	  
	  // FIXME: what does this do?
	  if (this->getWriteThruId() > 0) 
	    {
	      NodeBase *dsNode = channel->getDSNode();
	      dsNode->setWriteThruId(this->getWriteThruId());
	      dsNode->activate();
	    }
	  
	  if (nDataConsumed == nDataToConsume &&
	      nSignalsConsumed == nSignalsToConsume)
	  {
	    // less than a full ensemble remains, or 0 if flushing
	    this->deactivate(); 
	    
	    if (isFlushing)
	      {
		// no more inputs to read -- force downstream nodes
		// into flushing mode and activate them (if not
		// already active).  Even if they have no input,
		// they must fire once to propagate flush mode and
		// activate *their* downstream nodes.
		NodeBase *dsNode = channel->getDSNode();
		dsNode->setFlushing(true);
		dsNode->activate();
	      }
	    
	    this->setFlushing(false);
	  }
	}
      
      TIMER_STOP(input);
    }
  };
  
}  // end Mercator namespace

#endif
