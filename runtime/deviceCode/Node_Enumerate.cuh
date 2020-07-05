#ifndef __NODE_ENUMERATE_CUH
#define __NODE_ENUMERATE_CUH

#include <cassert>

#include "Node.cuh"
#include "ChannelBase.cuh"

#include "timing_options.cuh"

namespace Mercator  {

  //
  // @class Node_Enumerate
  // @brief MERCATOR node whose run() fcn takes one input per thread group
  // We use CRTP rather than virtual functions to derive subtypes of this
  // nod, so that the run() function can be inlined in fire().
  // The expected signature of run is
  //
  //   __device__ void run(const T &data)
  //
  // @tparam T type of input item
  // @tparam numChannels  number of output channels 
  // @tparam runWithAllThreads call run with all threads, or just as many
  //           as have inputs?
  // @tparam DerivedNodeType subtype that defines the run() function
  template<typename T, 
	   unsigned int numChannels,
	   unsigned int threadGroupSize,
	   unsigned int maxActiveThreads,
	   bool runWithAllThreads,
	   unsigned int THREADS_PER_BLOCK,
	   typename DerivedNodeType>
  class Node_Enumerate
    : public Node< NodeProperties<T, 
				  numChannels,
				  1, 
				  threadGroupSize,
				  maxActiveThreads,
				  runWithAllThreads,
				  THREADS_PER_BLOCK> > {
    
    typedef Node< NodeProperties<T,
				 numChannels,
				 1,
				 threadGroupSize,
				 maxActiveThreads,
				 runWithAllThreads,
				 THREADS_PER_BLOCK> > BaseType;
    
  public:
    
    __device__
    Node_Enumerate(unsigned int queueSize,
		   Scheduler *scheduler)
      : BaseType(queueSize, scheduler),
	dataCount(0),
	currentCount(0),
	nFrontierNodes(1),	// FIXME: get from compiler
	parentBuffer(queueSize),
	refCounts(queueSize)
    {}
    
  protected:

    using BaseType::getChannel;
    using BaseType::maxRunSize; 
    
    using BaseType::isFlushing;

    using BaseType::setCurrentParent;
    
    // make these downwardly available to the user
    using BaseType::getNumActiveThreads;
    using BaseType::getThreadGroupSize;
    using BaseType::isThreadGroupLeader;

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

    // reference count associated with current parent object -- needed
    // to generate Agg signal if enumeration takes multiple firings
    unsigned int *currRefCount;
    
    
    // Where parent objects of the enumerate node are stored.  Size is
    // set to the same as data queue currently
    Queue<T> parentBuffer;
    
    // Where the current number of references finished for each parent
    // object is stored.
    Queue<unsigned int> refCounts;
    
    
    //
    // @brief find the number of data items that need to be enumerated
    // from the current parent object.
    //
    // Like the run function, this is filled out by the user in the
    // generated code.  This function is called SINGLE THREADED.
    //
    __device__
    virtual
    unsigned int findCount() = 0;
    
    
    __device__
    bool startItem(const T &item, unsigned int *count)
    {
      __syncthreads();
      
      if (parentBuffer.getFreeSpace() == 0)
	{
	  if (IS_BOSS())
	    {
	      while (!parentBuffer.empty() && refCounts.getHead() == 0)
		{
		  parentBuffer.dequeue();
		  refCounts.dequeue();
		}
	    }
	  __syncthreads();
	  
	  if (parentBuffer.getFreeSpace() == 0)
	    {
	      if (IS_BOSS())
		printf("PARENT FULL\n");
	      
	      // buffer is actually full -- FIXME: initiate DS flushing
	      getChannel(0)->getDSNode()->activate();
	      
	      // re-enqueue ourselves
	      this->deactivate();
	      this->activate();
	      
	      return false;
	    }
	}
      
      // set new current parent and issue enumerate signal
      __shared__ unsigned int eltCount;
      if (IS_BOSS())
	{
	  T *currParent = &parentBuffer.enqueue(item);
	  currRefCount  = &refCounts.enqueue(nFrontierNodes);
	  
	  setCurrentParent(currParent);
		  
	  // Create new Enum signal to send downstream
	  Signal s_new(Signal::Enum);	
	  s_new.setParent(currParent);
	  
	  getChannel(0)->pushSignal(s_new);
	  
	  eltCount = this->findCount();
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
	  s_new.setRefCount(currRefCount);
	
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
      assert(numChannels == 1);
      
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
      
      // amount of space free on downstream data queue
      unsigned int dsSpace = channel->dsCapacity();
      
      // # of items already consumed from queue
      unsigned int nDataConsumed = 0;
      unsigned int nSignalsConsumed = 0;
      
      // state of partially emitted item, if any
      unsigned int myDataCount = dataCount;
      unsigned int myCurrentCount = currentCount;

      __syncthreads(); // protect channel capacity
      
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
		  channel->moveOutputToDSQueue(this->getWriteThruId());
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
		nCredits = this->signalHandler(nSignalsConsumed++);
	    }
	  
	  TIMER_STOP(run);
	      
	  TIMER_START(output);
	  
	  __syncthreads();
	  
	  //
	  // Check whether child has been activated by filling a queue
	  //
	  if (channel->checkDSFull())
	    {
	      channel->getDSNode()->activate();
	      anyDSActive = true;
	    }
	  
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
