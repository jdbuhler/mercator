#ifndef __NODE_ENUMERATE_CUH
#define __NODE_ENUMERATE_CUH

#include <cassert>
#include "Node.cuh"
#include "ChannelBase.cuh"
#include "timing_options.cuh"
#include "Queue.cuh"

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
	refCount(1),	//stimcheck: DEFAULT VALUE FOR REF COUNT IS 1
	parentBuffer(queueSize),
	refCounts(queueSize)
    {}
    
  protected:

    using BaseType::getChannel;
    using BaseType::maxRunSize; 
    
    using BaseType::nDSActive;
    using BaseType::isFlushing;
    
    // make these downwardly available to the user
    using BaseType::getNumActiveThreads;
    using BaseType::getThreadGroupSize;
    using BaseType::isThreadGroupLeader;

    //stimcheck: Add base type for setParent
    using BaseType::setCurrentParent;
    
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
  
    unsigned int dataCount;		// # of data items already enumerated from the node
    unsigned int currentCount;		// # of data items needed to be enumerated from the node
    
    unsigned int refCount;		// # of aggregate nodes that must be reached before freeing a parent, NOT YET IMPLEMENTED, DEFAULT IS 1

    Queue<T> parentBuffer;		// Where parent objects of the enumerate node are stored.  Size is set to the same as data queue currently
    Queue<unsigned int> refCounts;	// Where the current number of references finished for each parent object is stored.
    

   __device__
   virtual
   unsigned int findCount() {
	assert(false && "FindCount base called");
	return 0;
   }


   __device__
   bool isParentBufferFull() {
	return (parentBuffer.getCapacity() - parentBuffer.getOccupancy() == 0);
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
      Queue<T> &queue = this->queue; 
      unsigned int mynDSActive = 0;

      do
      {

      ////////////////////////////////////////////
      // FULL PARENT BUFFER CHECK
      ////////////////////////////////////////////
      if(isParentBufferFull()) {
	if(IS_BOSS())
	{
		printf("[%d] PARENT BUFFER FULL\n", blockIdx.x);
		this->activate();	//Re-enqueue and re-activate ourselves since we still have stuff to process
	
		for (unsigned int c = 0; c < numChannels; c++)
		{
			NodeBase *dsNode = getChannel(c)->getDSNode();
			dsNode->setWriteThruId(this->getEnumId());	//Set the writeThru ID of the downstream nodes to that of this node
			dsNode->activate();				//Activate the downstream nodes to pass the local flush mode
		}

		this->scheduler->setLocalFlush(this->getEnumId());
	}
	__syncthreads();
	return;			//Short circut ourselves since we don't have any more space to put our stuff
      }

      ////////////////////////////////////////////
      // EMIT ENUMERATE SIGNAL IF NEEDED
      ////////////////////////////////////////////
      if(dataCount == 0) {
	//Get a new parent object from the data queue
	if(IS_BOSS())
	{
		if(!(isParentBufferFull()))
		{
			unsigned int parentBase = parentBuffer.reserve(1);
			unsigned int refBase = refCounts.reserve(1);
			unsigned int offset = 0;

			const T &elt = queue.getElt(offset);
			const unsigned int &refelt = refCount;

			parentBuffer.putElt(parentBase, offset, elt);
			refCounts.putElt(refBase, offset, refelt);

			void* s;
			void* rc;

			s = parentBuffer.getVoidTail();
			rc = refCounts.getVoidHead();

			setCurrentParent(s);
			//setCurrentRefCount(rc);

			dataCount = this->findCount();
			currentCount = 0;
			//this->setCurrentParent(static_cast<void*>(queue.getElt(0)));
		}

		//Emit Enumerate Signal

		//Create new Enum signal to send downstream
		Signal s_new;
		s_new.setTag(Signal::SignalTag::Enum);	
		s_new.setParent(parentBuffer.getVoidHead());	//Set the parent for the new signal
		setCurrentParent(queue.getVoidHead());	//Set the new parent for the node
		using Channel = typename BaseType::Channel<void*>;
	
		//Reserve space downstream for the new signal
		for(unsigned int c = 0; c < numChannels; ++c)
		{
			Channel *channel = static_cast<Channel*>(getChannel(c));
	
			//If the channel is NOT an aggregate channel, send the new signal downstream
			if(!(channel->isAggregate()))
			{
				if(channel->dsSignalQueueHasPending())
				{
					s_new.setCredit(channel->getNumItemsProduced());
				}
				else
				{
					s_new.setCredit(channel->dsPendingOccupancy());
				}
				pushSignal(s_new, channel);
			}
		}
	}
      }

      __syncthreads();
      
      unsigned int nConsumed = 0;

      while (mynDSActive == 0 && dataCount != currentCount)
	{
	  assert(currentCount < dataCount);
	  unsigned int nItems = min(dataCount - currentCount, maxRunSize);
	  if(IS_BOSS())
		printf("[%d]\t\tnItems %d\t\tDC %d\t\tCC %d\t\tMRS %d\n", blockIdx.x, nItems, dataCount, currentCount, maxRunSize);
	  //assert(dataCount > currentCount);
	  __syncthreads();
	  
	  NODE_OCC_COUNT(nItems);
	  
	  //const T &myData = 
	  //  (tid < nItems
	  //   //? queue.getElt(nConsumed + tid)
	  //   ? queue.getElt(nTotalConsumed + nConsumed + tid)
	  //   : queue.getDummy()); // don't create a null reference
	  
	  TIMER_STOP(input);
	  
	  TIMER_START(run);

	  if (tid < nItems)
	    {
	      //n->run(myData);
	      this->push(currentCount + tid);
		printf("tid: %d\t\tCC: %d\t\tCC+tid: %d\n", tid, currentCount, currentCount+tid);
	    }
	  nConsumed += nItems;

	  __syncthreads();

	  //Add the number of items we just sent downstream to the current counter
	  currentCount += nItems;

	  __syncthreads();

	  TIMER_STOP(run);
	  
	  TIMER_START(output);
	  
	  for (unsigned int c = 0; c < numChannels; c++)
	    {
	      // check whether each channel's downstream node was activated
	      mynDSActive += getChannel(c)->moveOutputToDSQueue(this->getWriteThruId());
	    }

	  __syncthreads();
	  
	  TIMER_STOP(output);
	  
	  TIMER_START(input);
	}

	__syncthreads();

	if(dataCount == currentCount)
	{
		//Emit Agg Signal
		if(IS_BOSS())
		{
			printf("EMITTING AGG SIGNAL\n");
			//Create new Enum signal to send downstream
			Signal s_new;
			s_new.setTag(Signal::SignalTag::Agg);	
			s_new.setRefCount(refCounts.getVoidTail());	//Set the refCount to default 1 TODO
			using Channel = typename BaseType::Channel<void*>;
	
			//Reserve space downstream for the new signal
			for(unsigned int c = 0; c < numChannels; ++c)
			{
				Channel *channel = static_cast<Channel*>(getChannel(c));
	
				//If the channel is NOT an aggregate channel, send the new signal downstream
				if(!(channel->isAggregate()))
				{
					if(channel->dsSignalQueueHasPending())
					{
						s_new.setCredit(channel->getNumItemsProduced());
					}
					else
					{
						s_new.setCredit(channel->dsPendingOccupancy());
					}
					pushSignal(s_new, channel);
				}
				else
				{
					//Subtract from the current node's enumeration region ID's reference count
					s_new.getRefCount()[0] -= 1;
				}
			}
		}

		//release 1 data item from queue
		if(IS_BOSS())
		{
			COUNT_ITEMS(1);
			queue.release(1);
		}

		//if had credit, subtract 1
		if(this->numSignalsPending() > 0)
		{
			this->currentCreditCounter -= 1;
		}

		//Reset dataCount and currentCount to 0
		dataCount = 0;
		currentCount = 0;
	}

	__syncthreads();

	bool dsSignalQueueFull = false;
	if(this->numSignalsPending() > 0)
	{
		dsSignalQueueFull = this->signalHandler();
	}

	__syncthreads();

	//Signal Queue is full, short circut here
	if(dsSignalQueueFull)
	{
	//	this->deactivate();
		this->activate();	//NEED TO RE-ENQUEUE OURSELVES
		return;
	}

	__syncthreads();

      }
      while(mynDSActive == 0 && queue.getOccupancy() > 0);

      if (IS_BOSS())
	{
	  //COUNT_ITEMS(nTotalConsumed);  // instrumentation
	  //queue.release(nTotalConsumed);
	  
	  nDSActive = mynDSActive;

	  if (queue.getOccupancy() == 0)
	  //if (nTotalConsumed == nToConsume)
	    {
	      // less than a full ensemble remains, or 0 if flushing
	      this->deactivate(); 
	      
	      if (isFlushing)
		{
		  // no more inputs to read -- force downstream nodes
		  // into flushing mode and activte them (if not
		  // already active).  Even if they have no input,
		  // they must fire once to propagate flush mode and
		  // activate *their* downstream nodes.
		  for (unsigned int c = 0; c < numChannels; c++)
		    {
		      NodeBase *dsNode = getChannel(c)->getDSNode();
		      dsNode->setFlushing();
		      dsNode->activate();
		    }
		}
	    }
	}
#if 0
      TIMER_START(input);

      Queue<T> &queue = this->queue; 
      DerivedNodeType *n = static_cast<DerivedNodeType *>(this);

      // # of items available to consume from queue
      unsigned int nToConsume = queue.getOccupancy();
      
      // unless we are flushing, round # to consume down to a multiple
      // of ensemble width.
      //if (!isFlushing)
	//nToConsume = (nToConsume / maxRunSize) * maxRunSize;
      
      // # total of items consumed from queue
      unsigned int nTotalConsumed = 0;

      // # total of items to consume from the queue
      //unsigned int nTotalToConsume = queue.getOccupancy();
      unsigned int nTotalToConsume = 0;

      //if (!isFlushing)
	//nTotalToConsume = (nToConsume / maxRunSize) * maxRunSize;

      unsigned int mynDSActive = 0;

      // True when a downstream signal queue is full, so we stop firing.
      bool dsSignalFull = false;
      
	//Perform SAFIrE scheduling while we have signals.
      while (this->numSignalsPending() > 0 && !dsSignalFull && mynDSActive == 0)
	{
	      //Step 1: Determine if we need to get the next object on the queue
	      // This SHOULD only be when the dataCount is equal to the currentCount
	      // Base case: Default of 0 is set for both dataCount and currentCount
	      // so will call on first run.
	      // All subsequent calls to get the next data element are done when we
	      // have exhausted all enumerated elements of that object.
	      // NOTE: Setting of parent in the fire function is only done in
	      // enumerate nodes.  All other setting of parents MUST be done
	      // in the signal handler.
	      if(dataCount == currentCount)
	      {
		//if(parentBuffer.getCapacity() - parentBuffer.getOccupancy() > 0)
		if(!(isParentBufferFull()))
		{
			unsigned int parentBase = parentBuffer.reserve(1);
			unsigned int refBase = refCounts.reserve(1);
			unsigned int offset = 0;

			const T &elt = queue.getElt(offset);
			const unsigned int &refelt = refCount;

			parentBuffer.putElt(parentBase, offset, elt);
			refCounts.putElt(refBase, offset, refelt);

			void* s;
			void* rc;

			s = parentBuffer.getVoidTail();
			rc = refCounts.getVoidTail();

			setCurrentParent(s);
			//setCurrentRefCount(rc);

			dataCount = this->findCount();
			currentCount = 0;
			//this->setParent(static_cast<void*>(queue.getElt(0)));
		}
	      }

	      // # of items already consumed from queue
	      unsigned int nConsumed = 0;
	      nToConsume = this->currentCreditCounter;

		//Can ignore flushing here, since we need to get to signal boundary.
	      //if (!isFlushing)
		//nToConsume = (this->currentCreditCounter / maxRunSize) * maxRunSize;

	      nTotalToConsume += nToConsume;

	      while (nConsumed < nToConsume && mynDSActive == 0)
		{
		  unsigned int nItems = min(nToConsume - nConsumed, maxRunSize);
		  
		  NODE_OCC_COUNT(nItems);
		  
		  const T &myData = 
		    (tid < nItems
		     //? queue.getElt(nConsumed + tid)
		     ? queue.getElt(nTotalConsumed + nConsumed + tid)
		     : queue.getDummy()); // don't create a null reference
		  
		  TIMER_STOP(input);
		  
		  TIMER_START(run);
	
		  if (runWithAllThreads || tid < nItems)
		    {
		      //n->run(myData);
		    }
		  nConsumed += nItems;
		  
		  TIMER_STOP(run);
		  
		  TIMER_START(output);
		  
		  for (unsigned int c = 0; c < numChannels; c++)
		    {
		      // check whether each channel's downstream node was activated
		      mynDSActive += getChannel(c)->moveOutputToDSQueue();
		    }
		  
		  TIMER_STOP(output);
		  
		  TIMER_START(input);
		}
		nTotalConsumed += nConsumed;	//nConsumed Should be the same as nToConsume here

		//Syncthreads before entering the signal handeler, need to make sure that every
		//thread knows the current consumed totals.
		__syncthreads();

		dsSignalFull = this->signalHandler();
	}

	//Use normal nConsumed and nToConsume values after signals are all handled.
      unsigned int nConsumed = 0;

      nToConsume = queue.getOccupancy();

      // unless we are flushing, round # to consume down to a multiple
      // of ensemble width.
      if (!isFlushing)
	nToConsume = (nToConsume / maxRunSize) * maxRunSize;

      nTotalToConsume += nToConsume;

	//Resume normal AFIE scheduling once we have no signals remaining.
      while (nConsumed < nToConsume && mynDSActive == 0 && !dsSignalFull)
	{
	      while (nConsumed < nToConsume && mynDSActive == 0)
		{
		  unsigned int nItems = min(nToConsume - nConsumed, maxRunSize);
		  
		  NODE_OCC_COUNT(nItems);
		  
		  const T &myData = 
		    (tid < nItems
		     //? queue.getElt(nConsumed + tid)
		     ? queue.getElt(nTotalConsumed + nConsumed + tid)
		     : queue.getDummy()); // don't create a null reference
		  
		  TIMER_STOP(input);
		  
		  TIMER_START(run);
	
		  if (runWithAllThreads || tid < nItems)
		    {
		      //n->run(myData);
		    }
		  nConsumed += nItems;
		  
		  TIMER_STOP(run);
		  
		  TIMER_START(output);
		  
		  for (unsigned int c = 0; c < numChannels; c++)
		    {
		      // check whether each channel's downstream node was activated
		      mynDSActive += getChannel(c)->moveOutputToDSQueue();
		    }
		  
		  TIMER_STOP(output);
		  
		  TIMER_START(input);
		}
	}
      nTotalConsumed += nConsumed;
      
	//Release items as normal.
      if (IS_BOSS())
	{
	  COUNT_ITEMS(nTotalConsumed);  // instrumentation
	  queue.release(nTotalConsumed);
	  
	  nDSActive = mynDSActive;

	  if (nTotalConsumed == nTotalToConsume)
	  //if (nTotalConsumed == nToConsume)
	    {
	      // less than a full ensemble remains, or 0 if flushing
	      this->deactivate(); 
	      
	      if (isFlushing)
		{
		  // no more inputs to read -- force downstream nodes
		  // into flushing mode and activte them (if not
		  // already active).  Even if they have no input,
		  // they must fire once to propagate flush mode and
		  // activate *their* downstream nodes.
		  for (unsigned int c = 0; c < numChannels; c++)
		    {
		      NodeBase *dsNode = getChannel(c)->getDSNode();
		      dsNode->setFlushing();
		      dsNode->activate();
		    }
		}
	    }
	}
      
      TIMER_STOP(input);
#endif
    }
  };
}  // end Mercator namespace

#endif
