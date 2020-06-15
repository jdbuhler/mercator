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
    
   //
   // @brief find the number of data items that need to be enumerated
   // from the current parent object.
   //
   // Like the run function, this is filled out by the user in the
   // generated code.  This function is called SINGLE THREADED.
   //
   __device__
   virtual
   unsigned int findCount() {
	assert(false && "FindCount base called");
	return 0;
   }

   //
   // @brief a check that returns whether or not the parent buffer is full.
   //
   // @return bool returns true if the parent buffer is currently full,
   // false otherwise
   //
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

      //Short circut here if there are not inputs to the node.
      //This is the case when a block has NO INPUTS and is trying to
      //propagate the flushing mode.
      //if(queue.getOccupancy() == 0) {
	//break;
      //}

      ////////////////////////////////////////////
      // FULL PARENT BUFFER CHECK
      ////////////////////////////////////////////
      if(isParentBufferFull()) {
	__shared__ bool parentBufferReleased;
	if(IS_BOSS())
	{
		parentBufferReleased = false;
		for(unsigned int i = 0; i < parentBuffer.getCapacity(); ++i) {
			if(refCounts.getElt(i) != 0) {
				if(i > 0) {
					//Release all parents that were found to have 0 references remaining.
					refCounts.release(i);
					parentBuffer.release(i);
					parentBufferReleased = true;
				}
				break;
			}
		}
	
		//If we did not release from the parent buffer, make sure to re-activate ourselves and
		//set the local flushing mode for our enumeration region.  Activate downstream node(s)
		//if they are not already to propagate local flush and eventualy be able to free the
		//parent buffer once revisited.
		if(!parentBufferReleased) {
			this->deactivate();
			this->activate();	//Re-enqueue and re-activate ourselves since we still have stuff to process
	
			for (unsigned int c = 0; c < numChannels; c++)
			{
				NodeBase *dsNode = getChannel(c)->getDSNode();
				dsNode->setWriteThruId(this->getEnumId());	//Set the writeThru ID of the downstream nodes to that of this node
				dsNode->activate();				//Activate the downstream nodes to pass the local flush mode
			}

			this->scheduler->setLocalFlush(this->getEnumId());	//Set the local flushing flag for the region in the scheduler
		}
	}
	__syncthreads();	//Make sure all threads see whether or not the parent buffer is still full.
	if(!parentBufferReleased)
		return;		//Short circut ourselves since we don't have any more space to put new parents

	__syncthreads();
	if(parentBufferReleased) {
		//Remove the local flush flag from the scheduler if we have space available in our parent buffer
		if(IS_BOSS()) {
			this->scheduler->removeLocalFlush(this->getEnumId());
		}
	}
	__syncthreads();
      }

      //Short circut here if there are not inputs to the node.
      //This is the case when a block has NO INPUTS and is trying to
      //propagate the flushing mode.
      if(queue.getOccupancy() == 0) {
	break;
      }

      ////////////////////////////////////////////
      // EMIT ENUMERATE SIGNAL IF NEEDED
      ////////////////////////////////////////////
      if(dataCount == 0) { //Need to NOT look at element here unconditionally, will be garbage value if there is no data in the queue
	//Get a new parent object from the data queue
	if(IS_BOSS())
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

		//Emit Enumerate Signal

		//Create new Enum signal to send downstream
		Signal s_new;
		s_new.setTag(Signal::SignalTag::Enum);	
		s_new.setParent(parentBuffer.getVoidTail());	//Set the parent for the new signal
		setCurrentParent(queue.getVoidTail());	//Set the new parent for the node

		using Channel = typename BaseType::Channel<void*>;
	
		//Reserve space downstream for the new signal
		for(unsigned int c = 0; c < numChannels; ++c)
		{
			Channel *channel = static_cast<Channel*>(getChannel(c));
	
			//If the channel is NOT an aggregate channel, send the new signal downstream
			if(!(channel->isAggregate()))
			{
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
	  __syncthreads();
	  
	  NODE_OCC_COUNT(nItems);
	  
	  TIMER_STOP(input);
	  
	  TIMER_START(run);

	  //No call to run here, just push out the ids up to dataCount.
	  if (tid < nItems)
	    {
	      this->push(currentCount + tid);
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
			//Create new Agg signal to send downstream
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
		      dsNode->setFlushing(true);
		      dsNode->activate();
		    }
		  nDSActive = numChannels;
		  this->setFlushing(false);
		}
	    }
	}
    }
  };
}  // end Mercator namespace

#endif
