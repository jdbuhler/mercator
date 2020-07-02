#ifndef __NODE_ENUMERATE_CUH
#define __NODE_ENUMERATE_CUH

#include <cassert>

#include "Node.cuh"
#include "ChannelBase.cuh"
#include "Queue.cuh"

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
    
    unsigned int *currRefCount;
    
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
    
    //
    // @brief a check that returns whether or not the parent buffer is full.
    //
    // @return bool returns true if the parent buffer is currently full,
    // false otherwise
    //
    __device__
    bool isParentBufferFull() 
    {
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
	  
	  if (isParentBufferFull()) 
	    {
	      __shared__ bool parentBufferReleased;
	      if (IS_BOSS())
		{
		  parentBufferReleased = false;
		  for (unsigned int i = 0; i < parentBuffer.getCapacity(); ++i) 
		    {
		      if (refCounts.getElt(i) != 0) 
			{
			  if (i > 0) 
			    {
			      //Release all parents that were found to have 0
			      //references remaining.
			      refCounts.release(i);
			      parentBuffer.release(i);
			      parentBufferReleased = true;
			    }
			  break;
			}
		    }
	
		  //If we did not release from the parent buffer, make
		  //sure to re-activate ourselves and set the local
		  //flushing mode for our enumeration region.  Activate
		  //downstream node(s) if they are not already to
		  //propagate local flush and eventualy be able to free
		  //the parent buffer once revisited.
		  if (!parentBufferReleased) 
		    {
		      //Re-enqueue and re-activate ourselves since we
		      //still have stuff to process
		      this->deactivate();
		      this->activate();
		    
		      for (unsigned int c = 0; c < numChannels; c++)
			{
			  NodeBase *dsNode = getChannel(c)->getDSNode();
			
			  //Set the writeThru ID of the downstream nodes
			  //to that of this node
			  dsNode->setWriteThruId(this->getEnumId());
			
			  //Activate the downstream nodes to pass the
			  //local flush mode
			  dsNode->activate();
			}

		      //Set the local flushing flag for the region in
		      //the scheduler
		      this->scheduler->setLocalFlush(this->getEnumId());
		    }
		}
	    
	      //Make sure all threads see whether or not the parent
	      //buffer is still full.
	      __syncthreads(); 
	    
	      //Short circut ourselves since we don't have any more
	      //space to put new parents
	      if (!parentBufferReleased)
		return;

	      __syncthreads();
	      if (parentBufferReleased) 
		{
		  //Remove the local flush flag from the scheduler if we
		  //have space available in our parent buffer
		  if (IS_BOSS())
		    this->scheduler->removeLocalFlush(this->getEnumId());
		}
	      __syncthreads();
	    }

	  //Short circut here if there are not inputs to the node.
	  //This is the case when a block has NO INPUTS and is trying
	  //to propagate the flushing mode.
	  if (queue.empty())
	    break;
	  
	  ////////////////////////////////////////////
	  // EMIT ENUMERATE SIGNAL IF NEEDED
	  ////////////////////////////////////////////

	  // If we've consumed all the data from the current elt
	  if (dataCount == currentCount) 
	    { 
	      // Get a new parent object from the data queue
	      if (IS_BOSS())
		{
		  T *currParent = &parentBuffer.enqueue(queue.dequeue());
		  currRefCount  = &refCounts.enqueue(refCount);

		  setCurrentParent(currParent);
		  
		  dataCount = this->findCount();
		  currentCount = 0;
		  
		  //Emit Enumerate Signal
		  
		  //Create new Enum signal to send downstream
		  Signal s_new;
		  s_new.setTag(Signal::SignalTag::Enum);	
		  s_new.setParent(currParent);
		  
		  using Channel = typename BaseType::Channel<void*>;
		  
		  //Reserve space downstream for the new signal
		  for (unsigned int c = 0; c < numChannels; ++c)
		    {
		      Channel *channel = static_cast<Channel*>(getChannel(c));
		      
		      //If the channel is NOT an aggregate channel, send
		      //the new signal downstream
		      if (!channel->isAggregate())
			pushSignal(s_new, channel);
		    }
		  
		  // record that we consumed one item from input queue
		  COUNT_ITEMS(1);
		  if (this->numSignalsPending() > 0)
		    this->currentCreditCounter -= 1;		  
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
	      
	      //No call to run here, just push out the ids up to
	      //dataCount.
	      if (tid < nItems)
		{
		  this->push(currentCount + tid);
		}
	      nConsumed += nItems;

	      __syncthreads();

	      //Add the number of items we just sent downstream to the
	      //current counter
	      if (IS_BOSS())
		currentCount += nItems;
	      
	      __syncthreads();
	      
	      TIMER_STOP(run);
	      
	      TIMER_START(output);
	      
	      for (unsigned int c = 0; c < numChannels; c++)
		{
		  // check whether each channel's downstream node was
		  // activated
		  mynDSActive += 
		    getChannel(c)->moveOutputToDSQueue(this->getWriteThruId());
		}
	      
	      __syncthreads();
	      
	      TIMER_STOP(output);
	      
	      TIMER_START(input);
	    }
	  
	  __syncthreads();
	  
	  if (dataCount == currentCount)
	    {
	      //Emit Agg Signal
	      if (IS_BOSS())
		{
		  //Create new Agg signal to send downstream
		  Signal s_new;
		  s_new.setTag(Signal::SignalTag::Agg);	
		  s_new.setRefCount(currRefCount);
		  
		  using Channel = typename BaseType::Channel<void*>;
	
		  //Reserve space downstream for the new signal
		  for (unsigned int c = 0; c < numChannels; ++c)
		    {
		      Channel *channel = static_cast<Channel*>(getChannel(c));

		      //If the channel is NOT an aggregate channel,
		      //send the new signal downstream
		      if (!channel->isAggregate())
			{
			  pushSignal(s_new, channel);
			}
		      else
			{
			  //Subtract from the current node's
			  //enumeration region ID's reference count
			  (*s_new.getRefCount())--;
			}
		    }
		}
	    }
	  
	  __syncthreads();
	  
	  bool dsSignalQueueFull = false;
	  if (this->numSignalsPending() > 0)
	    {
	      dsSignalQueueFull = this->signalHandler();
	    }
	  
	  __syncthreads();
	  
	  //Signal Queue is full, short circut here
	  if (dsSignalQueueFull)
	    {
	      this->activate();	//NEED TO RE-ENQUEUE OURSELVES
	      return;
	    }
	  
	  __syncthreads();
	  
	}
      while (mynDSActive == 0 && queue.getOccupancy() > 0);
      
      if (IS_BOSS())
	{
	  nDSActive = mynDSActive;

	  if (queue.getOccupancy() == 0)
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
