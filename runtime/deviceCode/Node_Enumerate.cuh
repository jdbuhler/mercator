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
	nFrontierNodes(1),	// FIXME: get from compiler
	parentBuffer(queueSize),
	refCounts(queueSize)
    {}
    
  protected:

    using BaseType::getChannel;
    using BaseType::maxRunSize; 
    
    using BaseType::nDSActive;
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
      
      // # of items available to consume from queue
      unsigned int nToConsume = queue.getOccupancy();
      
      unsigned int nConsumed = 0;
      
      unsigned int mynDSActive = 0;
      
      while (nConsumed < nToConsume && mynDSActive == 0)
	{
	  ////////////////////////////////////////////
	  // FULL PARENT BUFFER CHECK
	  ////////////////////////////////////////////
	  
	  if (isParentBufferFull()) 
	    {
	      __shared__ bool parentBufferReleased;
	      if (IS_BOSS())
		{
		  printf("%d PARENT FULL\n", blockIdx.x);
		  
		  parentBufferReleased = false;
		  while (!parentBuffer.empty() && refCounts.getElt(0) != 0)
		    {
		      parentBuffer.dequeue();
		      refCounts.dequeue();
		      parentBufferReleased = true;
		    }
		  
		  //If we did not release from the parent buffer, make
		  //sure to re-activate ourselves and set the local
		  //flushing mode for our enumeration region.  Activate
		  //downstream node(s) if they are not already to
		  //propagate local flush and eventualy be able to free
		  //the parent buffer once revisited.
		  if (parentBufferReleased)
		    this->scheduler->removeLocalFlush(this->getEnumId());
		  else
		    {
		      // Re-enqueue and re-activate ourselves since we
		      // still have stuff to process
		      this->deactivate();
		      this->activate();
		      
		      for (unsigned int c = 0; c < numChannels; c++)
			{
			  NodeBase *dsNode = getChannel(c)->getDSNode();
			  
			  // Set the writeThru ID of the downstream nodes
			  // to that of this node
			  dsNode->setWriteThruId(this->getEnumId());
			  
			  // Activate the downstream nodes to pass the
			  // local flush mode
			  dsNode->activate();
			}
		      
		      // Set the local flushing flag for the region in
		      // the scheduler
		      this->scheduler->setLocalFlush(this->getEnumId());
		    }
		}
	      __syncthreads(); 
	      
	      // Short circut ourselves since we don't have any more
	      // space to put new parents
	      if (!parentBufferReleased)
		return;
	    }
	  
	  // Short circut here if there are not inputs to the node.
	  // This is the case when a block has NO INPUTS and is trying
	  // to propagate the flushing mode.
	  if (queue.empty())
	    break;
	  
	  ////////////////////////////////////////////
	  // EMIT ENUMERATE SIGNAL IF NEEDED
	  ////////////////////////////////////////////
	  
	  // If we've consumed all the data from the current elt
	  if (currentCount == dataCount)
	    { 
	      __syncthreads();
	      
	      // Get a new parent object from the data queue
	      if (IS_BOSS())
		{
		  T *currParent = &parentBuffer.enqueue(queue.dequeue());
		  currRefCount  = &refCounts.enqueue(nFrontierNodes);
		  
		  setCurrentParent(currParent);
		  
		  dataCount = this->findCount();
		  currentCount = 0;
		  
		  // Emit Enumerate Signal
		  
		  // Create new Enum signal to send downstream
		  Signal s_new(Signal::Enum);	
		  s_new.setParent(currParent);
		  
		  // Reserve space downstream for the new signal
		  for (unsigned int c = 0; c < numChannels; ++c)
		    {
		      auto *channel = getChannel(c);
		      
		      // If the channel is NOT an aggregate channel, send
		      // the new signal downstream
		      if (!channel->isAggregate())
			channel->pushSignal(s_new);
		    }
		  
		  // record that we consumed one item from input queue
		  COUNT_ITEMS(1);
		  if (this->numSignalsPending() > 0)
		    this->currentCreditCounter--;
		}
	    }
	  
	  __syncthreads();
	  	  
	  while (currentCount < dataCount && mynDSActive == 0)
	    {
	      unsigned int nItems = min(dataCount - currentCount, maxRunSize);
	      
	      NODE_OCC_COUNT(nItems);
	      
	      TIMER_STOP(input);
	      
	      TIMER_START(run);
	      
	      // No call to run here, just push out the ids up to
	      // dataCount.
	      if (tid < nItems)
		{
		  this->push(currentCount + tid);
		}
	      
	      TIMER_STOP(run);
	      
	      TIMER_START(output);
	      
	      for (unsigned int c = 0; c < numChannels; c++)
		{
		  // check whether each channel's downstream node was
		  // activated
		  mynDSActive += 
		    getChannel(c)->moveOutputToDSQueue(this->getWriteThruId());
		}
	      
	      TIMER_STOP(output);
	      
	      TIMER_START(input);
	      
	      __syncthreads();
	      if (IS_BOSS())
		currentCount += nItems;
	      __syncthreads();
	    }
	  
	  
	  if (currentCount == dataCount)
	    {
	      nConsumed++;
	      
	      //Emit Agg Signal
	      if (IS_BOSS())
		{
		  //Create new Agg signal to send downstream
		  Signal s_new(Signal::Agg);
		  s_new.setRefCount(currRefCount);
		  
		  //Reserve space downstream for the new signal
		  for (unsigned int c = 0; c < numChannels; ++c)
		    {
		      auto *channel = getChannel(c);
		      
		      //If the channel is NOT an aggregate channel,
		      //send the new signal downstream
		      if (!channel->isAggregate())
			{
			  channel->pushSignal(s_new);
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
	  

	  
#if 0
	  // FIXME: TERMPORARILY TURNED OFF because we aren't
	  // receiving signals at the input to enumerations right
	  // now. This needs to be rewritten to properly handle
	  // signals.

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
	      if (IS_BOSS())
		{
		  // forcibly re-enqueue ourselves
		  this->deactivate();
		  this->activate();
		}
	      
	      return;
	    }
#endif
	}
	
	
      if (IS_BOSS())
	{
	  nDSActive = mynDSActive;

	  if (nConsumed == nToConsume)
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
