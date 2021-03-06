#ifndef __PARENTBUFFER_CUH
#define __PARENTBUFFER_CUH

#include <cassert>
#include <climits>

#include "NodeBase.cuh"

namespace Mercator  {

  //
  // An arena allocator whose values are reference-counted.  We
  // maintain an external free list for simplicity rather than
  // chaining free entries together with internal pointers.
  // 
  // The arena takes as an (optional) argument a poitner to a
  // NodeBase, its presumable owner.  When the arena goes from full to
  // non-full, we call back to the owner to let it know that it should
  // unblock if it blocked when the buffer filled.
  //
  
  class RefCountedArena {
    
  public:
    
    static const unsigned int NONE = UINT_MAX;
    
    ///////////////////////////////////////////////////////
    // INIT/CLEANUP KERNEL FUNCIIONS
    ///////////////////////////////////////////////////////

    __device__
    RefCountedArena(unsigned int size)
      : size(size),
	freeList(new unsigned int [size]),
	refCounts(new unsigned int [size]),
	blockingNode(nullptr)
    {
      for (unsigned int j = 0; j < size; j++)
	freeList[j] = j;
      
      freeListSize = size;
    }

    __device__
    void setBlockingNode(NodeBase *iblockingNode)
    { blockingNode = iblockingNode; }
    
    __device__
    virtual ~RefCountedArena()
    {
      delete [] refCounts;
      delete [] freeList;
    }
    
    ///////////////////////////////////////////////////////
    
    // @brief true iff every entry in the buffer is in use
    __device__
    bool isFull() const
    { return (freeListSize == 0); }
    
    
    // @brief allocate a free entry in the buffer and return its
    // index.  The entry starts with a reference count of 1.
    __device__
    unsigned int alloc(unsigned int initialRefCount)
    {
      assert(IS_BOSS());
      
      assert(freeListSize > 0);
      
      unsigned int idx = freeList[--freeListSize];
      
      refCounts[idx] = initialRefCount;
      
      return idx;
    }
    
    // @brief decrement the reference count of entry idx by 1. Free it
    // if the count goes to 0.
    __device__
    void unref(unsigned int idx)
    {
      assert(IS_BOSS());
      
      assert(idx != NONE && idx < size);
      assert(refCounts[idx] > 0);
	  
      if (--refCounts[idx] == 0)
	{
	  freeList[freeListSize++] = idx;
	  
	  // unblock our node if we've cleared a significant amount of
	  // space -- we could do this even if only a single slot
	  // opens in the freelist, but a little hysteresis here
	  // should prevent frequent "flapping" between nearly blocked
	  // and unblocked status.
	  if (freeListSize > size/2 &&
	      blockingNode->isBlocked())
	    blockingNode->unblock();	  
	}
    }
    
  private:
    
    const unsigned int size;       // number of allocated entries
    
    unsigned int* const freeList;  // array listing all free entries
    unsigned int* const refCounts; // reference counts for allocated entries
    
    NodeBase* blockingNode;    // node that will block if arena fills
    
    unsigned int freeListSize; // # of entries on free list
  };
  
  
  //
  // a ParentBuffer is a reference-counted arena that allocates
  // storage of a particular type T for objects.
  //
  
  template <class T>
  class ParentBuffer : public RefCountedArena {
  public:
    
    ///////////////////////////////////////////////////////
    // INIT/CLEANUP KERNEL FUNCIIONS
    ///////////////////////////////////////////////////////
    
    __device__
    ParentBuffer(unsigned int size)
      : RefCountedArena(size),
	data(new T [size])
    {}
    
    __device__
    ~ParentBuffer()
    { delete [] data; }
    
    ////////////////////////////////////////////////////////

    //
    // @brief allocate an entry in the buffer with nrefs references
    // and set it to the item v.  Return the index of the newly
    // allocated entry.
    //
    __device__
    unsigned int alloc(const T &v, unsigned int initialRefCount)
    {
      assert(IS_BOSS());
      
      unsigned int idx = RefCountedArena::alloc(initialRefCount);
      data[idx] = v;
      return idx;
    }
    
    //
    // @brief get a pointer to an item from its index in the buffer.
    //
    __device__
    T *get(unsigned int idx) const
    { return &data[idx]; }
    
  private:
    
    T* const data;
  };
  
} // namespace Mercator

#endif
