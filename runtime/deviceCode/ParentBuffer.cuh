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
    RefCountedArena(unsigned int isize,
		    NodeBase *iblockingNode = nullptr)
      : size(isize),
	blockingNode(iblockingNode)
    {
      freeList = new unsigned int [size];
      for (unsigned int j = 0; j < size; j++)
	freeList[j] = j;
      freeListSize = size;
      
      refCounts = new unsigned int [size];
    }
    
    __device__
    virtual ~RefCountedArena()
    {
      delete [] refCounts;
      delete [] freeList;
    }
    
    ///////////////////////////////////////////////////////
    
    // true iff every entry in the buffer is in use
    __device__
    bool isFull() const
    { return (freeListSize == 0); }
    
    
    // Allocate a free entry in the buffer and return its index.  The
    // entry starts with a reference count of 1.
    __device__
    unsigned int alloc()
    {
      assert(IS_BOSS());
      
      assert(freeListSize > 0);
      
      unsigned int idx = freeList[--freeListSize];
      
      refCounts[idx] = 1;
      
      return idx;
    }

    // Increment the reference count of entry idx by 1.
    __device__
    void ref(unsigned int idx)
    {
      assert(IS_BOSS());
      
      assert(idx < size || idx == NONE);
      
      if (idx != NONE)
	++refCounts[idx];
    }
    
    // Decrement the reference count of entry idx by 1. Free it 
    // if the count goes to 0.
    __device__
    void unref(unsigned int idx)
    {
      assert(IS_BOSS());
      
      assert(idx < size || idx == NONE);
      
      if (idx != NONE)
	{
	  assert(refCounts[idx] > 0);
	  
	  if (--refCounts[idx] == 0)
	    {
	      freeList[freeListSize++] = idx;
	      if (freeListSize == 1 &&  // buffer was full, now is not
		  blockingNode != nullptr)
		blockingNode->unblock();	  
	    }
	}
    }
    
  private:
    
    const unsigned int size;         // number of allocated entries
    NodeBase* const blockingNode;    // node that will block if arena fills
    
    unsigned int *freeList;    // array listing all free entries
    unsigned int freeListSize; // # of entries on free list
    
    unsigned int *refCounts;   // reference counts for allocated entries
    
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
    ParentBuffer(unsigned int size,
		 NodeBase *blockingNode = nullptr)
      : RefCountedArena(size, blockingNode)
    { data = new T [size]; }
    
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
    unsigned int alloc(const T &v)
    {
      assert(IS_BOSS());
      
      unsigned int idx = RefCountedArena::alloc();
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
    
    T *data;
  };
  
} // namespace Mercator

#endif
