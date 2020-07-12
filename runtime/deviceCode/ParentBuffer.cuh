#ifndef __PARENTBUFFER_CUH
#define __PARENTBUFFER_CUH

#include <cassert>

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
    
    // A Handle is a pointer to an element of an arena that
    // allows the reference count to be adjusted
    class Handle {
    public:
      
      __device__
      Handle()
      {}
      
      __device__
      Handle(RefCountedArena *iarena, unsigned int iidx)
	: arena(iarena), idx(iidx)
      {}
      
      //
      // @brief get the arena that the Handle points to
      //
      __device__
      RefCountedArena *getArena() const { return arena; }
      
      //
      // @brief get the index of the element within the arena
      //
      __device__
      unsigned int getIdx() const { return idx; }

      //
      // @brief add a reference to the element that the handle
      // points to.
      //
      __device__
      void ref() const { arena->ref(idx); }
      
      //
      // @brief remove a reference to the element that the handle
      // points to, effectively invalidating it.
      //
      __device__
      void unref() const { arena->unref(idx); }

    private:
      
      RefCountedArena *arena; // memory area that Handle points to
      unsigned int idx;       // index of object in arena
    };
    
    
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
    
    // true iff every entry in the buffer is in use
    __device__
    bool isFull() const
    { return (freeListSize == 0); }
    
    
    // Allocate a free entry in the buffer and return a handle to
    // it. The entry starts with a reference count of 1.
    __device__
    Handle alloc()
    {
      assert(IS_BOSS());
      
      assert(freeListSize > 0);
      
      unsigned int idx = freeList[--freeListSize];
      
      refCounts[idx] = 1;
      
      return Handle(this, idx);
    }
    
  private:
    
    unsigned int size;         // number of allocated entries
    
    unsigned int *freeList;    // array listing all free entries
    unsigned int freeListSize; // # of entries on free list
    
    unsigned int *refCounts;   // reference counts for allocated entries
    
    NodeBase *blockingNode;    // node that will block if arena fills
    
    // Increment the reference count of entry idx by 1.
    __device__
    void ref(unsigned int idx)
    {
      assert(IS_BOSS());
      
      assert(idx < size);
      ++refCounts[idx];
    }
    
    // Decrement the reference count of entry idx by 1. Free it 
    // if the count goes to 0.
    __device__
    void unref(unsigned int idx)
    {
      assert(IS_BOSS());
      
      assert(idx < size);
      assert(refCounts[idx] > 0);
      
      if (--refCounts[idx] == 0)
	{
	  freeList[freeListSize++] = idx;
	  if (freeListSize == 1 &&  // buffer was full, now is not
	      blockingNode != nullptr)
	    blockingNode->unblock();	  
	}
    }
  };
  
  
  //
  // a ParentBuffer is a reference-counted arena that allocates
  // storage of a particular type T for objects.
  //
  
  template <class T>
  class ParentBuffer : public RefCountedArena {
  public:
    
    __device__
    ParentBuffer(unsigned int size,
		 NodeBase *blockingNode = nullptr)
      : RefCountedArena(size, blockingNode)
    { data = new T [size]; }
    
    __device__
    ~ParentBuffer()
    { delete [] data; }
    
    //
    // @brief allocate an entry in the buffer with nrefs references
    // and set it to the item v.  Return a handle to the newly
    // allocated entry.
    //
    
    __device__
    RefCountedArena::Handle alloc(const T &v)
    {
      assert(IS_BOSS());
      
      Handle h = RefCountedArena::alloc();
      data[h.getIdx()] = v;
      return h;
    }
    
    //
    // @brief get a poitner to an item from its handle.
    //
    __device__
    T *get(const RefCountedArena::Handle &h) const
    { return &data[h.getIdx()]; }
    
  private:
    
    T *data;
  };
  
} // namespace Mercator

#endif
