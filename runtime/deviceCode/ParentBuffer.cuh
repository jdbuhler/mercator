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
      
      __device__
      RefCountedArena *getArena() const { return arena; }
      
      __device__
      unsigned int getIdx() const { return idx; }
      
      __device__
      void unref() { arena->unref(idx); }
      
    private:
      
      RefCountedArena *arena;
      unsigned int idx;
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
    // it. The entry starts with a reference count of nrefs.
    __device__
    Handle alloc(unsigned int nrefs)
    {
      assert(IS_BOSS());
      
      assert(freeListSize > 0);
      
      unsigned int idx = freeList[--freeListSize];
      
      refCounts[idx] = nrefs;
      
      return Handle(this, idx);
    }
    
  private:
    
    unsigned int size;         // number of allocated entries
    
    unsigned int *freeList;    // array listing all free entries
    unsigned int freeListSize; // # of entries on free list
    
    unsigned int *refCounts;   // reference counts for allocated entries
    
    NodeBase *blockingNode;    // node that will block if arena fills
    
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
    
    __device__
    RefCountedArena::Handle alloc(const T &v, unsigned int nrefs)
    {
      assert(IS_BOSS());
      
      Handle h = RefCountedArena::alloc(nrefs);
      data[h.getIdx()] = v;
      return h;
    }
    
    __device__
    T *get(const RefCountedArena::Handle &h) const
    { return &data[h.getIdx()]; }
    
  private:
    
    T *data;
  };
  
} // namespace Mercator

#endif
