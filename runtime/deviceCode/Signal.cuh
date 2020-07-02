#ifndef __SIGNAL_CUH
#define __SIGNAL_CUH

class Signal {
private:
public:
  enum SignalTag {Enum, Agg};

private:
  int credit;
  
  void* parent;
  unsigned int* refCount;
  SignalTag tag;
  
public:
  __device__ 
  Signal() 
    : credit(0) 
  { }
  
  __device__ ~Signal() 
  { }
  
  __device__ 
  void setCredit(int c) 
  { credit = c; }
  
  __device__ 
  int getCredit() const 
  { return credit; }
  
  __device__ 
  void setTag(SignalTag t) 
  { tag = t; }
  
  __device__ 
  SignalTag getTag() const 
  { return tag; }
  
  __device__ 
  void setParent(void* p) 
  { parent = p; }
  
  __device__
  void *getParent() const 
  { return parent; }
  
  __device__ 
  void setRefCount(unsigned int* rc) 
  { refCount = rc; }
  
  __device__ 
  unsigned int* getRefCount() const
  { return refCount; }
};

#endif
