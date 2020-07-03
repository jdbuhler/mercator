#ifndef __SIGNAL_CUH
#define __SIGNAL_CUH

class Signal {
private:
public:
  enum SignalTag {Invalid, Enum, Agg};

private:
  
  SignalTag tag;
  
  void* parent;
  unsigned int* refCount;
  
  int credit;
  
public:
  __device__
  Signal()
    : tag(Invalid)
  { }
  
  __device__ 
  Signal(SignalTag itag) 
    : tag(itag), credit(0) 
  { }
  
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
  
  __device__ 
  void setCredit(int c) 
  { credit = c; }
  
  __device__ 
  int getCredit() const 
  { return credit; }
  
};

#endif
