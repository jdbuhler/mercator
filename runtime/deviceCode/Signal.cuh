#ifndef __SIGNAL_CUH
#define __SIGNAL_CUH

#include "ParentBuffer.cuh"

namespace Mercator {
  
  //
  // A Signal is a polymorphic tagged type that contains information
  // to be passed between nodes.  FIXME: we should really make this
  // a tagged union so that the size is not the sum of sizes for
  // fields associated with each type.
  //
  
  struct Signal {
    
    enum SignalTag {Invalid, Enum, Agg};
    
    SignalTag tag;   
    int credit;
    
    // fields for Enum
    RefCountedArena::Handle handle;
    
    // fields for Agg
    // (no fields needed)
    
    __device__
    Signal()
      : tag(Invalid)
    { }
    
    __device__ 
    Signal(SignalTag itag) 
      : tag(itag), credit(0)
    { }
  };
  
  // max # of signals produced by a node consuming a vector of data
  const unsigned int MAX_SIGNALS_PER_VEC = 2;
  
  // max # of signal produced by a node consuming one signal
  const unsigned int MAX_SIGNALS_PER_SIG = 1;
  
  // max # of signals produced by one pass through a node's run loop
  const unsigned int MAX_SIGNALS_PER_RUN =
    MAX_SIGNALS_PER_VEC + MAX_SIGNALS_PER_SIG;
  
} // namespace Mercator

#endif
