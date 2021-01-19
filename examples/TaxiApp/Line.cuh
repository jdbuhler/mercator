#ifndef __LINE_CUH
#define __LINE_CUH

#include <cuda_runtime.h>

struct Line {
  
  __device__ __host__
  Line () 
  {}
  
  __device__ __host__
  Line (unsigned int t, unsigned int s, unsigned int len) 
    : start(s), length(len), tag(t)
  {}
  
  unsigned int start;
  unsigned int length;
  unsigned int tag;
};

#endif
