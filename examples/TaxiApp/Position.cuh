#ifndef __POSITION_CUH
#define __POSITION_CUH

#include <cuda_runtime.h>

struct Position {
  
  __device__ __host__
  Position()
    : tag(0), latitude(0.0), longitude(0.0)
  {}
  
  __device__ __host__
  Position(unsigned int t, double lon, double lat) 
    : tag(t), latitude(lat), longitude(lon)
  {}
  
  unsigned int tag;
  double latitude;
  double longitude;
};

#endif
