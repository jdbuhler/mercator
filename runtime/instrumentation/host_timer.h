#ifndef __GPU_TIMER_H
#define __GPU_TIMER_H

//
// HOST_TIMER.CUH
// Host-side timers based on CUDA events
//
// MERCATOR
// Copyright (C) 2018 Washington University in St. Louis; all rights reserved.
//

#include "support/util.cuh"

class GpuTimer
{
public:
  
  GpuTimer()
  {
    cudaEventCreate(&_start);
    cudaEventCreate(&_stop);
  }
  
  ~GpuTimer()
  {
    cudaEventDestroy(_start);
    cudaEventDestroy(_stop);
  }
  
  void start(cudaStream_t stream = 0)
  {
    cudaEventRecord(_start, stream);
  }
  
  void stop(cudaStream_t stream = 0)
  {
    cudaEventRecord(_stop, stream);
  }
  
  float elapsed()
  {
    float elapsed;
    
    cudaEventSynchronize(_stop); // wait for stop event to finish
    gpuErrchk( cudaEventElapsedTime(&elapsed, _start, _stop) );
    
    return elapsed;
  }
  
private:
  
  cudaEvent_t _start;
  cudaEvent_t _stop;
};

#endif
