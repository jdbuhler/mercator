#ifndef __PIPE_ELT_T
#define __PIPE_ELT_T

#include <cuda_runtime.h>

// Type of data inputs for the SameTypePipe and DiffTypePipe apps
class PipeEltT  {     

private:

  // element ID number
  int ID;
  
  // number of iterations of dummy work to do
  int workIters;
  
  // dummy results, printed from server to ensure 
  //  dummy work is not optimized out
  float result;
  
  // useful for counting loops in self-loop topology
  int loopCount;
  
public:
  
  __host__ __device__ PipeEltT()
  { }
  
  __host__ __device__ PipeEltT(int iID, int iworkIters, int iloopCount = 0)
    : ID(iID),
      workIters(iworkIters), 
      result(0.0),
      loopCount(iloopCount)
  { }
  
  __host__ __device__ int get_ID() const 
  { return ID; }
  
  __host__ __device__ void set_ID(int newID) 
  { ID = newID; }
  
  __host__ __device__ int get_workIters() const 
  { return workIters; }
  
  __host__ __device__ void set_workIters(int k) 
  { workIters = k; }

  __host__ __device__ float get_floatResult() const
  { return result; }
  
  __host__ __device__ void set_floatResult(float f) 
  { result = f; }
  
  __host__ __device__ int get_loopCount() const 
  { return loopCount; }
  
  __host__ __device__ void dec_loopCount() 
  { if (loopCount > 0) loopCount--; }
  
  __host__ __device__ void inc_loopCount() 
  { loopCount++; }
};

#endif
