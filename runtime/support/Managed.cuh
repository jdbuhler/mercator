#ifndef __MANAGED_CUH
#define __MANAGED_CUH

/**
 * @file Managed.cuh
 * @brief Defines class to handle CUDA-managed mem.
 *
 * NB: Based on Mark Harris's Par4All blog entry on cudaMallocManaged.
 *
 */

#include <cstdlib>
#include <new>

#define CALLED_FROM_DEV defined(__CUDA_ARCH__)

/**
 * @class CudaManaged
 * @brief Defines class to handle CUDA-managed mem.
 *
 * NB: Based on Mark Harris's Par4All blog entry on cudaMallocManaged.
 *
 * Idea: if called from device, use standard allocs/frees;
 * if from host, use managed mem
 */
class CudaManaged {
public:
  /**
     
   * @brief Replacement for classic 'new' operator.
   * 
   * @param len Num bytes requested
   * @return void* Pointer to allocated mem
   *
   */
  __host__ __device__
  void *operator new(size_t len) 
  {
    void *ptr;
    
#if CALLED_FROM_DEV
    ptr = malloc(len);
#else
    cudaMallocManaged(&ptr, len);
#endif
    
    return ptr;
  }
  
  /**
   * @brief Replacement for classic 'delete' operator.
   *
   * @param ptr Pointer to mem to be released
   */
  __host__ __device__
  void operator delete(void *ptr) 
  {
#if CALLED_FROM_DEV
    free(ptr);
#else
    cudaFree(ptr);
#endif
  }
};

#endif
