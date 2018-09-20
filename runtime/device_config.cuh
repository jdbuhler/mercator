#ifndef __DEVICE_CONFIG_CUH
#define __DEVICE_CONFIG_CUH

//
// @file device_config.cuh
// @brief Device-side parameters and macros
//
// MERCATOR
// Copyright (C) 2018 Washington University in St. Louis; all rights reserved.
//

// number of threads in a warp
#define WARP_SIZE 32

// macros to identify 0th thread/block
#define IS_BOSS()       (threadIdx.x == 0)
#define IS_BOSS_BLOCK() (blockIdx.x  == 0)

__device__ __forceinline__
void crash()
{
  asm("trap;");
}

#endif
