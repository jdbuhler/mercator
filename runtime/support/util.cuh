#ifndef __UTIL_H
#define __UTIL_H

//
// UTIL.H
// Host-side utilities
//
// MERCATOR
// Copyright (C) 2018 Washington University in St. Louis; all rights reserved.
//

#include <iostream>
#include <cstdlib>

// GPU assert with useful context
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, 
		      const char *file, 
		      int line, 
		      bool abort=true)
{
  if (code != cudaSuccess) 
    {
      std::cerr << "GPUassert: "
		<< cudaGetErrorString(code) << ' '
		<< file << ':' << line << std::endl;
      
      if (abort) 
	exit(code);
    }
}

#endif
