//
// EVENFILER.CU
// Device-side app to filter even numbers
//
// MERCATOR
// Copyright (C) 2021 Washington University in St. Louis; all rights reserved.
//

#include "EvenFilter_dev.cuh"

__device__
unsigned int munge(unsigned int key)
{
  key = (key << 15) - key - 1;
  key = key ^ (key >> 12);
  key = key + (key << 2);
  key = key ^ (key >> 4);
  key = key * 2057;
  key = key ^ (key >> 16);
  return key;
}

__MDECL__
void EvenFilter_dev::
filter<InputView>::run(const size_t& inputItem, unsigned int nInputs)
{
  unsigned int tid = threadIdx.x;
  unsigned int v;
  
  if (tid < nInputs)
    v = munge((unsigned int) inputItem);
  
  // If no channel is specified, push sends a value to the module's
  // first output channel.
  push(v, tid < nInputs && (v % 2) == 0); // defaults to pushing to Out::accept

}

