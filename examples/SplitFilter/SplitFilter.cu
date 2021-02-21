//
// SPLITFILTER.CU
// Device-side splitting filter application
//
// MERCATOR
// Copyright (C) 2021 Washington University in St. Louis; all rights reserved.
//

#include "SplitFilter_dev.cuh"

__device__
uint32_t munge(uint32_t key)
{
  key = (key << 15) - key - 1;
  key = key ^ (key >> 12);
  key = key + (key << 2);
  key = key ^ (key >> 4);
  key = key * 2057;
  key = key ^ (key >> 16);
  return key;
}

//
// Hash each input item and then distribute the result to
// one of two output channels, depending on whether the
// hash is an odd or even number.
//
__MDECL__
void SplitFilter_dev::Filter<InputView>::
run(uint32_t const & inputItem, unsigned int nInputs)
{
  uint32_t v;
  
  if (threadIdx.x < nInputs)
    v = munge(inputItem);
  
  push(v, (threadIdx.x < nInputs), (v % 2 == 0 ? Out::accept : Out::reject));
}
