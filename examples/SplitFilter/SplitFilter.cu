//
// SPLITFILTER.CU
// Device-side splitting filter application
//
// MERCATOR
// Copyright (C) 2018 Washington University in St. Louis; all rights reserved.
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
__device__
void SplitFilter_dev::
Filter::run(const uint32_t& inputItem, InstTagT nodeIdx)
{
  uint32_t v = munge(inputItem);
  for(int i=0; i<1000;i++){
    v = munge(v);
  } 
  push(v, nodeIdx, (v % 2 == 0 ? Out::accept : Out::reject));
}
