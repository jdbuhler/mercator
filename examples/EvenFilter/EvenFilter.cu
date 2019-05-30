//
// EVENFILER.CU
// Device-side app to filter even numbers
//
// MERCATOR
// Copyright (C) 2018 Washington University in St. Louis; all rights reserved.
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

//
// Hash each input item and return only those hash values that
// are even numbers.
//
__device__
void EvenFilter_dev::
Filter::run(const unsigned int& inputItem, InstTagT nodeIdx)
{
  
  unsigned int v = munge(inputItem);
  for (int i=0; i<1000; i++){
    v = munge(v);
  }
  // If no channel is specified, push sends a value to the module's
  // first output channel.
  if (v % 2 == 0)
    push(v, nodeIdx); // eqv to "push(v, nodeIdx, Out::accept);"
}
