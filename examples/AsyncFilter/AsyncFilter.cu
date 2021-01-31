//
// ASYNCFILTER.CU
// Device-side code for async filter example
// (there is nothing asynchronous about this code; all the magic
//  happens in the host-side driver)
//
// MERCATOR
// Copyright (C) 2018 Washington University in St. Louis; all rights reserved.
//

#include "AsyncFilter_dev.cuh"

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
// Hash each input and emit only hashes that are 0 modulo the user's
// modulus
//
__MDECL__
void AsyncFilter_dev::
Filter<InputView>::run(unsigned int const & inputItem)
{
  unsigned int v = munge(inputItem);
  
  if (v % getModuleParams()->modulus == 0)
    push(v);
}
