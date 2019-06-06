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

#define UPPERBOUND 750000
#define LOWERBOUND 700000 

__device__
void AsyncFilter_dev::
Filter::init()
{
 //set upperbound for data collection
  if(IS_BOSS()){
    setFGContainerBounds((unsigned long long)LOWERBOUND, (unsigned long long)UPPERBOUND);
    }
  __syncthreads(); // all threads must see updates to the bounds
  
}

//
// Hash each input and emit only hashes that are 0 modulo the user's
// modulus
//
__device__
void AsyncFilter_dev::
Filter::run(const unsigned int& inputItem, InstTagT nodeIdx)
{
  unsigned int v = munge(inputItem);
  for(int i=0; i<10000;i++){
    v = munge(v);
  } 
  if (v % getParams()->modulus == 0)
    push(v, nodeIdx);
}
