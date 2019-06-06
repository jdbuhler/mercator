//
// MODFILTER.CU
// Device-side general modulo filtering application
//
// MERCATOR
// Copyright (C) 2018 Washington University in St. Louis; all rights reserved.
//

#include "ModFilter_dev.cuh"

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
void ModFilter_dev::
Filter::init()
{
 //set upperbound for data collection
  if(IS_BOSS()){
    setFGContainerBounds((unsigned long long)LOWERBOUND, (unsigned long long)UPPERBOUND);
    }
  __syncthreads(); // all threads must see updates to the bounds
  
}


//
// Hash each input item and emit the hash iff it is zero modulo
// the current node's modulus value.  This code is shared by all
// nodes of type Filter.
//
// Note that one call to run() may process inputs to *different
// nodes* in different GPU threads, so we need to use the per-thread
// nodeIdx variable to make sure we get the modulus parameter for the
// right node in each thread.
//
__device__
void ModFilter_dev::
Filter::run(const unsigned int& inputItem, InstTagT nodeIdx)
{
  unsigned int v = munge(inputItem);
  for(int i=0; i<10000;i++){
    v = munge(v);
  } 
  
  if (v % getParams()->modulus[nodeIdx] == 0)
    push(v, nodeIdx);
}
