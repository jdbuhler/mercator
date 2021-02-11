//
// SYNTHGAINS.CU
// Device-side app to test different streaming behavior.
//
// MERCATOR
// Copyright (C) 2021 Washington University in St. Louis; all rights reserved.
//

#include "SynthGain_dev.cuh"
/*
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
*/

__MDECL__
void SynthGain_dev::
type1<InputView>::run(const size_t& inputItem, unsigned int nInputs)
{
  unsigned int tid = threadIdx.x;
  //unsigned int v;

  auto params = getParams();
  float g = params->avgGain;
  unsigned int fullG = (unsigned int)(g);
  float partG = g - fullG;

  unsigned int totalOut = 0;
  //int loopOut = fullG;
  //if(partG > 0)
    //loopOut += 1;
  
  if (tid < nInputs) {
    totalOut = fullG;
    if (partG > 0.0) {
      totalOut += ((unsigned int)(nInputs * partG) <= tid ? 1 : 0);
    }
    //totalOut += 2;

    //v = munge((unsigned int) inputItem);

  }
  
  // If no channel is specified, push sends a value to the module's
  // first output channel.
  //push(v, tid < nInputs && (v % 2) == 0); // defaults to pushing to Out::accept
  //for(unsigned int i = 0; i < loopOut; ++i)

  using R = Mercator::BlockReduce<unsigned int, THREADS_PER_BLOCK>;
  auto &bcast = Mercator::broadcast<unsigned int, THREADS_PER_BLOCK>;
  
  unsigned int blockMaxHits = R::max(totalOut);
  blockMaxHits = bcast(blockMaxHits, 0);

  for(unsigned int i = 0; i < blockMaxHits; ++i)
    push(1, i < totalOut); // defaults to pushing to Out::accept

}

