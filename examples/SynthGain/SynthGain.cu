//
// SYNTHGAINS.CU
// Device-side app to test different streaming behavior.
//
// MERCATOR
// Copyright (C) 2021 Washington University in St. Louis; all rights reserved.
//

#include "SynthGain_dev.cuh"

//Set the number of iterations of extra work for each input.
#define ITERS 1000

//Function for doing extra work on every output.
__device__
size_t extra_work(size_t key)
{
  for(unsigned int i = 0; i < ITERS; ++i) {
  key = (key << 15) - key - 1;
  key = key ^ (key >> 12);
  key = key + (key << 2);
  key = key ^ (key >> 4);
  key = key * 2057;
  key = key ^ (key >> 16);
  }
  return key;
}

__MDECL__
void SynthGain_dev::
type1<InputView>::run(const size_t& inputItem, unsigned int nInputs)
{
  unsigned int tid = threadIdx.x;

  auto params = getParams();
  float g = params->avgGain;
  unsigned int fullG = (unsigned int)(g);
  float partG = g - fullG;

  unsigned int totalOut = 0;
  
  //Determine which threads should produce outputs.
  if (tid < nInputs) {
    totalOut = fullG;

    //Determine which threads should produce an extra output, when there is a partial gain.
    if (partG > 0.0) {
      totalOut += ((unsigned int)(nInputs * partG) <= tid ? 0 : 1);
    }
  }

  using R = Mercator::BlockReduce<unsigned int, THREADS_PER_BLOCK>;
  auto &bcast = Mercator::broadcast<unsigned int, THREADS_PER_BLOCK>;
  
  unsigned int blockMaxHits = R::max(totalOut);
  blockMaxHits = bcast(blockMaxHits, 0);

  //DEBUG
  //printf("[%d, %d] g=%lf\tfullG=%d\tpartG=%lf\ttotalOut=%d\tblockMaxHits=%d\tnInputs*partG=%d\tpartTest=%d\n", blockIdx.x, tid, g, fullG, partG, totalOut, blockMaxHits, (unsigned int)(nInputs*partG), ((unsigned int)(nInputs * partG) <= tid ? 0 : 1));

  //Output the total number of items for this thread.  Do extra work on the input before outputing.
  for(unsigned int i = 0; i < blockMaxHits; ++i)
    push(extra_work(inputItem), i < totalOut); // defaults to pushing to Out::accept

}

