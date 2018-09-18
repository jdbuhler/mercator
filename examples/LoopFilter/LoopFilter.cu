//
// LOOPFILTER.CU
// Device-side code for a loop-based filter cascade
//
// MERCATOR
// Copyright (C) 2018 Washington University in St. Louis; all rights reserved.
//

#include "LoopFilter_dev.cuh"

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
// Setup wraps each input item in a struct that also
// includes a loop count, which is initially 0.
//
__device__
void LoopFilter_dev::
Setup::run(const unsigned int& inputItem, InstTagT nodeIdx)
{
  UIntWithCounter elt;
  elt.v = inputItem;
  elt.count = 0;
  
  push(elt, nodeIdx);
}


//
// Filter hashes the input item's value and then tests it
// against the modulus corresponding to its count.  If
// the item passes, we recycle it for the next round of
// testing, or accept if all rounds have passed.
//
__device__
void LoopFilter_dev::
Filter::run(const UIntWithCounter& inputItem, InstTagT nodeIdx)
{
  unsigned int v = munge(inputItem.v);
  
  unsigned int modulus = getAppParams()->moduli[inputItem.count];
  
  if (v % modulus == 0)
    {
      if (inputItem.count == getAppParams()->numCycles - 1)
	{
	  // success!
	  push(v, nodeIdx, Out::accept);
	}
      else
	{
	  // recycle to try the next modulus
	  UIntWithCounter elt;
	  elt.v = v;
	  elt.count = inputItem.count + 1;
	  
	  push(elt, nodeIdx, Out::keepgoing);
	}	  
    }
}
