//
// MODFILTER.CU
// Device-side general modulo filtering application
//
// MERCATOR
// Copyright (C) 2021 Washington University in St. Louis; all rights reserved.
//

#include "ModFilter_dev.cuh"

__device__
void 
ModFilter_dev::Source::init()
{}

__device__
ModFilter_dev::Source::EltT
ModFilter_dev::Source::get(size_t idx) const
{ return idx * 2; }

__device__
void 
ModFilter_dev::Source::cleanup()
{}

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
// Hash each input item and emit the hash iff it is zero modulo
// the current node's modulus value.  This code is shared by all
// nodes of type Filter.
//
__MDECL__
void ModFilter_dev::
Filter<InputView>::run(const InputView &view, size_t base, unsigned int size)
{
  unsigned int tid = threadIdx.x;

  for (unsigned int offset = 0; offset < size; offset += getNumActiveThreads())
    {
      unsigned int idx = offset + tid;
      unsigned int v;

      if (idx < size)
	v = munge(view.get(base + idx));
      
      push(v, idx < size && (v % getParams()->modulus) == 0);
    }
}

