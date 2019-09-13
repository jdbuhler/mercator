#ifndef __QUEUEBASE_CUH
#define __QUEUEBASE_CUH

//
// @file Queue.cuh
// @brief MERCATOR queue object
//
// MERCATOR
// Copyright (C) 2018 Washington University in St. Louis; all rights reserved.
//

#include <cassert>

#include "device_config.cuh"

namespace Mercator  {
  class QueueBase  {
  
  public:

  __device__
  QueueBase(){}
  

  __device__
  virtual
  ~QueueBase(){}

  __device__
  virtual
  unsigned int getCapacity(unsigned int instIdx)const=0;
  __device__
  virtual
  unsigned int getOccupancy(unsigned int instIdx) const =0;
  __device__
  virtual
  unsigned int getUtilization(unsigned int instIdx) const=0;
  };  // end class Queue

}   // end Mercator namespace

#endif
