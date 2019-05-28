#ifndef __FINEGRAINED_TIMER_H
#define __FINEGRAINED_TIMER_H


//
// FG_Container.CUH
// Device-side timer container for use with device_timer.cuh
//
// MERCATOR
// Copyright (C) 2018 Washington University in St. Louis; all rights reserved.
//

#include <cassert>
#include "options.cuh"
#include "device_config.cuh" // for IS_BOSS
class FGContainer {
public:  
  __device__
  FGContainer() 
  {
    if (IS_BOSS()){
      assert(initContainment()>=0){
    }
  }
  __device__
  int recordStamp(DevClockT now){

    return 0;
  }
  __device__ 
  int dumpContainment(){
  
    return 0;
  }
private:
  __device__
  int initContainment(){
    
    return 0;
  }
};

#endif
