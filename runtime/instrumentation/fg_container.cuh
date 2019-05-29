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
#define BUCKETS 10001 //2^14
typedef unsigned long long DevClockT;

class FGContainer {

  public:  

    __device__
    FGContainer(){
      if (IS_BOSS()){
        maxCycle=INSTRUMENT_FG_TIME;
        for(int i=0; i< BUCKETS; i++){
          storage[i]=0;
        }
        bucketRange = maxCycle/(BUCKETS-1);
      }
    }

    __device__
    void recordStamp(DevClockT stamp){
        if(stamp>=maxCycle){
          storage[BUCKETS-1]++;
        }
        else{
          int idx =(int)(stamp / bucketRange); 
          storage[idx]++;
        }
    }

    __device__ 
    void dumpContainer(int blkIdx, int modId)const{
      for(int i=0; i< BUCKETS; i++){
        printf("%d,%u,%i,%llu\n",blkIdx, modId, storage[i], getIndexCycleCount(i));
      }
    }

  private:
    __device__ 
    DevClockT getIndexCycleCount(int i)const{      
        return (bucketRange*i)+bucketRange;
    }

    DevClockT maxCycle;
    DevClockT bucketRange;
    int storage[BUCKETS];
};

#endif
