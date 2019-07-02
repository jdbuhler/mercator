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
#define BUCKETS 10000 //2^14
#define PI 3.14159
#define E 2.71828


typedef unsigned long long DevClockT;

typedef struct {
  DevClockT mu;
  DevClockT sigma;
  int total;
}CompDistro; //compressed distrobution


class FGContainer {

  public:  

    __device__
    FGContainer(){
      if (IS_BOSS()){
        maxCycle=INSTRUMENT_FG_TIME;
        minCycle=0;
        for(int i=0; i< BUCKETS; i++){
          storage[i]=0;
        }
        bucketRange = (maxCycle - minCycle) /(BUCKETS-1);
      }
    }

    __device__
    void recordStamp(DevClockT stamp){
        assert(IS_BOSS());
        
        if(stamp>=maxCycle){
          storage[BUCKETS-1]++;
        }
        else if (stamp<minCycle){
          storage[0]++;
        }
        else{
          int idx =(int)((stamp-minCycle) / bucketRange); 
          //printf("idx: %i\n", idx);
          assert(idx<BUCKETS);
          assert(idx>0);
          storage[idx]++;
        }
      
    }

    __device__ 
    void dumpContainer(int blkIdx, int modId)const{
      
      for(int i=0; i< BUCKETS; i++){
        if(storage[i]>0)
          printf("%d,%u,%i,%llu\n",blkIdx, modId, storage[i], getIndexCycleCount(i));
      }
    

        //printf("%d,%u,%d,%d\n",blkIdx, modId, cd.mu ,cd.sigma);

    }

    __device__
    void setBounds(unsigned long long lowerBound, unsigned long long upperBound) {
      assert(IS_BOSS());
      assert(lowerBound<upperBound);
      maxCycle = (DevClockT)upperBound;
      minCycle = (DevClockT)lowerBound;
      bucketRange = (maxCycle - minCycle) /(BUCKETS-1);
    }

    __device__
    unsigned long long getLowerBound()const{
      return (unsigned long long) minCycle;
    }

    __device__
    unsigned long long getUpperBound()const{
      return (unsigned long long) maxCycle;
    }

  private:
    __device__ 
    DevClockT getIndexCycleCount(int i)const{      
        return ((bucketRange*i)+bucketRange)+minCycle;
    }
  
    __device__
    double computeGaussian(CompDistro* compd, double x){
      return ((1.0/(compd->sigma*(sqrt(2.0*PI)))))*pow(E,(-0.5*pow(((x-compd->mu)/(compd->sigma)),2.0)));
    }
    __device__
    double compute_post_mean(CompDistro* compd, DevClockT sample){
      return (compd->mu+(1*compd->sigma*sample))/ (1*(compd->sigma)+1)  ;
    }
    __device__
    double compute_post_var(CompDistro* compd){
      return (compd->sigma)/(1+compd->sigma);
    }


    CompDistro cd;
    DevClockT maxCycle;
    DevClockT minCycle;
    DevClockT bucketRange;
    int storage[BUCKETS];
};

#endif
