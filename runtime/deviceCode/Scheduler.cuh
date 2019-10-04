#ifndef __SCHEDULER_CUH
#define __SCHEDULER_CUH

//
// @file Scheduler.cuh
// @brief MERCATOR device-side application
//
// MERCATOR
// Copyright (C) 2018 Washington University in St. Louis; all rights reserved.
//

#include <cstdio>
#include <cassert>

// for PRNG for lottery scheduler
#include <curand_kernel.h> 

#include "options.cuh"

#include "device_config.cuh"

#include "support/collective_ops.cuh"

#include "ModuleTypeBase.cuh"

#include "instrumentation/device_timer.cuh"

#include "instrumentation/sched_counter.cuh"
namespace Mercator  {
  
  //
  // @class Scheduler
  // @brief Holds and schedules all modules in an application
  //
  
  template <unsigned int numModules,
      unsigned int THREADS_PER_BLOCK>
  class Scheduler {
    
  public:
    
    //
    // @brief Constructor
    // Called single-threaded from init kernel
    // 
    // @param mods array of modules that make up the application
    // @param sourceModuleIndex # of the source module in mods
    //
    __device__
    Scheduler()
    {
      assert(numModules > 0);
      
      init_lotteryScheduler(19);
    }
    
    //
    // @brief destructor
    // called single-threaded from cleanup kernel
    //
    __device__
    ~Scheduler() {}
    
    //
    // @brief run the MERCATOR application to consume all input
    //

  #ifdef SCHEDULER_MINSWITCHES
    __device__
    void run(ModuleTypeBase * const *modules, ModuleTypeBase *sourceModule)
    {
      int tid = threadIdx.x;
        #ifdef PRINTDBG
      int bid = blockIdx.x;
        #endif
      TIMER_START(scheduler);
      // reset the tail state for all modules except the source
      // (which figures out its own tail state)
      for (int base = 0; base < numModules; base += THREADS_PER_BLOCK)
      {
        int idx = base + tid;
        if (idx < numModules && modules[idx] != sourceModule){
          modules[idx]->setInTail(false);
          modules[idx]->deactivateAll();
        }
      }

      //force source active
      sourceModule->activate(0); 


      // main scheduling loop
      while (true){
        //keep track of how many times we g round this loop
        COUNT_SCHED_LOOP();
        //Tail check
        if (sourceModule->isInTail()){          
          for (int base = 0; base < numModules; base += THREADS_PER_BLOCK){
            int idx = base + tid;
            if (idx < numModules){
              modules[idx]->setInTail(true);
              modules[idx]->activateAll();
            }
          }
          // make sure everyone can see tail status
          __syncthreads();         
          //force source inactive
          sourceModule->deactivate(0);
        }
   
        // find first module that is fireable (active followed by inactive)
        int nextFire=-1;
        for (unsigned int i = 0; i < numModules; ++i){
          if (modules[i]->computeIsFireable()){
            nextFire = i;
            break;
          }
        }
        
        // If no module can be fired, either all have zero items pending
        // (we are done), or no module with pending inputs can fire
        // (we are deadlocked -- should not happen!).
        if (nextFire<0){
          break;
        }

        
        TIMER_STOP(scheduler);
        #ifdef PRINTDBG
          if(tid==0) printf("%i: firing module # %u\n",bid,  nextFire);
        #endif
        modules[nextFire]->fire(); 
        #ifdef PRINTDBG
          if(tid==0) printf("%i: module # %u done\n", bid, nextFire);
        #endif
        TIMER_START(scheduler);

        // make sure final state is visible to all threads
        __syncthreads();         
      }


      TIMER_STOP(scheduler);
       
      #ifndef NDEBUG
        // deadlock check -- make sure no module still has pending inputs
        bool hasPending = false;
        for (unsigned int j = 0; j < numModules; j++){
          unsigned int n = modules[j]->computeNumPendingTotal();
          hasPending |= (n > 0);
          #ifdef PRINTDBG
          if(tid==0){
            printf("%u:mod %u of %u has %u pending\n",blockIdx.x, j, numModules,  n);
          }
          #endif
        }
        assert(!hasPending);
      #endif
    }

  #else

    __device__
    void run(ModuleTypeBase * const *modules, ModuleTypeBase *sourceModule)
    {
      int tid = threadIdx.x;
      
      TIMER_START(scheduler);
      
      // reset the tail state for all modules except the source
      // (which figures out its own tail state)
      for (int base = 0; base < numModules; base += THREADS_PER_BLOCK)
      {
        int idx = base + tid;
        if (idx < numModules && modules[idx] != sourceModule)
        modules[idx]->setInTail(false);
      }
      
      // main scheduling loop
      while (true){
        // First, check if the global input buffer is exhausted by seeing
        // if the source module has run out of work to do.  If so, every
        // module should be in the tail of execution.
        //
        // FIXME: the tail indicator should become a signal that is
        // passed from the source down through the app, so that modules
        // do not switch to the tail of execution prematurely.
        //
        if (sourceModule->isInTail()){
            for (int base = 0; base < numModules; base += THREADS_PER_BLOCK){
              int idx = base + tid;
              if (idx < numModules)
                modules[idx]->setInTail(true);
            }
            __syncthreads(); // make sure everyone can see tail status
        }
        
        __shared__ unsigned int fireableCounts [numModules];
        
        // Calc number of inputs that can be fired (pending, and there
        // is space in the downstream queue to hold the results) for
        // each module.
        bool anyModuleFireable = false;
        for (unsigned int i = 0; i < numModules; ++i){
          ModuleTypeBase *mod = modules[i];
          // ignore full ensemble rule if we are in the tail of
          // execution, or if we are the source
          bool enforceFullEnsembles =
            (PREFER_FULL_ENSEMBLES   && 
             !sourceModule->isInTail() && 
             mod != sourceModule); 
          unsigned int numFireable =  
            mod->computeNumFireableTotal(enforceFullEnsembles);
          if (numFireable > 0)
            anyModuleFireable = true;
          if (IS_BOSS()){
            fireableCounts[i] = numFireable;
          }
        }
        
        // If no module can be fired, either all have zero items pending
        // (we are done), or no module with pending inputs can fire
        // (we are deadlocked -- should not happen!).
        if (!anyModuleFireable)
          break;
        
        // make sure all threads can see fireableCounts[], and
        // that all modules can see results of fireable calculation
        __syncthreads(); 
        
        //
        // Call the scheduling algorithm to pick next module to fire
        //
        unsigned int nextModuleIdx = chooseModuleToFire(fireableCounts);
        
        TIMER_STOP(scheduler);
        
        modules[nextModuleIdx]->fire();
        
        TIMER_START(scheduler);
      }
        
      __syncthreads(); // make sure final state is visible to all threads
        
      TIMER_STOP(scheduler);
        
      #ifndef NDEBUG
        // deadlock check -- make sure no module still has pending inputs
        bool hasPending = false;
        for (unsigned int j = 0; j < numModules; j++){
          unsigned int n = modules[j]->computeNumPendingTotal();
          hasPending |= (n > 0);
        }
        assert(!hasPending);
      #endif
    }

  #endif

#ifdef INSTRUMENT_TIME
    __device__
    void printTimersCSV() const
    {
      printf("%d,%d,%llu,%llu,%llu\n",
       blockIdx.x, -1, schedulerTimer.getTotalTime(), 0, 0);
    }
#endif

#ifdef INSTRUMENT_SCHED_COUNTS
  __device__
  void printLoopCount() const{
    printf("%u: Sched Loop Count: %llu\n", blockIdx.x, schedCounter.getLoopCount());
  }
#endif
    
  private:
    
    // random number generator used by lottery scheduler
    curandState_t     randState;
    
    // weights for lottery scheduler
    float           lotteryWeights[numModules];
    
#ifdef INSTRUMENT_TIME
    DeviceTimer schedulerTimer;
#endif

#ifdef INSTRUMENT_SCHED_COUNTS
    SchedCounter schedCounter;
#endif
    //
    // @brief initialize the state for the lottery scheduler
    // Called single-threaded from init kernel
    //
    // @param seed random seed for lottery PRNG
    //
    __device__
    void init_lotteryScheduler(int seed)
    {
      curand_init(seed, blockIdx.x, 0, &randState);
      
      // initially, we set all lottery weights equal
      for (unsigned int j = 0; j < numModules; ++j)
  lotteryWeights[j] = 1.0/numModules;
    }
    
    ////////////////////////////////////////////////////////////////
    // SCHEDULING POLICIES
    ////////////////////////////////////////////////////////////////
  
    //
    // @brief choose the module with the maximum number of items ready to fire
    //
    // @param count of fireable items for each module (0 if invalid)
    //
    __device__
    unsigned int chooseModule_maxOcc(const unsigned int *fireableCounts)
    {
      using ArgMax = BlockArgMax<unsigned int, 
         unsigned int, 
         THREADS_PER_BLOCK>;
      
      __shared__ unsigned int globalMaxIdx;
      unsigned int globalMaxFC             = 0;
      
      for (unsigned int base = 0; base < numModules; base += THREADS_PER_BLOCK)
  {
    unsigned int modIdx = base + threadIdx.x;
    unsigned int chunkSize = min(numModules - base, THREADS_PER_BLOCK);
    
    // Find the module with maximum fireable count within the
    // current block-sized chunk.
    unsigned int fireableCount = 
      (modIdx < numModules ? fireableCounts[modIdx] : 0);
    
    unsigned int maxFC;
    unsigned int maxIdx = ArgMax::argmax(modIdx, fireableCount, maxFC,
                 chunkSize);
  
    // Compare the max result for this chunk to the global max.
    if (IS_BOSS())
      {
        if (maxFC > globalMaxFC)
    {
      globalMaxFC = maxFC;
      globalMaxIdx = maxIdx;
    }
      }
    
    __syncthreads();
  }
            
      return globalMaxIdx;
    }
    
    //
    // @brief choose a module randomly from among all modules with > 0
    // fireable inputs using a weighted lottery
    //
    // @param count of fireable items for each module (0 if invalid)
    //
    __device__
    unsigned int chooseModule_lottery(const unsigned int *fireableCounts)
    {
      assert(numModules <= THREADS_PER_BLOCK);
      
      unsigned int modIdx = threadIdx.x;
      unsigned int fireableCount = 
  (modIdx < numModules ? fireableCounts[modIdx] : 0);
      
      // get scheduling weights for each module, zeroing out those that
      // cannot be fired right now.  (Assumes invalid module IDs have zero
      // fireableCounts).
      float myWeight = (fireableCount > 0 ? lotteryWeights[modIdx] : 0.0);
      
      // get progressive sum of all weights, including total, in parallel
      float totalWeight;
      
      using Scan = BlockScan<float, THREADS_PER_BLOCK>;
      float ivalFloor = Scan::exclusiveSum(myWeight, totalWeight);
      
      __shared__ float schedVal;
      if (IS_BOSS())
  {
    // curand_uniform chooses from (0, 1]; we want (0, totalWeight]
    schedVal = curand_uniform(&randState) * totalWeight;
  }
      __syncthreads();
      
      //
      // find highest entry < schedVal.
      //
      // There will be a single discontinuity from true to false.
      // The tail of the true range is the entry we want.  Note
      // that schedVal > 0, so first thread's input is always true.
      //
      
      using Disc = BlockDiscontinuity<bool, THREADS_PER_BLOCK>;
      bool isTail =  Disc::flagTails(ivalFloor < schedVal, false);
      
      __shared__ int chosenModIdx; 
      if (isTail)
  chosenModIdx = modIdx;
      __syncthreads();
      
      // FIXME: do something sensible to weights to bias future
      // selections toward modules that were not just fired
      
      return chosenModIdx;
    }
    
    //
    // @brief choose the next module to fire by calling a scheduling policy
    //
    // @param count of fireable items for each module (0 if invalid)
    //
    __device__
    unsigned int chooseModuleToFire(const unsigned int *fireableCounts)
    {
#if defined(SCHEDULER_MAXOCC)
      
      unsigned int modIdx = chooseModule_maxOcc(fireableCounts);
      
#elif defined(SCHEDULER_LOTTERY)
      
      // probabilistically choose module proportionally by weight
      unsigned int modIdx = chooseModule_lottery(fireableCounts);    

#elif defined(SCHEDULER_MINSWITCHES)
      unsigned int modIdx = 0;
#else
      
#error "INVALID SCHEDULER SELECTION"

#endif
      
      return modIdx;
    }
  };  // end Scheduler class
}   // end Mercator namespace

#endif
