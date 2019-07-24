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
      while (true)
	{
	  __shared__ unsigned int fireableCounts [numModules];
	  __shared__ unsigned int fireableSignalCounts [numModules];

	  // Calc number of inputs that can be fired (pending, and there
	  // is space in the downstream queue to hold the results) for
	  // each module.
	  bool anyModuleFireable = false;
	  bool anyModuleSignalFireable = false;
	  for (unsigned int i = 0; i < numModules; ++i)
	    {
	      ModuleTypeBase *mod = modules[i];

	      // ignore full ensemble rule if we are in the tail of
	      // execution, or if we are the source

	      bool enforceFullEnsembles =
		(PREFER_FULL_ENSEMBLES   && 
		 !mod->isInTail() && 
		 mod != sourceModule);
		
		__syncthreads();

	      //enforceFullEnsembles &= !(mod->hasCredit());

		//__syncthreads();

	      ////unsigned int numFireable =  
		////mod->computeNumFireableTotal(enforceFullEnsembles);

	      ////if (numFireable > 0)
		////anyModuleFireable = true;

	      ////__syncthreads();
	      
              //stimcheck:  Find the number of signals that remain to be fired for each module
	      unsigned int numSignalFireable =  
		mod->computeNumSignalFireableTotal(enforceFullEnsembles);

	      if (numSignalFireable > 0)
		anyModuleSignalFireable = true;

	      __syncthreads();

	      unsigned int numFireable =  
		mod->computeNumFireableTotal(enforceFullEnsembles);

	      if (numFireable > 0)
		anyModuleFireable = true;

	      __syncthreads();

	      if (IS_BOSS())
		{
		  fireableCounts[i] = numFireable;
		  fireableSignalCounts[i] = numSignalFireable;
		}

	    }
	  
	  // If no module can be fired, either all have zero items pending
	  // (we are done), or no module with pending inputs can fire
	  // (we are deadlocked -- should not happen!).
	  if (!anyModuleFireable && !anyModuleSignalFireable) {
	    break;
	  }
	  
	  // make sure all threads can see fireableCounts[], and
	  // that all modules can see results of firable calculation
	  __syncthreads(); 
	  
	  //
	  // Call the scheduling algorithm to pick next module to fire
	  //
	  unsigned int nextModuleIdx;
	  nextModuleIdx = chooseModuleToFire(fireableCounts, fireableSignalCounts, !anyModuleFireable && anyModuleSignalFireable);

	  __syncthreads();

	  if(IS_BOSS()) {
		////printf("[%d] Calling next module %d\n", blockIdx.x, nextModuleIdx);
		//assert(!(fireableCounts[nextModuleIdx] == 0 && fireableSignalCounts[nextModuleIdx] > 0 && modules[nextModuleIdx]->hasCredit() > 0));
	  }

	  __syncthreads(); ///

	  TIMER_STOP(scheduler);
	  
	  modules[nextModuleIdx]->fire();

	  TIMER_START(scheduler);
	}
      
      __syncthreads(); // make sure final state is visible to all threads
      
      TIMER_STOP(scheduler);
      
#ifndef NDEBUG
      // deadlock check -- make sure no module still has pending inputs
      // stimcheck:  Add the check for signal queue
      bool hasPending = false;
      bool hasPendingS = false;
      bool hasPendingC = false;
      bool hasAllInTail = true;
      for (unsigned int j = 0; j < numModules; j++)
	{
	  unsigned int n = modules[j]->computeNumPendingTotal();
	  unsigned int ns = modules[j]->computeNumPendingTotalSignal();
	  bool nc = modules[j]->hasCredit();
	  bool nt = modules[j]->isInTailInit();
	  hasPending |= (n > 0);
	  //if(n > 0) {
		//printf("HASPENDING FAILED [blockIdx %d, threadIdx %d]:\t%d REMAINING\n", blockIdx.x, threadIdx.x, n);
	  //}
	  hasPendingS |= (ns > 0);
	  hasPendingC |= (nc);
	  if(modules[j] != sourceModule && modules[j]->isEnum()) {
	  	hasAllInTail &= nt;
	  }
	}
      
      assert(hasAllInTail);
      assert(!hasPendingS);
      assert(!hasPendingC);
      assert(!hasPending);
#endif
    }

#ifdef INSTRUMENT_TIME
    __device__
    void printTimersCSV() const
    {
      printf("%d,%d,%llu,%llu,%llu\n",
	     blockIdx.x, -1, schedulerTimer.getTotalTime(), 0, 0);
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
    unsigned int chooseModule_maxOcc(const unsigned int *fireableCounts, const unsigned int *fireableSignalCounts, bool preferSignals)
    {
      __shared__ unsigned int globalMaxIdx;
      if(!preferSignals) {
      using ArgMax = BlockArgMax<unsigned int, 
				 unsigned int, 
				 THREADS_PER_BLOCK>;
      
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
      } //endif
      else {

      using ArgMax = BlockArgMax<unsigned int, 
				 unsigned int, 
				 THREADS_PER_BLOCK>;
      
      unsigned int globalMaxFC             = 0;
      
      for (unsigned int base = 0; base < numModules; base += THREADS_PER_BLOCK)
	{
	  unsigned int modIdx = base + threadIdx.x;
	  unsigned int chunkSize = min(numModules - base, THREADS_PER_BLOCK);
	  
	  // Find the module with maximum fireable count within the
	  // current block-sized chunk.
	  unsigned int fireableCount = 
	    (modIdx < numModules ? fireableSignalCounts[modIdx] : 0);
	  
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
      } //endelse
            
      return globalMaxIdx;
    }
    
    //
    // @brief choose a module randomly from among all modules with > 0
    // fireable inputs using a weighted lottery
    //
    // @param count of fireable items for each module (0 if invalid)
    //
    __device__
    unsigned int chooseModule_lottery(const unsigned int *fireableCounts, const unsigned int *fireableSignalCounts, bool preferSignals)
    {
      assert(numModules <= THREADS_PER_BLOCK);
      
      unsigned int modIdx = threadIdx.x;
      unsigned int fireableCount;
      if(preferSignals) {
      	fireableCount = 
		(modIdx < numModules ? fireableSignalCounts[modIdx] : 0);
      }
      else {
      	fireableCount = 
		(modIdx < numModules ? fireableCounts[modIdx] : 0);
      }

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
    unsigned int chooseModuleToFire(const unsigned int *fireableCounts, const unsigned int *fireableSignalCounts, bool preferSignals)
    {
#if defined(SCHEDULER_MAXOCC)
      
      unsigned int modIdx = chooseModule_maxOcc(fireableCounts, fireableSignalCounts, preferSignals);
      
#elif defined(SCHEDULER_LOTTERY)
      
      // probabilistically choose module proportionally by weight
      unsigned int modIdx = chooseModule_lottery(fireableCounts, fireableSignalCounts, preferSignals);
#else
      
#error "INVALID SCHEDULER SELECTION"

#endif
      
      return modIdx;
    }
  };  // end Scheduler class
}   // end Mercator namespace

#endif
