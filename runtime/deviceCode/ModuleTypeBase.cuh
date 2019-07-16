#ifndef __MODULE_TYPE_BASE_CUH
#define __MODULE_TYPE_BASE_CUH

//
// @file ModuleTypeBase.cuh
// @brief Base class of MERCATOR module type object,  used to
//   access different modules uniformly for scheduling,
//   initialziation, and finalization, and instrumentation printing.
//
// MERCATOR
// Copyright (C) 2018 Washington University in St. Louis; all rights reserved.
//
#include "options.cuh"
#include "QueueBase.cuh"
namespace Mercator  {
  
  //
  // @class ModuleTypeBase
  // @brief A fairly pure virtual base class for modules
  //
  
  class ModuleTypeBase {
  
  public:
    
    __device__
    ModuleTypeBase()
      : inTail(false)
    {}
    
    __device__
    virtual 
    ~ModuleTypeBase() 
    {}
    
    //
    // @brief Is this module in the tail of execution? (impacts
    // scheduling decisions by not waiting for full ensembles to fire)
    //
    __device__
    bool isInTail() const 
    { return inTail; }

    //
    // @brief Indicate whether the module is in the tail of execution.
    //
    // @param v value to set
    //
    __device__
    void setInTail(bool v)
    { inTail = v; }

    
    ///////////////////////////////////////////////////////////////////
    //NEW INTERFACE FOR SCHEDULER_MINSWITCHES (see ModuleType.cuh for details)
    ///////////////////////////////////////////////////////////////////

    __device__
    virtual
    unsigned int getActiveFlag(unsigned int instIdx) const = 0; 

    __device__
    virtual
    void flipActiveFlag(unsigned int instIdx)=0;

    //called with all threads
    __device__
    virtual
    unsigned int computeIsFireable(unsigned int modID_debug)= 0;
  
    __device__
    virtual
    QueueBase *getUntypedQueue() const =0;

    __device__
    virtual
    bool canStillFire(unsigned int instIdx) const =0; 

    ///////////////////////////////////////////////////////////////////
    //OLD SCHEDULING INTERFACE (see ModuleType.cuh for details)
    ///////////////////////////////////////////////////////////////////
    
    // called multithreaded, with enforceFullEnsembles same in all threads
    __device__
    virtual
    unsigned int computeNumFireableTotal(bool enforceFullEnsembles) = 0;
    
    // called multithreaded
    __device__
    virtual
    unsigned int computeNumPendingTotal() const = 0;
    
    // fire a module to consume some input
    __device__
    virtual 
    void fire() = 0;
    
    //
    // init and cleanup functions are called once at the beginning and
    // end of an app's run respectively, using all threads.  By
    // default, these functions are stubs; subclasses may override to
    // perform initial and final actions related to, e.g., module state.
    //
    
    __device__
    virtual void init() {}
    
    __device__
    virtual void cleanup() {}

    //////////////////////////////////////////////////////////////
    // INSTRUMENTATION PRINTING (see ModuleType.h for details)
    //////////////////////////////////////////////////////////////
    
#ifdef INSTRUMENT_TIME
    __device__
    virtual
    void printTimersCSV(unsigned int moduleId) const = 0;
#endif
    
#ifdef INSTRUMENT_OCC
    __device__
    virtual
    void printOccupancyCSV(unsigned int moduleId) const = 0;
#endif
    
#ifdef INSTRUMENT_COUNTS
    __device__
    virtual
    void printCountsCSV(unsigned int moduleid) const = 0;
#endif
    
  private:
    
    bool inTail; // are we in the tail of execution?
    
  };    // end class ModuleTypeBase
}   // end Mercator namespace

#endif
