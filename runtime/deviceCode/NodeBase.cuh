#ifndef __NODE_BASE_CUH
#define __NODE_BASE_CUH

//
// @file NodeBase.cuh
// @brief Base class of MERCATOR node object,  used to
//   access different nodes uniformly for scheduling,
//   initialization, finalization, and instrumentation printing.
//
// MERCATOR
// Copyright (C) 2019 Washington University in St. Louis; all rights reserved.
//
#include "options.cuh"

namespace Mercator  {
  
  //
  // @class NodeBase
  // @brief A fairly pure virtual base class for nodes
  //
  
  class NodeBase {
  
  public:
    
    __device__
    virtual 
    ~NodeBase() {}

    __device__
    virtual
    void activate() = 0;
    
    __device__
    virtual
    void deactivate() = 0;

    __device__
    virtual
    void decrDSActive() = 0;
    
    __device__
    virtual
    void setFlushing(bool) = 0;
    
    __device__
    virtual
    unsigned int numPending() = 0;

    __device__
    virtual
    void fire() = 0;

    __device__
    virtual
    void setCurrentParent(void* v) = 0;

    __device__
    virtual
    void setEnumId(unsigned int e) = 0;

    __device__
    virtual
    unsigned int getEnumId() = 0;

    __device__
    virtual
    void setWriteThruId(unsigned int w) = 0;

    __device__
    virtual
    unsigned int getWriteThruId() = 0;

    __device__
    virtual
    void freeParent() = 0;
    
    __device__
    virtual
    void init() {}

    __device__
    virtual
    void cleanup() {}
    

    //////////////////////////////////////////////////////////////
    // INSTRUMENTATION PRINTING (see Node.h for details)
    //////////////////////////////////////////////////////////////
    
#ifdef INSTRUMENT_TIME
    __device__
    virtual
    void printTimersCSV(unsigned int nodeId) const = 0;
#endif
    
#ifdef INSTRUMENT_OCC
    __device__
    virtual
    void printOccupancyCSV(unsigned int nodeId) const = 0;
#endif
    
#ifdef INSTRUMENT_COUNTS
    __device__
    virtual
    void printCountsCSV(unsigned int nodeId, bool inputOnly = false) const = 0;
#endif
    
  };    // end class NodeBase
}   // end Mercator namespace

#endif
