#ifndef __NODEFUNCTION_CUH
#define __NODEFUNCTION_CUH

//
// @file NodeFunction.cuh
// @brief base type for Node Function objects
//
// MERCATOR
// Copyright (C) 2021 Washington University in St. Louis; all rights reserved.
//

#include "NodeBaseWithChannels.cuh"

#include "ParentBuffer.cuh"

namespace Mercator  {

  //
  // @class NodeFunction
  //
  // @tparam numChannels  number of output channels
  //
  template<unsigned int numChannels>
  class NodeFunction {
    
  protected:
    
    using NodeType = NodeBaseWithChannels<numChannels>;
    
  public:
    
    ///////////////////////////////////////////////////////
    // INIT/CLEANUP KERNEL FUNCIIONS
    ///////////////////////////////////////////////////////
    
    __device__
    NodeFunction(RefCountedArena *iparentArena)
      : node(nullptr),
	parentArena(iparentArena),
	parentIdx(RefCountedArena::NONE)
    {}
    
    //
    // set the node associated with this node function (which may
    // be created only after the function itself.)  Subclasses can
    // can override this function to do stuff with the node, provided
    // they call up to their superclass's setNode function.
    //
    __device__
    void setNode(NodeType *inode)
    { node = inode; }
    
    ////////////////////////////////////////////////////////
    
    //
    // @brief return the parent arena associated with this NodeFunction
    //
    __device__
    RefCountedArena *getParentArena() const
    { return parentArena; }

    //
    // @brief return the current stored parent index
    //
    __device__
    unsigned int getParentIdx() const
    { return parentIdx; }
    
    //
    // @brief set the current stored parent index
    //
        __device__
    void setParentIdx(unsigned int i)
    { parentIdx = i; }
    
    ////////////////////////////////////////////////////////////
    
    //
    // init and cleanup functions for node
    // (may be overridden statically in subclasses)
    //
    __device__
    void init() {}

    __device__
    void cleanup() {}
    
    //
    //
    // begin and end stubs for enumeration and aggregation
    // (may be overridden statically in subclasses)
    //
    
    __device__
    void begin() {}
    
    __device__
    void end() {}
    
  protected:

    NodeType *node;
    
    RefCountedArena* const parentArena;

    unsigned int parentIdx;
  };
}  // end Mercator namespace

#endif
