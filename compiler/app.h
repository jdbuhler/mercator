#ifndef __APP_H
#define __APP_H

//
// APP.H
// Internal representation of MERCATOR application for validation
// and code generation.
//
// MERCATOR
// Copyright (C) 2018 Washington University in St. Louis; all rights reserved.
//

#include <string>
#include <vector>
#include <cassert>

#include "symboltable.h"

// A data type.  For now, this matches the input::DataType, except for
// being able to do deep copies.  Later, we may wish to change the
// representation used for analysis and codegen.

namespace input {
  class DataType;
};

struct DataType {
  
  std::string name; // name of data type
  DataType *from;   // aggregate for "from" type
  
  DataType(const input::DataType *ti);
  
  DataType(const std::string &iname);
  
  DataType(const std::string &iname, const std::string &fname);
  
  ~DataType()
  {
    if (from) 
      delete from;
  }
  
  bool isFromType() const 
  { return (from != nullptr); }
  
  void print() const;
};


// fwd declarations to resolve circular deps
class Edge;
class Node;
class Channel;
class ModuleType;


//
// A data parameter of an app, module, or node
//
struct DataItem {
  std::string name; 
  DataType *type;   
  
  DataItem(const std::string &iname,
	   DataType *itype)
    : name(iname),
      type(itype)
  {}
  
  ~DataItem()
  {
    delete type;
  }
  
  void print() const;
};


//
// An edge between two nodes
//
struct Edge {
  Node    *usNode;              // upstream node
  Channel *usChannel;           // channel of upstream node
  Node    *dsNode;              // downstream node
  unsigned int dsReservedSlots; // reserved slots in downstream queue

  Edge(Node *iusNode,
       Channel *iusChannel,
       Node *idsNode)
    : usNode(iusNode),
      usChannel(iusChannel),
      dsNode(idsNode),
      dsReservedSlots(0)
  {}
  
  void print() const;
};


//
// An output channel of a module
// FIXME: do we actually need to track the channel idx?
struct Channel {
  std::string name;
  unsigned int id;             // index in module's channel array

  DataType *type;              // output type of channel
  unsigned int  maxOutputs;    // max outputs/input on channel
  bool isAggregate;            // is the channel used for aggregates?
  
  Channel(const std::string &iname,
	  unsigned int iid,
	  DataType *itype,
	  unsigned int imaxOutputs,
          bool iisAggregate)
    : name(iname),
      id(iid),
      type(itype),
      maxOutputs(imaxOutputs),
      isAggregate(iisAggregate)
  {}
  
  ~Channel()
  {
    delete type;
  }
  
  void print() const;
};

//
// A node in the app's graph
//
class Node {
public:
  
  Node(const std::string &iname,
       ModuleType *imt,
       unsigned int imIdx);
  
  ~Node();
  
  const std::string &get_name() const
  { return name; }
  
  ModuleType *get_moduleType() const
  { return moduleType; }
  
  unsigned int get_idxInModuleType() const
  { return mIdx; }

  unsigned int get_regionId() const
  { return regionId; }

  unsigned int get_enumerateId() const
  { return enumerateId; }
  
  unsigned int get_nTerminalNodes() const
  { return nTerminalNodes; }
  
  unsigned int get_queueSize() const
  { return queueSize; }
  
  Edge *get_usEdge() const { return treeEdge; }
  
  Edge *get_dsEdge(int i) const { return dsEdges[i]; }
  
  bool get_isSource() const { return isSource; }

  Node *get_enumerator() const { return enumerator; }
  
  void set_dsEdge(int i, Edge *e) const { dsEdges[i] = e; }
  
  void set_regionId(int r) { regionId = r; }
  
  void set_enumerateId(int e) { enumerateId = e; }
  
  void set_nTerminalNodes(int n) { nTerminalNodes = n; }
  
  void set_isSource(bool v) { isSource = v; }

  void set_enumerator(Node *n) { enumerator = n; }
  
  bool isTerminalNode() const { return _isTerminalNode; };
  void setTerminalNode() { _isTerminalNode = true; }
  
  void print() const;
  
  friend class TopologyVerifier;
  
private:

  std::string name;
  ModuleType *moduleType;      // module of which we are an instance
  unsigned int mIdx;           // index of node in its module type

  unsigned int regionId;
  unsigned int enumerateId;
  unsigned int nTerminalNodes;
  
  bool isSource;
  bool _isTerminalNode;
  
  unsigned int queueSize;

  Node *enumerator;            // if we have an enumerated input, which
                               // node actually does the enumerating?
  
  Edge **dsEdges;              // one per channel
  
  // DFS-specific fields used in topology checking
  
  Edge *treeEdge;              // predecessor in acyclic app tree
  Edge *cycleEdge;             // predecessor on cycle, if any
  
  enum DfsStatus {
    NotVisited = 0,
    InProgress = 1,
    Finished   = 2
  };
  
  DfsStatus dfsStatus;
  unsigned int startTime;
  long multiplier;
};


//
// A module type for app nodes
//
class ModuleType {
public:
  // flags describing special module properties
  enum {
    F_isSimple            = 0x01,
    F_isSink              = 0x02,
    F_isEnumerate         = 0x04,
    F_isFormerlyEnumerate = 0x08,
  };
  
  ModuleType(const std::string &iname,
	     unsigned int iidx,
	     DataType *iinputType,
	     unsigned int inChannels,
	     unsigned int flags,
	     unsigned int inputLimit);
  
  ~ModuleType();
  
  //
  // inspectors
  //
  
  const std::string &get_name() const 
  { return name; }
  
  unsigned int get_idx() const 
  { return idx; }
  
  const DataType *get_inputType() const
  { return inputType; }
  
  unsigned int get_nChannels() const
  { return nChannels; }
  
  unsigned int get_inputLimit() const
  { return inputLimit; }
  
  unsigned int get_nElements() const
  { return nElements; }
  
  Channel *get_channel(unsigned int cId) const
  { 
    assert(cId < nChannels && channels[cId] != nullptr);
    return channels[cId]; 
  }

  
  //
  // mutators
  //
  
  void set_inputType(DataType *t)
  {
    if (inputType) delete inputType;
    inputType = t;
  }
  
  void set_inputLimit(unsigned int il)
  { inputLimit = il; }

  void set_nElements(unsigned int ni)
  { nElements = ni; }
  
  void set_channel(unsigned int cId, Channel *c)
  {
    assert(cId < nChannels && channels[cId] == nullptr);
    channels[cId] = c;
  }
  
  bool isSimple()            const { return flags & F_isSimple; }
  bool isSink()              const { return flags & F_isSink; }
  bool isEnumerate()         const { return flags & F_isEnumerate; }
  bool isUser()              const { return (flags & (F_isSink | F_isEnumerate)) == 0; }
  bool isFormerlyEnumerate() const { return flags & F_isFormerlyEnumerate; }
  
  void makeFormerlyEnumerate()
  { 
    flags &=  ~F_isEnumerate;
    flags |=  F_isFormerlyEnumerate;
  }

  bool hasNodeParams() const 
  {
    return nodeParams.size() > 0;
  }

  bool hasModuleParams() const 
  {
    return moduleParams.size() > 0;
  }
  
  bool hasState() const
  {
    return (nodeState.size() > 0);
  }
  
  void print() const;
  
  SymbolTable channelNames;  // maps channel namme -> idx in channels
  
  SymbolTable varNames;      // maps param name -> unique id
  
  std::vector<Node *> nodes; // all nodes of this module type
  
  std::vector<DataItem *> moduleParams; // all per-module parameters
  std::vector<DataItem *> nodeParams;   // all per-node parameters

  std::vector<DataItem *> nodeState;  // all per-node state

private:
  
  std::string name;        // module name
  unsigned int idx;        // index in global module array

  DataType *inputType;     // type of data we take in
  
  unsigned int nChannels;  // # of output channels
  Channel **channels;      // array of output channels
  
  unsigned int flags;      // flags for special module properties
  
  unsigned int inputLimit; // max threads per firing
  
  unsigned int nElements;  // items consumed by each thread
};  


//
// Application 
//
struct App {

  enum SourceKind { SourceIdx, SourceBuffer, SourceFunction };
  
  std::string name;
  
  unsigned int threadWidth;
  
  Node *sourceNode;
  SourceKind sourceKind;
  std::string sourceParam; // for array sources only
  
  std::vector<ModuleType *> modules;
  
  std::vector<Node *> nodes;
  
  std::vector<DataItem *> params;
  
  // maps regions to their head nodes
  std::vector<Node *> regionHeads;

  // maps regions to # of terminal nodes for each
  std::vector<unsigned int> regionNTerminalNodes; 
  
  SymbolTable moduleNames;  // maps module name -> idx in modules
  
  SymbolTable nodeNames;    // maps node names -> idx in nodes
  
  SymbolTable varNames;     // maps variable names -> unique ids
  
  App(const std::string &iname, unsigned int ithreadWidth)
    : name(iname),
      threadWidth(ithreadWidth),
      sourceNode(nullptr),
      sourceKind(SourceIdx)
  {}
  
  ~App()
  {
    for (DataItem *p : params)
      delete p;
    
    for (ModuleType *mt : modules)
      delete mt;

    // modules are responsible for freeing nodes of their type
  }
  
  bool hasParams() const { return params.size() > 0; }
  
  void print() const;
};

#endif
