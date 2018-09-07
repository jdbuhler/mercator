// INPUTSPEC.H
// Representation of MERCATOR input spec
//
// These objects are used directly inside the parser to build up all
// the information gleaned from the spec about an application.
 
#ifndef __INPUTSPEC_H
#define __INPUTSPEC_H

#include <string>
#include <vector>
#include <cassert>

class ASTContainer;

namespace input {
  
  // type of a data item in MERCATOR
  struct DataType {
    std::string name; // name of data type
    DataType *from;   // aggregate for "from" type
    
    DataType(const std::string &name)
      : name(name), from(nullptr)
    {}
    
    ~DataType()
    {
      if (from) 
	delete from;
    }
    
    bool isFromType() const 
    { return (from != nullptr); }
  };
  
  // specification of a module's output channel
  struct ChannelSpec {
    std::string name; // channel names
    DataType *type;   // type of values emitted on channel
    int maxOutputs;   // max outputs per input
    bool isVariable;  // is outputs/input fixed or variable?
    
    ChannelSpec(const std::string &name,
		DataType *type,
		int maxOutputs,
		bool isVariable)
      : name(name),
	type(type),
	maxOutputs(maxOutputs),
	isVariable(isVariable)
    {}
    
    ~ChannelSpec()
    {
      delete type;
    }
  };
  
  
  //
  // Description of a module type's output
  // Output may be a list of channels,
  // the special "Sink", or the special
  // "void" indicating a non-sink with no
  // outputs (but presumably some side-effects.
  //
  struct OutputSpec {
    
    enum Kind {
      isVoid, isSink, isOther
    };
    
    Kind kind;
    std::vector<ChannelSpec *> *channels;
    
    OutputSpec(std::vector<ChannelSpec *> *ichannels)
      : kind(isOther),
	channels(ichannels)
    {}
    
    OutputSpec(Kind kind)
      : kind(kind),
	channels(nullptr)
    {}
    
    ~OutputSpec()
    { 
      if (channels) 
	delete channels; 
    }
  };
  
  //
  // Description of a module type
  //
  
  struct ModuleTypeStmt {
    
    // flags set when parsing module type spec
    enum { 
      isEnumerate = 0x01,
      isAggregate = 0x02 
    };
    
    std::string name;                     // name of module
    unsigned int flags;                   // assorted properties
    DataType *inputType;                  // type of module input
    std::vector<ChannelSpec *> channels;  // output channels
    
    
    ModuleTypeStmt(DataType *inputType,
		   OutputSpec *outputSpec)
      : flags(0),
	inputType(inputType)
    {
      assert(inputType);
      
      channels = *(outputSpec->channels);
      
      delete outputSpec; // clean up spec now that we've used it
    }
    
    ~ModuleTypeStmt()
    {
      delete inputType;
      
      for (ChannelSpec *channel : channels)
	delete channel;
    }
  };
  
    
  //
  // Statement limiting the number of threads
  // that can concurrently execute a module.
  //
  
  struct ILimitStmt {
    std::string module;
    int limit;
    
    ILimitStmt(const std::string &module,
	       int limit)
      : module(module),
	limit(limit)
    {}
  };
  
  
  //
  // Statement specifying that a module should
  // run with all threads, regardless of how many
  // inputs it has.
  //
  
  struct AllThreadsStmt {
    std::string module;
    
    AllThreadsStmt(const std::string &module)
      : module(module)
    {}
  };
  
  
  //
  // Statement specifying a mapping other than
  // one thread/item for a module.
  //
  
  struct MappingStmt {
    std::string module;
    int nmap;
    bool isSIMD; // true if mapping is 1:nmap rather than nmap:1
    
    MappingStmt(const std::string &module,
		int nmap, bool isSIMD = false)
      : module(module),
	nmap(nmap),
	isSIMD(isSIMD)
    {}
  };
  
  
  //
  // module type of a node.  May be either a string referencing a
  // module name or a special source/sink type referencing its
  // datatype.
  //
  
  struct NodeType {
    
    enum Kind {
      isSource, isSink, isOther
    };
    
    std::string name;    // names a module for standard nodes
    Kind kind;           // indicates source or sink type
    DataType *dataType;  // element type of source or sink
    
    NodeType(const std::string &name)
      : name(name),
	kind(isOther),
	dataType(nullptr)
    {}
    
    NodeType(Kind kind,
	     DataType *dataType)
      : name(""),
	kind(kind),
	dataType(dataType)
    {}
    
    ~NodeType()
    { 
      if (dataType) delete dataType; 
    }
  };
  
  
  //
  // Description of an application node
  //
  
  struct NodeStmt {
    std::string name;
    NodeType *type;
    
    NodeStmt(const std::string &name,
	     NodeType *type)
      : name(name),
	type(type)
    {}
    
    ~NodeStmt()
    { delete type; }
  };
  
  //
  // Description of an edge connecting two nodes
  //
  
  struct EdgeStmt {
    std::string from;
    std::string fromchannel;
    std::string to;
    
    EdgeStmt(const std::string &from,
	     const std::string &fromchannel,
	     const std::string &to)
      : from(from),
	fromchannel(fromchannel),
	to(to)
    {}
  };
  
  
  //
  // Description of a data value stored globally
  // or within a single module type, either
  // singly or per node.
  //
  
  struct DataStmt {
    std::string name;
    std::string scope;
    DataType *type;
    bool isParam;
    bool isPerNode;
    
    DataStmt(const std::string &name,
	     const std::string &scope)
      : name(name),
	scope(scope),
	type(nullptr),
	isParam(false),
	isPerNode(false)
    {}
    
    ~DataStmt()
    {
      if (type)
	delete type;
    }
  };
  
  // all topological info for a MERCATOR application
  struct AppSpec {
    std::string name;
    ASTContainer *typeInfo;
    
    std::vector<ModuleTypeStmt *> modules;
    
    std::vector<NodeStmt *>  nodes;
    
    std::vector<EdgeStmt>    edges;
    
    std::vector<ILimitStmt>  ilimits;
    
    std::vector<AllThreadsStmt> allthreads;
    
    std::vector<MappingStmt> mappings;
    
    std::vector<DataStmt *>  vars;
    
    AppSpec(const std::string &name)
      : name(name),
	typeInfo(nullptr)
    {}
    
    ~AppSpec()
    {
      for (ModuleTypeStmt *module : modules)
	delete module;

      for (NodeStmt *node : nodes)
	delete node;

      for (DataStmt *var : vars)
	delete var;
    }
  };  
}

#endif
