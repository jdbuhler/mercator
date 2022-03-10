//
// BUILDAPP.CC
// Build an internal application representation from a parsed spec
//
// MERCATOR
// Copyright (C) 2018 Washington University in St. Louis; all rights reserved.
//

#include <iostream>

#include "buildapp.h"
#include "typecheck.h"
#include "options.h"

using namespace std;

// local prototypes

static
bool validateType(const input::DataType *t, ASTContainer *typeInfo);
static
bool validateCompatibleTypes(const DataType *tu,
			     const DataType *td,
			     ASTContainer *typeInfo);
static
void validateTopologyComplete(const App *app);


//
// buildApp()
// Convert an app spec into an internal representation of the app
//
App *buildApp(const input::AppSpec *appSpec)
{
  App *app = new App(appSpec->name,
		     appSpec->threadWidth == 0
		     ? options.threadsPerBlock
		     : appSpec->threadWidth);
  
  for (const input::ModuleTypeStmt *mts : appSpec->modules)
    {
      int mId = app->modules.size();
      
      //
      // VALIDATE that module name is unique for this app, and
      // record mapping from name to index in global module array.
      //
      if (!app->moduleNames.insertUnique(mts->name, mId))
	{
	  cerr << "ERROR: module type " << mts->name
	       << " redefined." << endl;
	  abort();
	}

      //
      // VALIDATE that module's input type is well-defined
      //
      assert(mts->inputType);
      if (!validateType(mts->inputType, appSpec->typeInfo))
	{
	 cerr << "ERROR: module type " << mts->name
	      << " has unknown input data type "
	      <<  mts->inputType->name
	      << endl;
	  abort();
	} 
      
      ModuleType *module = new ModuleType(mts->name,
					  mId,
					  new DataType(mts->inputType),
					  mts->channels.size(),
					  mts->flags,
					  app->threadWidth);
      
      int cId = 0;
      for (const input::ChannelSpec *cs : mts->channels)
	{
	  //
	  // VALIDATE that channel name is unique for this module, and
	  // record mapping from name to index in module's channel array
	  //
	  if (!module->channelNames.insertUnique(cs->name, cId))
	    {
	      cerr << "ERROR: channel name " 
		   << mts->name << "::" << cs->name
		   << " redefined."
		   << endl;
	      abort();
	    }

	  //
	  // VALIDATE that channel's output type is well-defined
	  //
	  if (!validateType(cs->type, appSpec->typeInfo))
	    {
	      cerr << "ERROR: channel "
		   << mts->name << "::" << cs->name
		   << " has unknown output data type "
		   << cs->type->name
		   << endl;
	      abort();
	    }
	  
	  Channel *channel = new Channel(cs->name, cId,
					 new DataType(cs->type),
					 cs->maxOutputs,
					 cs->isVariable,
					 cs->isAggregate);
	  
	  module->set_channel(cId++, channel);
	}

      if (module->isEnumerate())
	{
	  //
	  // create a new enumerating module
	  //
	  
	  string enumModuleName = "__enumerateFor_" + module->get_name();
	  
	  ModuleType *enumModule = new ModuleType(enumModuleName,
						  mId + 1,
						  new DataType(mts->inputType),
						  1, 
						  ModuleType::F_isEnumerate,
						  app->threadWidth);
	  
	  app->moduleNames.insertUnique(enumModuleName, mId + 1);
      
	  // NB: max output count 1 is patently false, but it doesn't
	  // matter excpet for cycle checking -- which doesn't work
	  // with enumeration right now.
	  Channel *channel = new Channel("__out", 0,
					 new DataType("unsigned int", 
						      mts->inputType->name),
					 1, true, false);
	  
	  enumModule->set_channel(0, channel);
	  enumModule->channelNames.insertUnique("__out", 0);
	  
	  //
	  // make the module labeled "enumerate" by the user
	  // formerly enumerating, and fix its input type.
	  //
	  
	  module->set_inputType(new DataType("unsigned int", 
					     mts->inputType->name));
	  module->makeFormerlyEnumerate();
	  
	  app->modules.push_back(module);
	  app->modules.push_back(enumModule);
	  

	}
      else
	app->modules.push_back(module);
    }
  
  for (const input::AllThreadsStmt is : appSpec->allthreads)
    {
      //
      // VALIDATE that allthreads stmt refers to an extant module
      //
      int mId = app->moduleNames.find(is.module);
      if (mId == SymbolTable::NOT_FOUND)
	{
	  cerr << "ERROR: allthreads statement refers to nonexistent module type "
	       << is.module
	       << endl;
	  abort();
	}
      
      ModuleType *module = app->modules[mId];
      
      module->set_useAllThreads();
    }

  for (const input::InterruptStmt is : appSpec->interrupt)
    {
      //
      // VALIDATE that interrupt stmt refers to an extant module
      //
      int mId = app->moduleNames.find(is.module);
      if (mId == SymbolTable::NOT_FOUND)
	{
	  cerr << "ERROR: interrupt statement refers to nonexistent module type "
	       << is.module
	       << endl;
	  abort();
	}
      
      ModuleType *module = app->modules[mId];

      module->set_isInterrupt();
    }

  for (const input::ILimitStmt is : appSpec->ilimits)
    {
      //
      // VALIDATE that input limit refers to an extant module
      //
      int mId = app->moduleNames.find(is.module);
      if (mId == SymbolTable::NOT_FOUND)
	{
	  cerr << "ERROR: ilimit statement refers to nonexistent module type "
	       << is.module
	       << endl;
	  abort();
	}
      
      ModuleType *module = app->modules[mId];
      
      module->set_inputLimit(std::min(is.limit, module->get_inputLimit()));
    }
  
  for (const input::MappingStmt is : appSpec->mappings)
    {
      //
      // VALIDATE that mapping stmt refers to an extant module
      //
      int mId = app->moduleNames.find(is.module);
      if (mId == SymbolTable::NOT_FOUND)
	{
	  cerr << "ERROR: mapping statement refers to nonexistent module type "
	       << is.module
	       << endl;
	  abort();
	  
	}
      
      ModuleType *module = app->modules[mId];
      
      if (is.isSIMD)
	module->set_nThreads(is.nmap); // nmap threads/input
      else
	module->set_nElements(is.nmap);  // nmap inputs/thread
    }
  
  for (const input::NodeStmt *ns : appSpec->nodes)
    {
      int nGlobalId = app->nodes.size();
      
      //
      // VALIDATE that node name is unique for this app, and
      // record mapping from name to index in global node array.
      //
      if (!app->nodeNames.insertUnique(ns->name, nGlobalId))
	{
	  cerr << "ERROR: node " << ns->name
	       << " redefined." << endl;
	  abort();
	}
      
      //
      // Look up the module type of this node.  Sink nodes are handled
      // specially according to their data types; we create their
      // module types if they do not yet exist.
      //
      
      int mId;
      if (ns->type->kind == input::NodeType::isSink)
	{
	  const string &typeStr = ns->type->dataType->name;
		  
	  //
	  // VALIDATE that typeStr names a valid type
	  // 
	  if (!appSpec->typeInfo->queryType(typeStr))
	    {
	      cerr << "ERROR: node " << ns->name 
		   << " references unknown data type " << typeStr 
		   << endl;
	      abort();
	    }
	  
	  unsigned int typeId = appSpec->typeInfo->typeId(typeStr);

	  string moduleName = "__MTR_SINK_" + to_string(typeId);
	  
	  // retrieve the sink module type; create it if it
	  // doesn't exist
	  mId = app->moduleNames.find(moduleName);
	  if (mId == SymbolTable::NOT_FOUND)
	    {
	      mId = app->modules.size();
	      
	      ModuleType *module = 
		new ModuleType(moduleName,
			       mId,
			       new DataType(typeStr), 0, 
			       ModuleType::F_isSink,
			       app->threadWidth);
	      
	      app->moduleNames.insertUnique(moduleName, mId);
	      
	      app->modules.push_back(module);
	    }
	}
      else // non-sink node
	{
	  //
	  // VALIDATE that node's module type is valid, and
	  // get that module
	  //
	  mId = app->moduleNames.find(ns->type->name);
	  if (mId == SymbolTable::NOT_FOUND)
	    {
	      cerr << "ERROR: node " << ns->name
		   << " has unknown module type "
		   << ns->type->name
		   << endl;
	      abort();
	    }
	}
      
      ModuleType *module = app->modules[mId];
      
      int nLocalId = module->nodes.size();
      
      Node *node = new Node(ns->name,
			    module,
			    nLocalId);

      app->nodes.push_back(node);
      module->nodes.push_back(node);
      
      if (module->isFormerlyEnumerate())
	{
	  //
	  // we need an enumerate node prior to this node
	  //
	  string enumName = "__enumerateFor_" + ns->type->name;
	  int emId = app->moduleNames.find(enumName);
	  
	  ModuleType *enumModule = app->modules[emId];
	  
	  int enLocalId = enumModule->nodes.size();
	  
	  string enumNodeName = "__enumerateFor_" + ns->name;
	  Node *enumNode = new Node(enumNodeName,
				    enumModule,
				    enLocalId);
	  
	  app->nodes.push_back(enumNode);
	  enumModule->nodes.push_back(enumNode);
	  
	  app->nodeNames.insertUnique(enumNodeName, nGlobalId + 1);
	  
	  // add an edge from the enumerate node to the given node
	  Edge *edge = new Edge(enumNode, enumModule->get_channel(0), node);
	  
	  enumNode->set_dsEdge(0, edge);

	  node->set_enumerator(enumNode);
	}

      
      //Set isCycle for the node to false by default.
      node->set_isCycle(false);

      //
      // For every node statement, check to see if the node is a simple
      // cycle, and set the appropriate flag for potential topology modifications.
      //
      for (const input::CycleStmt cs : appSpec->cycle)
         {
	    if(cs.name == ns->name) {
		//cout << "cs name: " << cs.name << "\t\tns name: " << ns->name << endl;
      		node->set_isCycle(true);
		node->set_nLayers(cs.layers);
	    }
         }
      
      //
      // Modify the application topology if a node is a simple cycle.
      // By default a simple cycle is unrolled as a pipeline of the same node using
      // the default output channel (0, __out).
      // FIXME: Add other cycle implementations (ones that do not modify topology)
      //
      if(node->get_isCycle())
	{
	  Node* previousNode = node;
	  unsigned int nLayers = node->get_nLayers();

	  // Remember the outgoing edge of the node, this will be reused for the
	  // output channel of the last layer in the cycle.
	  Edge* dsEdge = node->get_dsEdge(0);

	  // Add a param to the node for the layer information.
	  DataItem *v;
	  
	  v = new DataItem("__layer",
	                   new DataType("unsigned int"));
	  module->nodeParams.push_back(v);

	  //
	  // For each layer, create a new node and the necessary edges.
	  // We reuse the initial node's name in the pipeline, and so only need
	  // nLayers - 1 more nodes to complete the cycle.  The first node in the
	  // pipeline is named exactly as the user defined; subsequent nodes are
	  // labeled with __l#_, starting at 0.
	  //
	  for(unsigned int i = 0; i < nLayers - 1; ++i)
	   {
	     // Create the name of the next node.
	     string nextNodeName = "__l" + to_string(i);
	     nextNodeName += "_";
	     nextNodeName += ns->name;

	     // Make a new localId for the new node, used for topology seraching.
	     int nextNLocalId = module->nodes.size() + 1;

	     // Create the next node.
	     Node* nextNode = new Node(nextNodeName,
			               module,
				       nextNLocalId);

	     // FIXME: Set the default cycle status to false.
	     // Necessary to prevent gen_deviceapp_ctor.cc from tyring to set layer
	     // indices internal to the cycle.
	     nextNode->set_isCycle(false);

	     // Add the node to the app's list of nodes; add the node to the module's list of nodes
	     app->nodes.push_back(nextNode);
	     app->nodeNames.insertUnique(nextNodeName, nextNLocalId);
             module->nodes.push_back(nextNode);

	     // Add the new node AFTER the current one.
	     Edge* edge = new Edge(previousNode, module->get_channel(0), nextNode);
	     previousNode->set_dsEdge(0, edge);

	     previousNode = nextNode;

	     // If we have reached the last layer, set the dsEdge of the final node to the initial
	     // dsEdge of the simple cycle.
	     if(i == nLayers - 2) {
		previousNode->set_dsEdge(0, dsEdge);
	     }
	   }

	}
        
    }  
  
  for (const input::EdgeStmt es : appSpec->edges)
    {
      //
      // VALIDATE that upstream node of edge exists
      //
      int usnId = app->nodeNames.find(es.from);
      if (usnId == SymbolTable::NOT_FOUND)
	{
	  cerr << "ERROR: edge has nonexistent upstream node "
	       << es.from << endl;
	  abort();
	}

      Node *usNode = app->nodes[usnId];

      //
      // If the node is a cycle, modify the upstream node of this edge.
      //
      if (usNode->get_isCycle())
	{
	  string loopName = "__l" + to_string(usNode->get_nLayers() - 2);
	  loopName += "_";
	  loopName += usNode->get_name();

	  usnId = app->nodeNames.find(loopName);
      	  if (usnId == SymbolTable::NOT_FOUND)
      	    {
	      cerr << "ERROR: new cycle edge has nonexistent upstream node "
	           << loopName << endl;
	      abort();
	    }
	  usNode = app->nodes[usnId];
	}

      //
      // VALIDATE that downstream node of edge exists
      //
      int dsnId = app->nodeNames.find(es.to);
      if (dsnId == SymbolTable::NOT_FOUND)
	{
	  cerr << "ERROR: edge has nonexistent downstream node "
	       << es.to << endl;
	  abort();
	}
      
      Node *dsNode = app->nodes[dsnId];
      
      //
      // redirect edges into a formerly enumerate node to its actual
      // enumerate node.
      //
      if (dsNode->get_moduleType()->isFormerlyEnumerate())
	{
	  string enumName = "__enumerateFor_" + dsNode->get_name();
	  int emId = app->nodeNames.find(enumName);
	  dsNode = app->nodes[emId];
	}
      
      //
      // VALIDATE that upstream channel of edge exists if specified,
      // or that it is unambiguous if not specified.
      //
      
      ModuleType *mod = usNode->get_moduleType();
      int uscId;
      if (es.fromchannel == "") // not specified
	{
	  if (mod->get_nChannels() != 1)
	    {
	      cerr << "ERROR: ambiguous channel specification for edge "
		   << "out of " << usNode->get_name()
		   << endl;
	      abort();
	    }
	  else
	    uscId = 0; // use the unique channel
	}
      else
	{
	  uscId = mod->channelNames.find(es.fromchannel);
	  if (uscId == SymbolTable::NOT_FOUND)
	    {
	      cerr << "ERROR: edge has nonexistent upstream channel "
		   << mod->get_name() << "::" << es.fromchannel
		   << endl;
	      abort();
	    }
	}
      Channel *usChannel = mod->get_channel(uscId);
      
      Edge *edge = new Edge(usNode, usChannel, dsNode);
      
      
      //
      // VALIDATE that types at the two endpoints of the edge are
      // compatible.
      //
      if (!validateCompatibleTypes(usChannel->type,
				   dsNode->get_moduleType()->get_inputType(),
				   appSpec->typeInfo))
	{
	  const DataType *usType = usChannel->type;
	  const DataType *dsType = dsNode->get_moduleType()->get_inputType();
	  
	  cerr << "ERROR: edge "
	       << usNode->get_name() << "::" << usChannel->name
	       << " -> "
	       << dsNode->get_name()
	       << " has incompatible endpoint types "
	       << (usType ? usType->name : "null") 
	       << " -> "
	       << (dsType ? dsType->name : "null")
	       << endl;
	  abort();
	}
      
      // VALIDATE that no node/channel combo is used as the upstream end
      // of more than one edge.
      if (usNode->get_dsEdge(uscId) != nullptr)
	{
	  cerr << "ERROR: channel " 
	       << usNode->get_name() << "::" << usChannel->name 
	       << " is used as the upstream end of multiple edges ("
	       << usNode->get_dsEdge(uscId)->usNode->get_name() << ", "
	       << usNode->get_dsEdge(uscId)->dsNode->get_name() << ") : ("
	       << usNode->get_name() << ", "
	       << dsNode->get_name() << ")"
	       << endl;
	  abort();
	}
      
      usNode->set_dsEdge(uscId, edge);
    }
  
    
  int vId = 0;
  for (const input::DataStmt *var : appSpec->vars)
    {      
      if (var->scope == "") // app-level variable
	{
	  //
	  // VALIDATE that variable name is unique for this app, and
	  // record mapping from name to index in global variable array.
	  //
	  if (!app->varNames.insertUnique(var->name, vId++))
	    {
	      cerr << "ERROR: app variable "
		   << app->name << "::" << var->name
		   << " is redefined."
		   << endl;
	      abort();
	    }
	  
	  //
	  // VALIDATE that variable's type is well-defined
	  //
	  if (!validateType(var->type, appSpec->typeInfo))
	    {
	      cerr << "ERROR: variable "
		   << app->name << "::" << var->name
		   << " has unknown data type "
		   << var->type->name
		   << endl;
	      abort();
	    }

	  assert(var->isParam);
	  assert(!var->isPerNode);
	  DataItem *v = new DataItem(var->name,
				     new DataType(var->type));
	  
	  app->params.push_back(v);
	}
      else // module-level variables
	{
	  //
	  // VALIDATE that scope names an extant module.
	  //
	  int mId = app->moduleNames.find(var->scope);
	  if (mId == SymbolTable::NOT_FOUND)
	    {
	      cerr << "ERROR: scope of module variable "
		   << var->scope << "::" << var->name
		   << " does not name a valid module."
		   << endl;
	      abort();
	    }
	  
	  ModuleType *mod = app->modules[mId];
	  
	  //
	  // VALIDATE that variable name is unique for this module, and
	  // record mapping from name to index in module's 
	  //
	  if (!mod->varNames.insertUnique(var->name, vId++))
	    {
	      cerr << "ERROR: module variable "
		   << var->scope << "::" << var->name
		   << " is redefined."
		   << endl;
	      abort();
	    }
	  
	  //
	  // VALIDATE that variable's type is well-defined
	  //
	  if (!validateType(var->type, appSpec->typeInfo))
	    {
	      cerr << "ERROR: module variable "
		   << var->scope << "::" << var->name
		   << " has unknown data type "
		   << var->type->name
		   << endl;
	      abort();
	    }
	  
	  DataItem *v = new DataItem(var->name,
				     new DataType(var->type));
	  
	  if (var->isParam)
	    {
	      vector<DataItem *> &vars = 
		(var->isPerNode ? mod->nodeParams : mod->moduleParams);
	      
	      vars.push_back(v);
	    }
	  else // state -- per-node only
	    {
	      assert(var->isPerNode);
	      mod->nodeState.push_back(v);
	    }
	}   
    }
  
  //
  // Each sink module type has a parameter which is the data
  // describing its sink (passed from the host).
  //
  for (ModuleType *mod : app->modules)
    {
      if (mod->isSink())
	{
	  string dataType = mod->get_inputType()->name;
	  DataItem *v;
	  
	  v = new DataItem("sinkData",
			   new DataType("Mercator::SinkData<"
					+ dataType
					+ ">"));
	  mod->nodeParams.push_back(v);
	}

    }

    
  //
  // Process the source designation.  VALIDATE that exactly
  // one node has been designated source.
  //
  if (appSpec->sources.size() == 0)
    {
      cerr << "ERROR: app " << app->name
	   << " has no designated source node." << endl;
      abort();
    }
  else if (appSpec->sources.size() > 1)
    {
     cerr << "ERROR: app " << app->name
	  << " has two or more designated source nodes." << endl;
     abort();
    }
  else
    {
      const input::SourceStmt &ss = appSpec->sources[0];
      
      int nid = app->nodeNames.find(ss.node);
      if (nid == SymbolTable::NOT_FOUND)
	{
	  cerr << "ERROR: source statement designates nonexistent node "
	       << ss.node << endl;
	  abort();
	}

      Node *node = app->nodes[nid];
      if (node->get_moduleType()->isFormerlyEnumerate())
	node = node->get_enumerator();

      node->set_isSource(true);
      app->sourceNode = node;
      
      if (ss.kind == input::SourceStmt::SourceIdx)
	{
	  app->sourceKind = App::SourceIdx;
	  
	  // make sure input type of source module is size_t
	  const DataType *inputType = 
	    app->sourceNode->get_moduleType()->get_inputType();
	  
	  if (!appSpec->typeInfo->compareTypes(inputType->name, "size_t"))
	    {
	      cerr << "ERROR: input type of source node "
		   << node->get_name()
		   << "should be size_t\n";
	      abort();
	    }
	}
      else if (ss.kind == input::SourceStmt::SourceBuffer)
	app->sourceKind = App::SourceBuffer;
      else // function
	app->sourceKind = App::SourceFunction;
    }

  // make sure app's graph has a source, and that no edges are omitted.
  validateTopologyComplete(app);
  

  //
  // VALIDATE that all modules are used in the app.  If a module
  // is not used, warn and do not generate code for it.
  //
  vector<ModuleType *> usedMods;
  for (ModuleType *mod : app->modules)
    {
      if (mod->nodes.size() > 0)
	usedMods.push_back(mod);
      else
	{
	  cerr << "WARNING: module type " << mod->get_name()
	       << " has no instance; not generating code for it"
	       << endl;
	}
    }
  
  if (usedMods.size() < app->modules.size())
    app->modules = usedMods;
  
  return app;
}


//
// validateType()
// Check that a data type mentioned in the spec is well-defined.
//
static
bool validateType(const input::DataType *t, ASTContainer *typeInfo)
{
  // verify that each type in the hierarchy of a from type is valid
  while (t != nullptr)
    {
      if (!typeInfo->queryType(t->name))
	return false;
      
      t = t->from;
    }
  
  return true;
}


//
// validateCompatibleTypes()
// Check that two types at ends of an edge are compatible.
//
static
bool validateCompatibleTypes(const DataType *tu,
			     const DataType *td,
			     ASTContainer *typeInfo)
{
  // neither type may be entirely null
  if (tu == nullptr || td == nullptr)
    return false;
  
  // verify that each type in the hierarchy of a from type is compatible
  while (tu != nullptr && td != nullptr)
    {
      if (!typeInfo->compareTypes(tu->name, td->name))
	return false;
      
      tu = tu->from;
      td = td->from;
    }
  
  // upstream type must have at least as many from levels as
  // downstream type
  return (td == nullptr);  
}


//
// validateTopologyComplete()
// Check that the topology specified for an app has no missing
// pieces.  Every app's graph must have a source node, and every
// node must have all of its output channels attached to edges.
//
// (NB: we may consider relaxing the latter condition if we wish to
// allow users to discard some outputs of some nodes.)
//
static
void validateTopologyComplete(const App *app)
{
  //
  // VALIDATE that app has a source node
  //
  if (app->sourceNode == nullptr)
    {
      cerr << "ERROR: app " << app->name
	   << " has no source node."
	   << endl;
      abort();
    }
  
  for (const Node *node : app->nodes)
    {
      int nChannels = node->get_moduleType()->get_nChannels();
      
      //
      // VALIDATE that node has an edge attached to
      // each of its output channels.  The runtime supports
      // throwing away outputs to unconnected channels, but
      // we should at least warn that this is happening.
      //
      for (int j = 0; j < nChannels; j++)
	{
	  if (node->get_dsEdge(j) == nullptr)
	    {
	      cerr << "WARNING: node " << node->get_name()
		   << " has no edge attached to output channel "
		   << node->get_moduleType()->get_channel(j)->name
		   << endl
		   << "  Outputs to this channel will be discarded."
		   << endl;
	    }
	}
    }
}
