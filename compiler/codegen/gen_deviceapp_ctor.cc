//
// @file gen_deviceapp_ctor.h
// @brief generate device-side app constructor
//
// MERCATOR
// Copyright (C) 2018 Washington University in St. Louis; all rights reserved.

#include "gen_deviceapp_ctor.h"

#include "../options.h"

#include "app.h"

#include "Formatter.h"
#include "codegen_utils.h"

using namespace std;


//
// @brief generate statements that initialize the channels of
//   a node
//
// @param mod module type of node whose channels are being generated
// @param nodeObj name of node object to receive channels
// @param nodeFunctionObj name of node function object to receive 
//     channel buffers, if appropriate
// @param f Formatter to receive generated code
static
void genNodeChannelInitStmts(const ModuleType *mod,
			     const string &nodeObj,
			     const string &nodeFcnObj,
			     Formatter &f)
{
  // create output channels
  int nChannels = mod->get_nChannels();

  // init output channels
  for (int j=0; j < nChannels; ++j)
    {
      const Channel *channel = mod->get_channel(j);
      
      unsigned int spaceRequired;
      if (mod->isUser())
	spaceRequired = channel->maxOutputs * mod->get_inputLimit();
      else // source or enumerate can keep going until channel fills
	spaceRequired = 1; 
      
      f.add(nodeObj + "->initChannel<"
	    + channel->type->name + ">("
	    + mod->get_name() + "::Out::" + channel->name + ", "
		+ to_string(spaceRequired)
	    + (channel->isAggregate ? ", true);" : ");"));
      
      if (mod->isUser() && !mod->get_useAllThreads())
	{
	  // channel is buffered -- create its buffer object
	  f.add(nodeFcnObj + "->initChannelBuffer<"
		+ channel->type->name + ">("
		+ mod->get_name() + "::Out::" + channel->name + ", "
		+ to_string(channel->maxOutputs)
		+ ");");
	}
    }
}

//
// @brief generate statements to construct an individual module object inside
//   the app's constructor
//
// @param moduleObj name of module object being built
// @param mod module type being built
// @param app application being codegen'd
// @param f Formatter to receive generated code
//
static
void genNodeConstruction(const string &nodeObj,
			 const Node *node,
			 const ModuleType *mod,
			 const App *app,
			 Formatter &f)
{
  string hostModuleType = "Host::" + mod->get_name();
  string deviceModuleType  = mod->get_name();
  
  string nodeFunctionObj = nodeObj + "Fcn";
  
  string hostNodeParamObj   = "params->n" + mod->get_name() + 
    "[" + to_string(node->get_idxInModuleType()) + "]";
  string hostModuleParamObj = "params->p" + mod->get_name();
  string hostAppParamObj    = "params->appParams";
  
  // allocate the node object
  {
    if (!mod->isSource())
      {
	// allocate the corresponding node function object
	
	string nextStmt =
	  deviceModuleType + "* " + nodeFunctionObj +
	  " = new " + deviceModuleType + "(";
	
	vector<string> arglist;
	
	string arenaObj;
	if (node->get_regionId() > 0) // node is in a non-base region
	  {
	    // add pointer to region head's parent buffer. which
	    // already exists because we construct nodes in
	    // topological order
	    
	    Node *enumNode = app->regionHeads[node->get_regionId()];
	    string enumNodeFcnObj = "d"  + enumNode->get_name() + "Fcn";
	    arenaObj = enumNodeFcnObj + "->getParentArena()";
	  }
	else
	  arenaObj = "nullptr";
	
	arglist.push_back(arenaObj);
	
	if (mod->isEnumerate())
	  arglist.push_back(to_string(node->get_enumerateId()));
	
	if (mod->hasNodeParams())
	  arglist.push_back("&" + hostNodeParamObj);
	
	if (mod->hasModuleParams())
	  arglist.push_back("&" + hostModuleParamObj);
	
	if (app->hasParams())
	  arglist.push_back("&" + hostAppParamObj);
	
	if (arglist.size() > 0)
	  {
	    nextStmt += arglist[0];
	    for (unsigned int j = 1; j < arglist.size(); j++)
	      nextStmt += ", " + arglist[j];
	  }
	
	nextStmt += ");";
	
	// verify that allocation succeeded	
	f.add(nextStmt);
	f.add("assert(" + nodeFunctionObj + " != nullptr);");
	
	// Now allocate the node object
	string NodeType = "Mercator::Node< " + 
	  mod->get_inputType()->name +
	  ", " + to_string(mod->get_nChannels()) +
	  ", THREADS_PER_BLOCK"
	  ", " + deviceModuleType + ">";
	
	nextStmt =
	  NodeType + " * " + nodeObj + " = new " + NodeType + "(";
	
	nextStmt += "&scheduler";
	
	nextStmt += ", " + to_string(node->get_regionId());
	
	const Edge *usEdge = node->get_usEdge();
	nextStmt += ", d" + usEdge->usNode->get_name();
	nextStmt += ", " + to_string(usEdge->usChannel->id);
	
	nextStmt += 
	  ", " + to_string(node->get_queueSize() * options.queueScaler);
	
	nextStmt += ", " + nodeFunctionObj;
	
	nextStmt += ");";
	
	// verify that allocation succeeded	
	f.add(nextStmt);
	f.add("assert(" + nodeObj + " != nullptr);");
      }
    else
      {
	// source node 
	string nextStmt =
	  deviceModuleType + "* " + nodeObj + " = new " + deviceModuleType + "(";
	
	nextStmt += "&scheduler";
	
	nextStmt += ", " + to_string(node->get_regionId());
	
	nextStmt += ", tailPtr";
	
	if (mod->hasNodeParams())
	  nextStmt += ", &" + hostNodeParamObj;
	
	if (mod->hasModuleParams())
	  nextStmt += ", &" + hostModuleParamObj;
	
	if (app->hasParams())
	  nextStmt += ", &" + hostAppParamObj;
	
	nextStmt += ");";
	
	f.add(nextStmt);
	f.add("assert(" + nodeObj + " != nullptr);"); // verify that allocation succeeded
      }
    
    // allocate the channels for the node
    genNodeChannelInitStmts(mod, nodeObj, nodeFunctionObj, f);
  }
}


//
// @brief Code-gen device-side app constructor
//
void genDeviceAppConstructor(const App *app,
			     Formatter &f)
{
  string DeviceAppClass = app->name + "_dev";
  
  f.add("__device__");
  f.add(genFcnHeader("", 
		     DeviceAppClass,
		     "size_t *tailPtr, "
		     "const " + app->name + "::Params *params"));
  
  f.add("{");
  f.indent();
  
  f.add("using Host = " + app->name + ";");
  f.add("");
  
  // instantiate all nodes of the app in topological order
  
  f.add("// initialize each node of the app on the device");
  
  for (const Node *node : app->nodes)
    {
      string nodeObj = "d" + node->get_name();
            
      genNodeConstruction(nodeObj, node, node->get_moduleType(), app, f);
      f.add("");
    }
  
  // create an array of all nodes to initialize the scheduler 
  
  string nextStmt = "Mercator::NodeBase *nodes[] = {";
  for (const Node *node : app->nodes)
    {
      string nodeObj = "d" + node->get_name();
      
      nextStmt += nodeObj + ", ";
    }
  nextStmt += "};";
  f.add(nextStmt);
  
  f.add("// tell device app about all nodes");
  f.add("registerNodes(nodes);");
  
  f.unindent();
  f.add("}");    
}
