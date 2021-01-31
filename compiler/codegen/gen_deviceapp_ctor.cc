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
			     const string modType,
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
      else // enumerate can keep going until channel fills
	spaceRequired = 1; 
      
      f.add(nodeObj + "->initChannel<"
	    + channel->type->name + ">("
	    + modType + "::Out::" + channel->name + ", "
		+ to_string(spaceRequired)
	    + (channel->isAggregate ? ", true);" : ");"));
      
      if (mod->isUser() && !mod->get_useAllThreads())
	{
	  // channel is buffered -- create its buffer object
	  f.add(nodeFcnObj + "->initChannelBuffer<"
		+ channel->type->name + ">("
		+ modType + "::Out::" + channel->name + ", "
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
  string deviceModuleKind = mod->get_name();
  string deviceModuleType  = deviceModuleKind +
    "<" + (node->get_isSource()
	   ? "Source" 
	   : "Mercator::Queue<" + mod->get_inputType()->name + ">")
    + ">";
  
  string nodeFunctionObj = nodeObj + "Fcn";
  
  string hostNodeParamObj   = "params->n" + mod->get_name() + 
    "[" + to_string(node->get_idxInModuleType()) + "]";
  string hostModuleParamObj = "params->p" + mod->get_name();
  string hostAppParamObj    = "params->appParams";
  
	
  // allocate the node function object
  
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
	    arenaObj = enumNodeFcnObj + "->getArena()";
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
  
  
  // allocate the containing node object
  
  if (node->get_isSource())
    {
      // create the actual source object
      f.add("Source *sourceObj = new Source(tailPtr, params);");
      
      // allocate the node object
      string NodeType = "Mercator::Node_Source<" + 
	mod->get_inputType()->name +
	", " + to_string(mod->get_nChannels()) +
	", Source"
	", " + deviceModuleKind + ">";
      
      nextStmt =
	NodeType + " * " + nodeObj + " = new " + NodeType + "(";
      
      nextStmt += "&scheduler";
      
      nextStmt += ", " + to_string(node->get_regionId());
      
      nextStmt += ", sourceObj";
      
      nextStmt += ", " + nodeFunctionObj;
      
      nextStmt += ");";
      
      f.add(nextStmt);
    }
  else
    {
      // only nodes in non-zero enumeration regions use signals
      bool usesSignals = (node->get_regionId() > 0);
      
      string NodeType = "Mercator::Node_Queue<" + 
	mod->get_inputType()->name +
	", " + to_string(mod->get_nChannels()) +
	", " + to_string(usesSignals) +
	", " + deviceModuleKind + ">";
      
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
      
      f.add(nextStmt);
    }
  
  // verify that allocation succeeded	
  f.add("assert(" + nodeObj + " != nullptr);");
  
  // allocate the channels for the node
  genNodeChannelInitStmts(mod, deviceModuleType, nodeObj, nodeFunctionObj, f);
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
  
  f.add("// construct each node of the app on the device");
  
  for (const Node *node : app->nodes)
    {
      if (!node->get_moduleType()->isSource()) // source was not built
	{
	  string nodeObj = "d" + node->get_name();
	  genNodeConstruction(nodeObj, node, node->get_moduleType(), app, f);
	  f.add("");
	}
    }
  
  // create an array of all nodes to initialize the scheduler 
  
  string nextStmt = "Mercator::NodeBase *nodes[] = {";

  for (const Node *node : app->nodes)
    {
      if (!node->get_moduleType()->isSource()) // source was not built
	{
	  string nodeObj = "d" + node->get_name();
	  nextStmt += nodeObj + ", ";
	}
    }
  nextStmt += "};";
  f.add(nextStmt);
  
  f.add("// tell device app about all nodes");
  f.add("registerNodes(nodes);");
  
  f.unindent();
  f.add("}");    
}
