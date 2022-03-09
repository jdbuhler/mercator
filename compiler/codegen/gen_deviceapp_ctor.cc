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
// @brief generate statements to construct an individual module object inside
//   the app's constructor
//
// @param nodeObj name of node object being built
// @param node representation of node object
// @param app application being codegen'd
// @param f Formatter to receive generated code
//
static
void genNodeConstruction(const string &nodeObj,
			 const Node *node,
			 const App *app,
			 Formatter &f)
{
  const ModuleType *mod = node->get_moduleType();

  string hostModuleType = "Host::" + mod->get_name();
  string deviceModuleKind = mod->get_name();
  string deviceModuleType = deviceModuleKind +
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
    {
      arglist.push_back(to_string(node->get_enumerateId()));
      arglist.push_back(to_string(node->get_nTerminalNodes()));
    }
  
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
      
      nextStmt += ", " + to_string(node->isTerminalNode());
      
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
}


//
// @brief generate statements that initialize the channels of
//   a node
//
// @param nodeObj name of node object to receive channels
// @param node the actual node representation
// @param f Formatter to receive generated code
static
void genNodeChannelInitStmts(const string &nodeObj,
			     const Node *node,
			     Formatter &f)
{
  const ModuleType *mod = node->get_moduleType();
  string nodeFcnObj = nodeObj + "Fcn";
  
  // create output channels and associate them with the node
  for (unsigned int j = 0; j < mod->get_nChannels(); ++j)
    {
      const Channel *channel = mod->get_channel(j);
      const Node *dsNode = node->get_dsEdge(j)->dsNode;
      string dsNodeObj = "d" + dsNode->get_name();
	  
      unsigned int spaceRequired;
      if (mod->isUser())
	spaceRequired = channel->maxOutputs * mod->get_inputLimit();
      else // enumerate can keep going until channel fills
	spaceRequired = 1; 
      
      string ChannelType = "Mercator::Channel<" + channel->type->name + ">";
      
      f.add("{");
      f.indent();
      
      f.add(ChannelType + "*channel = new " + ChannelType + "("
	    + to_string(spaceRequired) + ", "
	    + (channel->isAggregate ? "true, " : "false, ")
	    + dsNodeObj + ", "
	    + dsNodeObj + "->getQueue(), "
	    + dsNodeObj + "->getSignalQueue());");
      
      f.add("assert(channel != nullptr);");
      
      f.add(nodeObj + "->setChannel(" + to_string(j) + ", channel);");
      
      if (mod->isUser() && !mod->get_useAllThreads() && !mod->get_isInterrupt())
	{
	  // channel is buffered -- create its buffer object
	  f.add(nodeFcnObj + "->initChannelBuffer<"
		+ channel->type->name + ">("
		+ to_string(j) + ", "
		+ to_string(channel->maxOutputs)
		+ ");");
	}
      
      f.unindent();
      f.add("}");
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
  
  f.add("// construct each node of the app on the device");
  
  for (const Node *node : app->nodes)
    {
      string nodeObj = "d" + node->get_name();
      genNodeConstruction(nodeObj, node, app, f);
      f.add("");
    }
  
  f.add("// construct the output channels for each node");
  
  for (const Node *node : app->nodes)
    {
      string nodeObj = "d" + node->get_name();
      genNodeChannelInitStmts(nodeObj,
			      node,
			      f);
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

  /*
  f.add("");

  // set the layer information for nodes if needed
  // Note: Layer index starts at 0
  for (const Node *node : app->nodes)
    {
      if(node->get_isCycle()) {
        //for (const input::CycleStmt cs : appSpec->cycle)
        //   {
        //      if(cs.name == node->get_name()) {
        //          node->set_nLayers(cs.layers);
	//      }
        //   }
	printf("FOUND CYCLE %s\n", node->get_name().c_str());
        // set the layer for each subsequent node created for the unrolled cycle
        // set layer info for first node
	string nodeParamsBegin = "auto __cycleparams = ";
	string startNodeName = "d" + node->get_name();
	startNodeName += "Fcn";
	string assignStartParams = nodeParamsBegin + startNodeName;
	assignStartParams += "->getParams();";
	f.add(assignStartParams);

	string setStartParams = "__cycleparams->__layer = 0;";
	f.add(setStartParams);

	f.add("");

	unsigned int nLayers = node->get_nLayers();

	// loop for later layers
	for(unsigned int i = 0; i < nLayers - 2; ++i) {
	   printf("BUILDING LAYER %d\n", i);
           string nodeName = "d__l" + to_string(i);
	   nodeName += "_" + node->get_name();
	   nodeName += "Fcn";

	   string assignParams = "__cycleparams = " + nodeName;
	   assignParams += "->getParams();";
	   f.add(assignParams);

	   string setParams = "__cycleparams->__layer = " + to_string(i + 1);
	   setParams += ";";
	   f.add(setParams);
	   f.add("");
	}
      }
    }
    */  
  f.unindent();
  f.add("}");    
}
