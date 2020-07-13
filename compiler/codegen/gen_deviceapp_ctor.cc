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
  
  string hostNodeParamObj   = "params->n" + mod->get_name() + 
    "[" + to_string(node->get_idxInModuleType()) + "]";
  string hostModuleParamObj = "params->p" + mod->get_name();
  string hostAppParamObj    = "params->appParams";
  
  // allocate the node object
  {
    string nextStmt =
      deviceModuleType + "* " + nodeObj + " = new " + deviceModuleType + "(";
    
    if (mod->isSource())
      nextStmt += "tailPtr";
    else
      nextStmt += to_string(node->get_queueSize() * options.queueScaler);
   
    nextStmt += ", &scheduler";
    
    nextStmt += ", " + to_string(node->get_regionId());
    
    if (mod->isEnumerate())
      nextStmt += ", " + to_string(node->get_enumerateId());
    
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
}

    
//
// @brief Generate statements that connect the modules of the app
//    according to the user's edge specifications
//
// @param app app being codegen'd
// @param f Formatter to receive generated code
//
static
void genEdgeInitStmts(const App *app,
		      Formatter &f)
{
  for (const ModuleType *mod : app->modules)
    {
      int nChannels = mod->get_nChannels();
	
      if (nChannels > 0)
	{
	  f.add("// set outgoing edges for nodes of module type " + 
		mod->get_name());
	  
	  string hostModuleType   = "Host::"   + mod->get_name();
	  string deviceModuleType = mod->get_name();
	  string moduleObj = "d" + mod->get_name();
	  
	  // set downstream queues
	  for (int usChannel = 0; usChannel < nChannels; ++usChannel)
	    {
	      string channelName = mod->get_channel(usChannel)->name;
	      
	      for (const Node *usNode : mod->nodes)
		{
		  const Edge *dsEdge = usNode->get_dsEdge(usChannel);
		  if (!dsEdge) // output channel is not connected
		    continue;
		  
		  string usNodeObj = "d" + usNode->get_name();
		  string dsNodeObj = "d" + dsEdge->dsNode->get_name();
		  
		  f.add(usNodeObj + "->setDSEdge(" +
			deviceModuleType +
			"::Out::" + channelName + ", " +
			dsNodeObj + ", " +
			to_string(dsEdge->dsReservedSlots) +  ");");
		}
	    }
	  
	  f.add("");
	}
    }
}

void
connectRegionHeads(const App *app,
		   Formatter &f)
{
  for (const Node *node : app->nodes)
    {
      if (node->get_regionId() > 0) // node is in a non-base region
	{
	  Node *enumNode = app->regionHeads[node->get_regionId()];
	  
	  string enumNodeObj = "d"  + enumNode->get_name();
	  string childNodeObj = "d" + node->get_name();
	  
	  f.add("// set parent arena ptr for node in region " + 
		to_string(node->get_regionId()));
	  
	  f.add(childNodeObj + "->setParentArena(" +
		enumNodeObj + "->getParentArena());");
	}
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
  
  // instantiate all nodes of the app
  
  f.add("// initialize each node of the app on the device");
  
  for (const ModuleType *mod : app->modules)
    {
      string deviceModuleType = mod->get_name();

      for (const Node *node : mod->nodes)
	{
	  string nodeObj = "d" + node->get_name();
	  genNodeConstruction(nodeObj, node, mod, app, f);
	  f.add("");
	}
    }
  
  // connect the instances of each module by edges
  genEdgeInitStmts(app, f);
  f.add("");
  connectRegionHeads(app, f);
  f.add("");
  
  // create an array of all modules to initialize the scheduler 
  int srcNode;
  int j = 0;
  string nextStmt = "Mercator::NodeBase *nodes[] = {";
  for (const Node *node : app->nodes)
    {
      string nodeObj = "d" + node->get_name();
      
      nextStmt += nodeObj + ", ";
      
      if (node->get_moduleType()->isSource())
	srcNode = j; 
      
      j++;
    }
  nextStmt += "};";
  f.add(nextStmt);
  
  f.add("// tell device app about all nodes");
  f.add("registerNodes(nodes, " + to_string(srcNode) + ");");
  
  f.unindent();
  f.add("}");    
}
