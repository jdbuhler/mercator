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
   
    nextStmt += "&scheduler";
    
    nextStmt += ", " + to_string(node->get_regionId());
    
    if (mod->isSource())
      nextStmt += ", tailPtr";
    else
      {
	const Edge *usEdge = node->get_usEdge();
	nextStmt += ", d" + usEdge->usNode->get_name();
	nextStmt += ", " + to_string(usEdge->usChannel->id);
	
	nextStmt += 
	  ", " + to_string(node->get_queueSize() * options.queueScaler);
	
	string arenaObj;
	if (node->get_regionId() > 0) // node is in a non-base region
	  {
	    // add pointer to region head's parent buffer. which
	    // already exists because we construct nodes in
	    // topological order
	    
	    Node *enumNode = app->regionHeads[node->get_regionId()];
	    string enumNodeObj = "d"  + enumNode->get_name();
	    arenaObj = enumNodeObj + "->getParentArena()";
	  }
	else
	  arenaObj = "nullptr";
	
	nextStmt += ", " + arenaObj;
      }
    
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
