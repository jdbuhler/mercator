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
void genModuleConstruction(const string &moduleObj,
			   const ModuleType *mod,
			   const App *app,
			   Formatter &f)
{
  string hostModuleType = "Host::" + mod->get_name();
  string deviceModuleType  = mod->get_name();
  
  string hostModuleParamObj = "params->p" + mod->get_name();
  string hostAppParamObj    = "params->appParams";
  
  f.add("{");
  f.indent();
  
  // create const array of queue sizes per instance
  {
    string nextStmt = "const unsigned int queueSizes[] = {";
    printf("%s\n-----------------------------\n", mod->get_name().c_str());
    for (const Node *node : mod->nodes) {
      nextStmt += to_string(node->get_queueSize() * options.queueScaler) + ", ";
      printf("\tQUEUE SIZES: nodeQueueSize(%d) * optionsQueueScalar(%d) = %d\n", node->get_queueSize(), options.queueScaler, node->get_queueSize() * options.queueScaler);
    }
    nextStmt += "};";
    
    f.add(nextStmt);
  }
  
  // allocate the module object
  {
    string nextStmt =
      moduleObj + " = new " + deviceModuleType + "(";
    
    if (mod->isSource())
      nextStmt += "tailPtr, ";
    
    nextStmt += "queueSizes";
    
    if (mod->hasParams())
      nextStmt += ", &" + hostModuleParamObj;
    
    if (app->hasParams())
      nextStmt += ", &" + hostAppParamObj;
    
    nextStmt += ");";
    
    f.add(nextStmt);
    f.add("assert(" + moduleObj + " != nullptr);"); // verify that allocation succeeded
  }
  
  f.unindent();
  f.add("}");
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
		  const Edge *dsEdge = 
		    usNode->get_dsEdge(usChannel);
		  
		  if (!dsEdge) // output channel is not connected
		    continue;
		  
		  const Node *dsNode = dsEdge->dsNode;
		    
		  const ModuleType *dsMod = dsNode->get_moduleType();
		  
		  string dsHostModuleType = "Host::" + dsMod->get_name();
		  string dsModuleObj = "d" + dsMod->get_name();
		  
		  // break this call across two lines for readability
		  f.add(moduleObj + "->setDSEdge(" +
			deviceModuleType +
			"::Out::" + channelName + ", " +
			hostModuleType + "::Node::" +
			usNode->get_name() + ", ");
		  
		  f.indentAfter('(');
		  
		  f.add(dsModuleObj + ", " +
			dsHostModuleType + "::Node::" + 
			dsNode->get_name() + ");");
		  
		  f.unindent();
		}
	    }
	  
	  f.add("");
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
  
  // instantiate all modules of the app
  
  f.add("// initialize each module of the app on the device");
  
  for (const ModuleType *mod : app->modules)
    {
      string modVar = mod->get_name();
      
      string deviceModuleType = mod->get_name();
      string moduleObj = "d" + mod->get_name();
      
      f.add(deviceModuleType + "* " + moduleObj + ";");
      
      genModuleConstruction(moduleObj, mod, app, f);
      f.add("");
    }
  
  // connect the instances of each module by edges
  genEdgeInitStmts(app, f);
  
  // create an array of all modules to initialize the scheduler 
  int srcMod;
  int j = 0;
  string nextStmt = "Mercator::ModuleTypeBase *mods[] = {";
  for (const ModuleType *mod : app->modules)
    {
      string moduleObj = "d" + mod->get_name();
      
      nextStmt += moduleObj + ", ";
      
      if (mod->isSource())
	srcMod = j; 
      
      j++;
    }
  nextStmt += "};";
  f.add(nextStmt);
  
  f.add("// tell scheduler about the modules in this app");
  f.add("registerModules(mods, " + to_string(srcMod) + ");");
  
  f.unindent();
  f.add("}");    
}
