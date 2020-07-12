//
// MAIN.CC
// Main program for MERCATOR spec compiler
//
// MERCATOR
// Copyright (C) 2018 Washington University in St. Louis; all rights reserved.
//

#include <iostream>
#include <vector>

#include "options.h"

#include "inputspec.h"
#include "parse.h"
#include "buildapp.h"
#include "topoverify.h"

#include "codegen/gen_hostapp_class.h"
#include "codegen/gen_deviceapp_class.h"

using namespace std;

//
// form names of generated files
//

static
string host_class_cuhfile(const string &appName)
{ return options.outputPath + appName + ".cuh"; }

static
string host_class_cufile(const string &appName)
{ return options.outputPath + appName + "_init.cu"; }

static
string device_class_cuhfile(const string &appName)
{ return options.outputPath + appName + "_dev.cuh"; }

static
string skeleton_cufile(const string &appName)
{ return options.outputPath + appName + ".cu.skl"; }


//
// emit a list of the files generated from a given spec file, in
// a form suitable for a make dependency
//
static
void emitOutputFileNames(const string &sourceFile,
			 const vector<input::AppSpec *> &appSpecs)
{
  for (const input::AppSpec *appSpec : appSpecs)
    {
      const string &appName = appSpec->name;
      
      if (options.appToBuild != "" &&
	  options.appToBuild != appName)
	continue;
      
      cout << host_class_cuhfile(appName)   << ' '
	   << host_class_cufile(appName)    << ' '
	   << device_class_cuhfile(appName) << ' ';
    }

  cout << ": " << sourceFile << endl;
}


//
// fix up the application spec to expand modules and nodes of
// enumerated type into actual enumerating module/node and
// receiver of enumeration
//
static
void redefineEnums(input::AppSpec* appSpec)
{
  vector<input::NodeStmt *> newNodes;
  vector<input::ModuleTypeStmt *> newModules;
  
  //
  // Expand each node A of an enumerating type into a pair of nodes
  // __enumerateFor_A -> A
  //
  
  for (input::NodeStmt* nodeSpec : appSpec->nodes)
    {
      input::ModuleTypeStmt *modSpec = nullptr;
      
      for (input::ModuleTypeStmt *ms : appSpec->modules) 
	{
	  if (ms->name == nodeSpec->type->name)
	    {
	      modSpec = ms;
	      break;
	    }
	}
      
      if (!modSpec) // don't try to fix nodes with nonexistent module types
	continue;
      
      if (modSpec->isEnumerate())
	{
	  input::NodeType* enumNodeType = 
	    new input::NodeType("__enumerateFor_" + modSpec->name);
	  
	  input::NodeStmt* enumNodeSpec = 
	    new input::NodeStmt("__enumerateFor_" + nodeSpec->name, 
				enumNodeType);
	  newNodes.push_back(enumNodeSpec);
	  
	  for (input::EdgeStmt &edgeSpec : appSpec->edges)
	    {
	      //Check for the edge to this node, and re-route
	      //it to the new one
	      if (edgeSpec.to == nodeSpec->name)
		edgeSpec.to = enumNodeSpec->name;
	    }
	  
	  // Add the edge between the new node and the user defined one
	  input::EdgeStmt newEdgeSpec = 
	    input::EdgeStmt(enumNodeSpec->name, "out", nodeSpec->name);
	  
	  appSpec->edges.push_back(newEdgeSpec);
	}
    }
  
  appSpec->nodes.insert(appSpec->nodes.end(), 
			newNodes.begin(),
			newNodes.end());
  
  //
  // Now fix up any module type A whose input is tagged "enumerate"
  // to create the expected enumeration module type __enumerateFor_A
  // and to make A itself take in integers from the parent type.
  //
  for (input::ModuleTypeStmt* modSpec : appSpec->modules)
    {
      if (modSpec->isEnumerate())
	{
	  //
	  // create single output channel specifier of enumerate module
	  //
	  input::DataType* dt  = new input::DataType("unsigned int");
	  input::DataType* dtt = new input::DataType(modSpec->inputType->name);
	  dt->from = dtt;
	  
	  input::ChannelSpec* enumChanSpec = 
	    new input::ChannelSpec("out",
				   dt,
				   1,	//Max Outputs PER input
				   true,
				   false);
	  
	  std::vector<input::ChannelSpec* >* enumChannels = 
	    new std::vector<input::ChannelSpec* >();
	  
	  enumChannels->push_back(enumChanSpec);
	  
	  input::OutputSpec* enumOutSpec = 
	    new input::OutputSpec(enumChannels);
	  
	  //
	  // create and save the actual enumerate module type
	  //
	  
	  input::ModuleTypeStmt* enumForSpec = 
	    new input::ModuleTypeStmt(modSpec->inputType, enumOutSpec);
	  
	  enumForSpec->name = "__enumerateFor_" + modSpec->name;
	  enumForSpec->setEnumerate();
	  
	  newModules.push_back(enumForSpec);
	  
	  //
	  // change the original module to be non-enumerating but
	  // take the integer output of the real enumerating type
	  //
	  
	  dt = new input::DataType("unsigned int");
	  dtt = new input::DataType(modSpec->inputType->name);
	  dt->from = dtt;
	  modSpec->clearEnumerate();
	  
	  modSpec->inputType = dt;
	}
    }
  
  appSpec->modules.insert(appSpec->modules.end(), 
			  newModules.begin(),
			  newModules.end());
}

//
// compile app specs to applications and generate their code
//
static
void compileApps(const vector<input::AppSpec *> &appSpecs,
		 const vector<string> &references)
{  
  TopologyVerifier tv;
  
  //stimcheck: Removed constness of appSpec, need to modify the appSpec
  //to add the mercator and user generated Enum nodes.
  for (input::AppSpec *appSpec : appSpecs)
    {
      redefineEnums(appSpec);

      appSpec->printAll();

      App *app = buildApp(appSpec);
      
      tv.verifyTopology(app);
      
      if (options.appToBuild != "" &&
	  options.appToBuild != app->name)
	continue;
      
#if 1
      app->print();
#endif
      
      const string &appName = appSpec->name;
      
      if (options.generateSkeletons)
	{
	  genDeviceAppSkeleton(skeleton_cufile(appName), app);
	}
      else
	{
	  genHostAppHeader(host_class_cuhfile(appName), app, references);
	  
	  genHostAppConstructor(host_class_cufile(appName), app);
	  
	  genDeviceAppHeader(device_class_cuhfile(appName), app);
	}
      
      delete app;
    }
}


/** 
 * @brief MERCATOR front-end processing: parse mtr files and call code gen.
 *
 */
int main(int argc, char * argv[])
{
  if (!parseCommandLine(argc, argv))
    exit(1);
  
  for (const string &sourceFile : options.sourceFiles)
    {
      // parse every app in the input file
      vector<string> references;
      vector<input::AppSpec *> appSpecs = 
	parseInput(sourceFile,
		   options.typecheckIncludePaths,
		   references);
      
      if (appSpecs.size() == 0)
	{
	  cerr << "WARNING: no apps found in spec file " << sourceFile << endl;
	}
      
      if (options.emitDeps)
	emitOutputFileNames(sourceFile, appSpecs);
      else
	compileApps(appSpecs, references);
    }
  
  return 0;
}
