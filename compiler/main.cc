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

static
void redefineEnum(input::AppSpec* appSpec)
{
	//for(input::ModuleTypeStmt* mod : appSpec->modules)
	for(unsigned int z = 0; z < appSpec->modules.size(); ++z)
	{
		input::ModuleTypeStmt* mod = appSpec->modules.at(z);
		cout << mod->name << endl;
		//If the moduleType is an enumerate
		if(mod->flags & 0x04 && !(mod->flags & 0x08))
		{
			cout << "IS AN ENUMERATE: " << mod->name << endl;
			input::DataType* dt = new input::DataType("unsigned int");
			input::DataType* dtt = new input::DataType(mod->inputType->name);
			dt->from = dtt;
			//stimcheck: TODO Set from type for __enumerateFor module here, so getParent can be codegen'd.
			input::ChannelSpec* enumChanSpec = new input::ChannelSpec("out",
									       dt,
									       1,	//Max Outputs PER input
									       true,
									       false);

			std::vector<input::ChannelSpec* >* enumChannels = new std::vector<input::ChannelSpec* >();

			enumChannels->push_back(enumChanSpec);

			input::OutputSpec* enumOutSpec = new input::OutputSpec(enumChannels);
			cout << "HERE1" << endl;
			input::ModuleTypeStmt* enumFor = new input::ModuleTypeStmt(mod->inputType, enumOutSpec);
			cout << "HERE2" << endl;
			enumFor->name = "__enumerateFor_" + mod->name;
			enumFor->flags |= 0x04;
			enumFor->flags |= 0x08;	//Set as finalized, so we don't enumerate this again by accident

			appSpec->modules.push_back(enumFor);
			cout << "HERE3" << endl;

			
			dt = new input::DataType("unsigned int");
			dtt = new input::DataType(mod->inputType->name);
			dt->from = dtt;
			mod->flags = 0x00;	//Zero out this enumerate module's flags, not needed here anymore
			enumFor->flags |= 0x20;	//Set the ignore type checking flag
			mod->inputType = dt;	//Change the input type to the user defined enumerate module to unsigned int

			cout << "HERE4" << endl;
			//for(input::NodeStmt* node : appSpec->nodes)
			for(unsigned int i = 0; i < appSpec->nodes.size(); ++i)
			{
				input::NodeStmt* node = appSpec->nodes.at(i);
				//Check if the current node is of the enumerate module type we are currently replacing
				cout << "HERE5" << endl;
				if(node->type->name == mod->name)
				{
					cout << "HERE6" << endl;
					input::NodeType* nt = new input::NodeType("__enumerateFor_" + mod->name);
					cout << "HERE7" << endl;
					input::NodeStmt* enumNode = new input::NodeStmt("__enumerateFor_" + node->name, nt);
					cout << "HERE8" << endl;
					appSpec->nodes.push_back(enumNode);
					cout << "HERE9" << endl;

					//Add the edge between the new node and the user defined one
					input::EdgeStmt edge = input::EdgeStmt(enumNode->name, "out", node->name);
					cout << "HERE10" << endl;

					//for(input::EdgeStmt edge : appSpec->edges)
					for(unsigned int j = 0; j < appSpec->edges.size(); ++j)
					{
						input::EdgeStmt* edgee = &(appSpec->edges.at(j));
						cout << "HERE11" << endl;
						//Check for the edge to this node, and re-route it to the new one
						if(edgee->to == node->name)
						{
							cout << "HERE12" << endl;
							edgee->to = enumNode->name;
						}
					}
					cout << "HERE13" << endl;
					appSpec->edges.push_back(edge);
					cout << "HERE14" << endl;
				}
				cout << "HERE15" << endl;
			}

			/*
			for(input::EdgeStmt edge : appSpec->edges)
			{
				//Check for the edge to this module, and re-route it to the new one
				if(edge.to == mod->name)
				{
					
				}
			}
			*/
		}
	}
	cout << "EXITING ENUM FIXER" << endl;
}


//
// compile app specs to applications and generate their code
//
static
void compileApps(vector<input::AppSpec *> &appSpecs,
		 const vector<string> &references)
{  
  for (input::AppSpec *appSpec : appSpecs)
    {
      //pre-process current AppSpec to properly compile enumeration/aggregation
      redefineEnum(appSpec);

	appSpec->printAll();

      App *app = buildApp(appSpec);
	cout << "APP BUILT" << endl;
      TopologyVerifier::verifyTopology(app);
	cout << "TOPOLOGY VERIFIED" << endl;
      
      if (options.appToBuild != "" &&
	  options.appToBuild != app->name)
	continue;
      
#if 0
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
