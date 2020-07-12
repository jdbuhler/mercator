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
// compile app specs to applications and generate their code
//
static
void compileApps(const vector<input::AppSpec *> &appSpecs,
		 const vector<string> &references)
{  
  TopologyVerifier tv;
  
  for (const input::AppSpec *appSpec : appSpecs)
    {
      App *app = buildApp(appSpec);
      
      tv.verifyTopology(app);
      
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
