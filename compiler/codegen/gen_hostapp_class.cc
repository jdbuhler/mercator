//
// @file gen_hostapp_class.cc
// @brief code-gen MERCATOR app on host
//
// MERCATOR
// Copyright (C) 2018 Washington University in St. Louis; all rights reserved.
//

#include <iostream>

#include "gen_hostapp_class.h"

#include "app.h"

#include "Formatter.h"
#include "codegen_utils.h"

using namespace std;

//
// @brief Genenerate the app-wide parameter structure
//
// @param app Application for which to generate output
// @param f Formatter to receive code
//
static
void genHostAppParameterStruct(const App *app,
			       Formatter &f)
{
  f.add("struct AppParams {");
  f.indent();
  
  for (const DataItem *param : app->params)
    f.add(param->type->name + " " + param->name + ";");
  
  f.unindent();
  f.add("};");
}


//
// @brief Generate the structure of per-module parameters for a module.
//
// @param mod Module for which to generate output
// @param f Formatter to receive code
//
static
void genHostModuleParameterStruct(const ModuleType *mod,
				  Formatter &f)
{
  f.add("struct ModuleParams {");
  f.indent();

  // per-module parameters  
  for (const DataItem *param : mod->moduleParams)
    f.add(param->type->name + " " + param->name + ";");
  
  f.unindent();
  f.add("};");
}

//
// @brief Generate the structure of per-node parameters for a module.
//
// @param mod Module for which to generate output
// @param f Formatter to receive code
//

static
void genHostNodeParameterStruct(const ModuleType *mod,
				Formatter &f)
{
  f.add("struct NodeParams {");
  f.indent();

  // per-node parameters
  for (const DataItem *param : mod->nodeParams)
    f.add(param->type->name + " " + param->name + ";");
  
  f.unindent();
  f.add("};");
}
  

//
// @brief Generate the host-side class definition for a module
//
// @param mod Module for which to generate output
// @param f Formatter to receive code
//
static
void genHostModuleClass(const ModuleType *mod,
			Formatter &f)
{
  f.add("class " + mod->get_name() + " {");
  f.indent();
  
  f.add("public:", true);
  
  // generate structure to hold module-wide parameters
  genHostModuleParameterStruct(mod, f);
  f.add("");
  
  // generate structure to hold per-node parameters
  genHostNodeParameterStruct(mod, f);
  f.add("");
  
  // generate accessor for module-wide parameters  
  f.add(genFcnHeader("ModuleParams*",
		     "getParams",
		     ""));
  f.add("{ return params; }");
  f.add("");
  
  // constructor takes pointer to module-wide parameter structure
  f.add(genFcnHeader("",
		     mod->get_name(),
		     "ModuleParams *iparams"));
  f.add(" : params(iparams) {}");
  f.add("");
  
  f.add("private:", true);
  
  // generate params object pointer storage
  // (does not change after construction)
  f.add("ModuleParams * const params;");
  
  f.unindent();
  f.add("};");
}


//
// @brief generate the host-side class definitions for each node of
// a given module type.  These classes exist only for modules that
// have per-node parameters; they let the user set the parameters of
// a single node directly rather than modifying an entry in the module's
// per-node parameter array.
//
// @param mod Module being codegen'd
// @param f Formatter to receive generated code
//
static
void genHostNodeClasses(const ModuleType *mod,
			Formatter &f)
{
  for (const Node *node : mod->nodes)
    {
      f.add("class " + node->get_name() + " {");
      f.indent();
      
      f.add("public:", true);
      
      //
      // generate functions to access/mutate parameter information
      //
      
      if (mod->isSink())
	{
	  string dataType = mod->get_inputType()->name;
	  
	  // capture sink-associated buffer
	  f.add(genFcnHeader("void",
			     "setSink",
			     "Mercator::Buffer<"
			     + dataType
			     + ">& buffer"));
	  f.add("{");
	  f.indent();
	  
	  f.add("params->sinkData.bufferData = buffer.getData();");
	  
	  f.unindent();
	  f.add("}");
	}
      else
	{
	  // accessor for node parameters
	  f.add(genFcnHeader(mod->get_name() + "::NodeParams*",
			     "getParams",
			     ""));
	  f.add("{ return params; }");
	}
      f.add("");

      // constructor takes pointer to node parameter structure 
      f.add(genFcnHeader("",
			 node->get_name(),
			 mod->get_name() + "::NodeParams *iparams"));
      f.add(" : params(iparams) {}");
      f.add("");
      
      f.add("private:", true);
      
      // generate params object pointer storage
      // (does not change after construction)
      
      f.add(mod->get_name() + "::NodeParams * const params;");
      
      f.unindent();
      f.add("};");
      f.add("");
    }
}


//
// @brief Generate a single struct containing all parameters for the
//   application and each of its modules and nodes. We pass this
//   structure down to the device.
//
// @param app Application for which to generate output
// @param f Formatter to receive code
//
static
void genHostAppAllParameters(const App *app,
			     Formatter &f)
{
  f.add("struct Params {");
  f.indent();
  
  if (app->sourceKind == App::SourceBuffer)
    {
      string sourceType = 
	app->sourceNode->get_moduleType()->get_inputType()->name;
      
      f.add("Mercator::BufferData<" + sourceType + "> *sourceBufferData;");
    }
  else
    f.add("size_t nInputs;");
  f.add("");
  
  f.add("AppParams appParams;");
  f.add("");
  
  for (const ModuleType *mod : app->modules)
    {
      f.add(mod->get_name() + "::ModuleParams p" + mod->get_name() + ";");
      
      f.add(mod->get_name() + "::NodeParams n" + mod->get_name() + 
	    "[" + to_string(mod->nodes.size()) + "];");
    }
  
  f.unindent();
  f.add("};");
}



//
// @brief generate the entire host-side header for a MERCATOR app
//
void genHostAppHeader(const string &hostClassFileName,
		      const App *app,
		      const vector<string> &userIncludes)
{
  Formatter f;
  
  // add include guard header for host
  {
    string incGuard = genIncludeGuardName(app->name);
    f.add("#ifndef " + incGuard);
    f.add("#define " + incGuard);
    f.add("");    
  }

  // add MERCATOR includes
  
  f.add(genUserInclude("version.h"));
  f.add(genUserInclude("hostCode/AppDriverBase.cuh"));
  f.add("");
  
  if (app->sourceKind == App::SourceBuffer)
    f.add(genUserInclude("io/Buffer.cuh"));
  
  f.add(genUserInclude("io/Sink.cuh"));
  f.add("");
  
  // include any files needed to define user-referenced types
  if (userIncludes.size() > 0)
    {
      for (const string &inc : userIncludes)
	f.add(genUserInclude(inc));
      
      f.add("");
    }
  
  // begin host app class
  f.add("class " + app->name + " {");
  f.indent();

  f.add("public:", true);
    
  // generate constants for compile-time app properties
  f.add("static const int NUM_NODES = " + 
	to_string(app->nodes.size()) + ";");
  f.add("");

  // report mapping of IDs to modules, used for statistics
  f.add("// NODE IDENTIFIERS: ");
  for (unsigned int j = 0; j < app->nodes.size(); j++)
    {
      const Node *node = app->nodes[j];
      const ModuleType *mod = node->get_moduleType();
      
      f.add("// " + to_string(j) + " = " + node->get_name() +
	    (mod->isSink()
	     ? " [" + mod->get_inputType()->name + "]" 
	     : ""));
    }
  f.add("// -1 = MERCATOR scheduler");
  f.add("");
  
  // generate app-level parameter structure
  genHostAppParameterStruct(app, f);  
  f.add("");
  
  // generate accessor for app-level parameters
  f.add(genFcnHeader("AppParams*",
		     "getParams",
		     ""));
  f.add("{ return &allParams.appParams; }");
  f.add("");
  
  if (app->sourceKind == App::SourceBuffer)
    {
      string sourceType = 
	app->sourceNode->get_moduleType()->get_inputType()->name;
      
      f.add(genFcnHeader("void", "setSource", 
			 "const Mercator::Buffer<" + sourceType + "> &buffer"));
      f.add("{ allParams.sourceBufferData = buffer.getData(); }");
    }
  else
    {
      f.add(genFcnHeader("void", "setNInputs", "size_t size"));
      f.add("{ allParams.nInputs = size; }");
    }
  
  f.add("");
			 
  // generate each module's host-side class
  for (const ModuleType *mod : app->modules)
    {
      genHostModuleClass(mod, f);
      f.add("");
      
      // generate any per-node classes for this module
      if (mod->nodeParams.size() > 0)
	genHostNodeClasses(mod, f);
    }
  
  // generate combined structure of all app/module parameters
  genHostAppAllParameters(app, f);
  f.add("");
  
  f.add("public:", true);
    
  // create instance of each host-side module, which provides
  // read/write access to its parameters
  for (const ModuleType *mod : app->modules)
    {
      if (mod->hasModuleParams())
	f.add(mod->get_name() + " " + mod->get_name() + ";");
      
      // generate any per-node instances for this module
      if (mod->hasNodeParams())
	{
	  for (const Node *node : mod->nodes)
	    f.add(node->get_name() + " " + node->get_name() + ";");
	}
    }
  f.add("");
  
  // declare constructor (defined in separate file)
  f.add(genFcnHeader("", app->name, 
		     "cudaStream_t stream = 0, int deviceId = -1") + ";");
  f.add("");

  // getNBlocks() function calls the app driver to get # of active blocks
  f.add(genFcnHeader("int", "getNBlocks", ""));
  f.extend(" const");
  f.add("{ return driver->getNBlocks(); }");
  f.add("");
  
  // run function calls the app driver with the current state of the
  // parameter structure
  f.add(genFcnHeader("void", "run", ""));
  f.add("{ driver->run(&allParams); }");
  f.add("");

  // runAsync function is like run but returns before kernel is complete
  f.add(genFcnHeader("void", "runAsync", ""));
  f.add("{ driver->runAsync(&allParams); }");
  f.add("");

  // join function waits for a kernel launched with runAsync
  f.add(genFcnHeader("void", "join", ""));
  f.add("{ driver->join(); }");
  f.add("");
  
  // destructor cleans up parameter struct and driver
  f.add(genFcnHeader("", "~" + app->name, ""));
  f.add("{");
  f.indent();

  f.add("delete driver;");
  
  f.unindent();
  f.add("}");
  f.add("");
  
  f.add("private:", true);

  // our combined parameter structure
  f.add("Params allParams;");
  f.add("");
  
  // driver to launch device application kernels
  f.add("Mercator::AppDriverBase<Params> *driver;");
  f.add("");
  
  f.unindent();
  f.add("}; // end class " + app->name); // end of class
  
  f.add("#endif"); // end of include guard
  
  f.emit(hostClassFileName);
}

//
// @brief generate the constructor for the host app in its own file
//   (to avoid contaminating the host app header with device code)
//
// @param hostClassFileName name of file to write to
// @param app application being codegen'd
//
void genHostAppConstructor(const string &hostClassFileName,
			   const App *app)
{
  Formatter f;
  
  string DeviceAppClass = app->name + "_dev";
  
  f.add(genUserInclude(DeviceAppClass + ".cuh"));
  f.add("");
  
  f.add(genUserInclude("hostCode/AppDriver.cuh"));
  f.add("");
  
  // constructor connects all modules to their pieces of the
  // combined parameter structure
  f.add(genFcnHeader("", app->name + "::" + app->name, 
		     "cudaStream_t stream, int deviceId"));
  f.add(" :");
  f.indentAfter(':');
  
  bool firstParam = true;
  for (unsigned int j = 0; j < app->modules.size(); j++)
    {
      const ModuleType *mod = app->modules[j];
      
      if (mod->hasModuleParams())
	{
	  f.add((firstParam ? " " : ", ") + mod->get_name() + 
		"(&allParams.p" + mod->get_name() + ")");
	  firstParam = false;
	}
      
      // initialize any node objects from their module object
      if (mod->hasNodeParams())
	{
	  for (unsigned int k = 0; k < mod->nodes.size(); k++)
	    {
	      const Node *node = mod->nodes[k];
	      f.add((firstParam ? " " : ", ") + node->get_name() + 
		    "(&allParams.n" + mod->get_name() + 
		    "[" + to_string(k) + "])");
	      firstParam = false;
	    }
	}
    }
  
  f.unindent();
  f.add("{");
  f.indent();
  
  f.add("if (deviceId == -1) cudaGetDevice(&deviceId);");
  f.add("driver = new Mercator::AppDriver<Params, " + DeviceAppClass + ">(stream, deviceId);");
  
  f.unindent();
  f.add("}");
  
  f.emit(hostClassFileName);
}
