//
// @file gen_deviceapp_class.cc
// @brief code-gen a MERCATOR app on the device
//
// MERCATOR
// Copyright (C) 2018 Washington University in St. Louis; all rights reserved.
//

#include <iostream>

#include "gen_deviceapp_class.h"
#include "gen_deviceapp_ctor.h"

#include "app.h"

#include "Formatter.h"
#include "codegen_utils.h"

#include "options.h"

using namespace std;

//
// @brief return the argument list for a module's run() function
//
// @param mod Module whose code is being generated
// @return generated argument list string
//
static string
genDeviceModuleRunFcnParams(const ModuleType *mod)
{
  // run fcn parameters
  string inputType = mod->get_inputType()->name;
  
  string runFcnParams;
  if(mod->get_nElements() > 1)
    {
      runFcnParams = 
	"const " + inputType + "* inputItems, " 
	"InstTagT* nodeIdxs";
    }
  else
    {
      runFcnParams = 
	"const " + inputType + "& inputItem, " 
	"InstTagT nodeIdx";
    }
  
  return runFcnParams;
}

//
// @brief return the argument list for a module's begin() and end() functions
//
// @return generated argument list string
//
static string
genDeviceModuleBeginEndFcnParams()
{
  // run fcn parameters
  //string inputType = mod->get_inputType()->name;
  
  string fcnParams = "InstTagT nodeIdx";
  /*
  if(mod->get_nElements() > 1)
    {
      runFcnParams = 
	"const " + inputType + "* inputItems, " 
	"InstTagT* nodeIdxs";
    }
  else
    {
      runFcnParams = 
	"const " + inputType + "& inputItem, " 
	"InstTagT nodeIdx";
    }
  */
  return fcnParams;
}

//
// @brief generate the base type for a module, based on its properties
//
// @param mod Pointer to module for which to generate typep
// @return generated type string
//
static
string genDeviceModuleBaseType(const ModuleType *mod)
{
  string baseType;
  
  if (mod->isSource())
    {
      baseType =
	"ModuleType_Source"
	"<" + mod->get_channel(0)->type->name
	+ ", " + to_string(mod->get_nChannels())
	+ ", THREADS_PER_BLOCK>";
    }
  else
    {
      // get data type for this module
      string inTypeString = mod->get_inputType()->name;
      
      if (mod->isSink())
	{
	  baseType =
	    "ModuleType_Sink"
	    "<" + inTypeString 
	    + ", " + to_string(mod->nodes.size())
	    + ", THREADS_PER_BLOCK>";
	}
      else // regular module
	{
	  string moduleTypeVariant;
	  
	  // stimcheck: Add special enumerate module type to the list of regular modules
	  if (mod->get_isEnumerate()) {
	    moduleTypeVariant = "ModuleType_Enumerate";
	  }
	  else if (mod->get_nElements() > 1) {
	    moduleTypeVariant = "ModuleType_ManyItems";
	  }
	  else {
	    moduleTypeVariant = "ModuleType_SingleItem";
	  }
	  
	  baseType =
	    moduleTypeVariant
	    + "<" + inTypeString 
	    + ", " + to_string(mod->nodes.size())
	    + ", " + to_string(mod->get_nChannels())
	    + ", " + to_string(mod->get_nThreads())
	    + ", " + to_string(mod->get_inputLimit())
	    + ", " + to_string(mod->get_useAllThreads()) //runWithAllThreads
	    + ", THREADS_PER_BLOCK"
	    ", " + mod->get_name() // for CRTP
	    + ">";
	}
    }
  
  return baseType;
}


//
// @brief generate statements that initialize the channels of
//   a module, for use inside its constructor
//
// @param mod Module whose code is being generated
// @f Formatter to receive generated code
static
void genDeviceModuleChannelInitStmts(const ModuleType *mod,
				     Formatter &f)
{
  // create output channels
  int nChannels = mod->get_nChannels();
  if (nChannels > 0) // false for SINK modules
    {
      // init output channels
      for (int j=0; j < nChannels; ++j)
	{
	  const Channel *channel = mod->get_channel(j);
	  
	  f.add("{");
	  f.indent();
	  
	  // Create const array of reservedSlot counts for each
	  // queue targeted by this channel (one per node)
	  string nextStmt = "const unsigned int reservedSlots[] = {";
	  for (const Node *node : mod->nodes)
	    {
	      const Edge *dsEdge = node->get_dsEdge(j);
	      unsigned int reserved = 
		(dsEdge ? dsEdge->dsReservedSlots : 0);
	      
	      nextStmt += to_string(reserved) + ", ";
	    }
	  nextStmt += "};";
	  f.add(nextStmt);
	  
	  // format: module->initChannel<type>(outstream-enum,
	  //   outputsPerInput, capacity);
	  f.add("initChannel<"
		+ channel->type->name + ">("
		+ "Out::" + channel->name + ", "
		+ to_string(channel->maxOutputs) + ", "
		+ "reservedSlots);");
	  
	  f.unindent();
	  f.add("}");
	  
	  f.add("");
	}
    }
}


//
// @brief Build a module's enumeration of its output channels.
//
// @param mod module whose code is being generated
// @param f Formatter to receive generated code
//
static 
void genDeviceModuleOutChannelEnum(const ModuleType *mod,
				   Formatter &f)
{
  f.add("struct Out {");
  f.indent();

  f.add("enum {");
  f.indent();

  // emit enums with names for all channels
  int nChannels = mod->get_nChannels();
  for (int i = 0; i < nChannels; ++i)
    {
      string outName = mod->get_channel(i)->name;
      
      f.add(outName + " = " + to_string(i) + ",");
    } 
  
  f.unindent();
  f.add("};");
  
  f.unindent();
  f.add("};");
}


//
// @brief generate constructor for device-side module
//
// @param app app being codegen'd
// @param mod module whose code is being generated
// @param f Formatter to receive result
//
static
void genDeviceModuleConstructor(const App *app,
				const ModuleType *mod,
				Formatter &f)
{
  string baseType = genDeviceModuleBaseType(mod);
  
  struct Arg  { string type; string name; };
  struct Init { string name; string initExpr; };
  
  vector<Arg>  args;
  vector<Init> inits;
  vector<string> baseArgs;
  
  // source takes global tail pointer
  if (mod->isSource())
    {
      args.push_back({"size_t *", "tailPtr"});
      baseArgs.push_back("tailPtr");
    }
  
  // all modules take array of queue sizes
  args.push_back({"const unsigned int*", "queueSizes"});
    
  // modules with parameters have mod parameter accessor
  if (mod->hasParams())
    {
      string paramsType = app->name + "::" + mod->get_name() + "::Params";
      args.push_back({"const " + paramsType + "*", "iparams"});
      
      inits.push_back({"params", "iparams"});
    }
  
  // in apps with aprams, all modules have app parameter accessor
  if (app->hasParams())
    {
      string paramsType = app->name + "::" + "AppParams";
      args.push_back({"const " + paramsType + "*", "iappParams"});
      
      inits.push_back({"appParams", "iappParams"});
    }
  
  // all modules but the source pass the queue sizes to their base type
  if (!mod->isSource())
    baseArgs.push_back("queueSizes");
  
  // emit function declaration
  f.add("__device__");  
  f.add(mod->get_name() + "(");
  f.indentAfter('(',0);
  f.indent(1);
  for (unsigned int j = 0; j < args.size(); j++) // function arguments
    {
      f.add(args[j].type + " " + args[j].name +
	    (j == args.size() - 1 ? "" : ","));
    }
  f.unindent();
  f.add(")");
  f.unindent();
  
  f.indent();
  
  // emit call to base type constructor
  f.add(": " + baseType + "(");
  f.indentAfter('(',0);
  f.indent(1);
  for (unsigned int j = 0; j < baseArgs.size(); j++) // base type fcn arguments
    {
      f.add(baseArgs[j] + 
	    (j == baseArgs.size() - 1 ? "" : ","));
    }
  f.unindent();
  f.add(")");
  f.unindent();
  
  // emit initialization of other params
  if (inits.size() > 0)
    {
      f.extend(",");
      
      for (unsigned int j = 0; j < inits.size(); j++) // fcn arguments
	{
	  f.add(inits[j].name + "(" + inits[j].initExpr + ")" +
		(j == inits.size() - 1 ? "" : ","));
	}
    }
  
  f.unindent();
  
  f.add("{");
  f.indent();
  
  // initialize device module's output channels
  genDeviceModuleChannelInitStmts(mod, f);
  
  // stimcheck: initialize flag for enumerate, for debug purposes only
  if(mod->get_isEnumerate())
    {
       f.add("setEnum(true);");
    }
  f.unindent();
  f.add("}");
}


//
// @brief generate a device-side module class
//
// @param mod Module whose code is being generated
// @param f Formatter to receive generated code
//
static
void genDeviceModuleClass(const App *app,
			  const ModuleType *mod,
			  Formatter &f)
{
  string baseType = genDeviceModuleBaseType(mod);
  string classHeader = 
    mod->get_name() + " final "
    ": public Mercator::" + baseType;
  
  f.add("class " + classHeader + " {");
  f.indent();
  
  f.add("public:", true);
  
  if (mod->get_nChannels() > 0)
    {
      f.add("// enum defining output channels");
      genDeviceModuleOutChannelEnum(mod, f);
      f.add("");
    }
  
  // make host-side node enumeration available to user functions in the
  // device-side class (useful mainly for state initialization)
  f.add("typedef " + app->name + "::" + mod->get_name() + "::Node Node;");
  f.add("");
  
  // constructor
  genDeviceModuleConstructor(app, mod, f);
  f.add("");
  
  // stimcheck: In addition to Sources and Sinks, DO NOT make a run stub for Enumerates.
  if (!mod->isSource() && !mod->isSink() && !mod->get_isEnumerate())
    {
      // run function (public because of CRTP)
      f.add("__device__");
      f.add(genFcnHeader("void",
			 "run", 
			 genDeviceModuleRunFcnParams(mod)) + ";");
      f.add("");
    }
  
  // stimcheck: Add begin and end functions to the codegened headers, make
  // them blank stubs as necessary too (when unused).
  if (!mod->isSource() && !mod->isSink())
    {
      // begin and end functions (public because of CRTP)
      if (app->isPropagate.at(mod->get_idx()))
	{
	   // create headers for begin and end
   	   f.add("__device__");
           f.add(genFcnHeader("void",
			      "begin", 
			      genDeviceModuleBeginEndFcnParams()) + ";");
           f.add("");
   	   f.add("__device__");
           f.add(genFcnHeader("void",
			      "end", 
			      genDeviceModuleBeginEndFcnParams()) + ";");
           f.add("");
   	   f.add("__device__");
		//printf("FROM NAME: %s\n", mod->get_inputType()->from->name.c_str());
		cout << "BEFORE PRINT" << endl;
		if(mod->get_inputType()->from == nullptr)
			cout << "NULL FROM" << endl;
		if(mod->get_inputType()->from->name.empty())
			cout << "EMPTY FROM TYPE" << endl;
		cout << "FROM NAME: " << mod->get_inputType()->from->name << endl;
           f.add(genFcnHeader(mod->get_inputType()->from->name + "*",
			      "getParent", 
			      genDeviceModuleBeginEndFcnParams()));
	   f.add("{ return static_cast< " + mod->get_inputType()->from->name + "* >(currentParent[nodeIdx]); }");
           f.add("");
	}
      else
	{
	   // create empty stubs for modules without enumIds.
   	   f.add("__device__");
           f.add(genFcnHeader("void",
			      "begin", 
			      genDeviceModuleBeginEndFcnParams()) + "{ }");
           f.add("");
   	   f.add("__device__");
           f.add(genFcnHeader("void",
			      "end", 
			      genDeviceModuleBeginEndFcnParams()) + "{ }");
           f.add("");
	}
    }
	cout << "FINISHED ENUM STUB GEN. . ." << endl;

  // stimcheck: Add findCount function header to the codegened headers of
  // enumerate modules.
  if (!mod->isSource() && !mod->isSink() && mod->get_isEnumerate())
    {
      // findCount function (public because of CRTP)
      f.add("__device__");
      f.add(genFcnHeader("unsigned int",
			 "findCount", 
			 genDeviceModuleBeginEndFcnParams()) + ";");
      f.add("");
    }

  f.add("private:", true);
  f.add("");
  
  if (!mod->isSource() && !mod->isSink())
    {
      // generate reflectors for user code to learn about this module
      f.add("using " + baseType + "::getNumInstances;");
      f.add("using " + baseType + "::getNumActiveThreads;");
      f.add("using " + baseType + "::getThreadGroupSize;");
      f.add("using " + baseType + "::isThreadGroupLeader;");
      
      f.add("");
    }
  
  if (mod->hasParams())
    {
      // generate module parameter storage
      
      string paramsType = app->name + "::" + mod->get_name() + "::Params";
      
      f.add("const " + paramsType + "* const params;");
      f.add("");
      
      // generate module parameter accessor
      
      f.add("__device__");
      f.add(genFcnHeader("const " + paramsType + "*",
			 "getParams",
			 "") + " const");
      f.add("{ return params; }");
      f.add("");
    }
  
  if (app->hasParams())
    {
      // generate app-wide parameter storage
      string paramsType = app->name + "::AppParams";
      
      f.add("const " + paramsType + "* const appParams;");
      f.add("");
      
      // generate app-wide parameter accessor
      f.add("__device__");
      f.add(genFcnHeader("const " + paramsType + "*",
			 "getAppParams",
			 "") + " const");
      f.add("{ return appParams; }");
      f.add("");
    }
  

  if (mod->hasState())
    {
      // generate module state structure
      
      f.add("struct State {");
      f.indent();
      
      for (const DataItem *var : mod->nodeState)
	{
	  f.add(var->type->name + " " + var->name + 
		"[" + to_string(mod->nodes.size()) + "];");
	}
      
      f.unindent();
      f.add("};");
      
      f.add("");
      
      // generate state accessor
      
      f.add("__device__");
      f.add(genFcnHeader("State*",
			 "getState",
			 ""));
      f.add("{ return &state; }");
      
      f.add("");
      
      // generate state storage
      f.add("State state;");
      
      f.add("");
      
      if (!mod->isSource() && !mod->isSink())
	{
	  // generate hook functions (filled by user)
	  f.add("__device__ void init(); // called once per block before run()");
	  f.add("__device__ void cleanup(); // called once per block after run()");
	}
    }
  
  if (mod->isSource())
    {
      // initialize the source from the data passed down from the host
      f.add("__device__");
      f.add(genFcnHeader("void", "init", ""));
      f.add("{");
      f.indent();
      
      f.add("if (threadIdx.x == 0)");
      f.add("{");
      f.indent();
      
      f.add("state.source[0] = createSource(params->sourceData[0], &state.sourceMem[0]);");
      f.add("setInputSource(state.source[0]);");
      
      f.unindent();
      f.add("}");
      
      f.unindent();
      f.add("}");
  
      f.add("");
    }
  else if (mod->isSink())
    {
      // initialize the sink from the data passed down from the host
      f.add("__device__");
      f.add(genFcnHeader("void", "init", ""));
      f.add("{");
      f.indent();
      
      f.add("if (threadIdx.x < getNumInstances())");
      f.add("{");
      f.indent();
      
      f.add("state.sink[threadIdx.x] = createSink(params->sinkData[threadIdx.x], &state.sinkMem[threadIdx.x]);");
      f.add("setOutputSink(threadIdx.x, state.sink[threadIdx.x]);");
      
      f.unindent();
      f.add("}");
      
      f.unindent();
      f.add("}");
  
      f.add("");
    } 
  
  f.unindent(); 
  f.add("}; // end class " + mod->get_name());
}



//
// @brief generate the entire device-side header for a MERCATOR app
//
void genDeviceAppHeader(const string &deviceClassFileName,
			const App *app)
{
  Formatter f;
  
  string DeviceAppClass = app->name + "_dev";
  
  // add include guard header for device
  {
    string incGuard = genIncludeGuardName(DeviceAppClass);
    f.add("#ifndef " + incGuard);
    f.add("#define " + incGuard);
    f.add("");    
  }
  
  f.add(genUserInclude(app->name + ".cuh"));
  f.add("");
  
  f.add(genUserInclude("deviceCode/DeviceApp.cuh"));
  
  {
    // include only the module type specializations needed by the app
    bool needsSingleItem = false, needsMultiItem = false, needsEnumerate = false;
    
    for (const ModuleType *mod : app->modules)
      {
	if (mod->isSource() || mod->isSink())
	  continue;
	else if (mod->get_nElements() > 1)
	  needsMultiItem = true;
	else 
	  needsSingleItem = true;

	if (mod->get_isEnumerate())
	  needsEnumerate = true;
      }
    
    if (needsSingleItem)
      f.add(genUserInclude("deviceCode/ModuleType_SingleItem.cuh"));
    
    if (needsMultiItem)
      f.add(genUserInclude("deviceCode/ModuleType_MultiItem.cuh"));

    if (needsEnumerate)
      f.add(genUserInclude("deviceCode/ModuleType_Enumerate.cuh"));
    
    f.add(genUserInclude("deviceCode/ModuleType_Source.cuh"));
    f.add(genUserInclude("deviceCode/ModuleType_Sink.cuh"));
    f.add("");
  }
  
  // begin device app class
  f.add("class " + DeviceAppClass +
	" : public Mercator::DeviceApp<" 
	+ app->name + "::NUM_MODULES, "
	+ to_string(options.threadsPerBlock) + "," 
	+ to_string(options.deviceStackSize) + ","
	+ to_string(options.deviceHeapSize)
	+ "> {");
  f.indent();
  
  // generate class def for each module type
  for (const ModuleType *mod : app->modules)
    {
      genDeviceModuleClass(app, mod, f);
      f.add("");
    }

  f.add("public:", true);
  
  // generate constructor for entire app
  genDeviceAppConstructor(app, f);
  
  f.unindent();
  f.add("}; // end class " + DeviceAppClass);
  
  f.add("#endif"); // end of include guard  

  f.emit(deviceClassFileName);
}


//////////////////////////////////////////////////////////////////////
// SKELETON WITH USER FUNCTIONS
/////////////////////////////////////////////////////////////////////

//
// @brief generate the device-side skeleton file with run() functions
//
void genDeviceAppSkeleton(const string &skeletonFileName,
			  const App *app)
{
  Formatter f;
  
  string DeviceAppClass = app->name + "_dev";
  
  // user's skeleton should include the codegen'd class declaration
  f.add(genUserInclude(DeviceAppClass + ".cuh"));
  f.add("");
  
  // generate class def for each module type
  for (const ModuleType *mod : app->modules)
    {
      if (mod->isSource() || mod->isSink())
	continue;
      
      if (mod->hasState())
	{
	  // generate init function 
	  f.add("__device__");
	  f.add(genFcnHeader("void",
			     DeviceAppClass + "::\n" + 
			     mod->get_name() + "::init", 
			     ""));
	  
	  f.add("{");
	  f.add("");
	  f.add("}");
	  
	  f.add("");
	}

      if (app->isPropagate.at(mod->get_idx()))
	{
	  //generate begin function
	  f.add("__device__");
	  f.add(genFcnHeader("void",
			     DeviceAppClass + "::\n" + 
			     mod->get_name() + "::begin", 
			     genDeviceModuleBeginEndFcnParams()));
	  
	  f.add("{");
	  f.add("");
	  f.add("}");
	  
	  f.add("");
	}
      
      // stimcheck: Generate run functions for every module EXCEPT ENUMERATES.
      if (!(mod->get_isEnumerate()))
	{
          // generate run function
          f.add("__device__");
          f.add(genFcnHeader("void",
			     DeviceAppClass + "::\n" + 
			     mod->get_name() + "::run", 
			     genDeviceModuleRunFcnParams(mod)));
      
          f.add("{");
          f.indent();
      
          f.add("");
      
          f.unindent();
          f.add("}");
      
          f.add("");
	}

      if (mod->get_isEnumerate())
	{
	  //generate findCount function
	  f.add("__device__");
	  f.add(genFcnHeader("unsigned int",
			     DeviceAppClass + "::\n" + 
			     mod->get_name() + "::findCount", 
			     genDeviceModuleBeginEndFcnParams()));
	  
	  f.add("{");
	  f.add("\treturn 0;\t//Replace this return with the number of elements found for this enumeration.");
	  f.add("}");
	  
	  f.add("");
	}
      
      if (app->isPropagate.at(mod->get_idx()))
	{
	  //generate end function
	  f.add("__device__");
	  f.add(genFcnHeader("void",
			     DeviceAppClass + "::\n" + 
			     mod->get_name() + "::end", 
			     genDeviceModuleBeginEndFcnParams()));
	  
	  f.add("{");
	  f.add("");
	  f.add("}");
	  
	  f.add("");
	}

      if (mod->hasState())
	{
	  // generate cleanup function
	  f.add("__device__");
	  f.add(genFcnHeader("void",
			     DeviceAppClass + "::\n" + 
			     mod->get_name() + "::cleanup", 
			     ""));
	  
	  f.add("{");
	  f.add("");
	  f.add("}");
	  
	  f.add("");
	}
    }
  
  f.emit(skeletonFileName);
}
