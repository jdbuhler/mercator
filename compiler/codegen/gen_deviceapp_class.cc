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
	"const " + inputType + "* inputItems";
    }
  else
    {
      runFcnParams = 
	"const " + inputType + "& inputItem";
    }
  
  if (mod->get_useAllThreads())
    {
      runFcnParams += ", unsigned int nInputs";
    }
  
  return runFcnParams;
}


//
// @brief generate the base type for a module, based on its properties
//
// @param mod Pointer to module for which to generate type
// @return generated type string
//
static
string genDeviceModuleBaseType(const ModuleType *mod)
{
  string baseType;
  
  // get data type for this node
  string inTypeString = mod->get_inputType()->name;
  
  if (mod->isSink())
    {
      baseType =
	"NodeFunction_Sink"
	"<" + inTypeString
	+ ", InputView"
	", THREADS_PER_BLOCK>";
    }
  else if (mod->isEnumerate())
    {
      baseType = 
	"NodeFunction_Enumerate<" + inTypeString
	+ ", InputView"
	    ", THREADS_PER_BLOCK"
	+ ", " + mod->get_name() // for CRTP
	+ ">";
    }
  else
    {
      string moduleTypeVariant;
      
      if (!mod->get_useAllThreads()) //runWithAllThreads
	moduleTypeVariant = "NodeFunction_Buffered";
      else
	moduleTypeVariant = "NodeFunction_User";
      
      baseType =
	moduleTypeVariant
	+ "<" + inTypeString
	+ ", " + to_string(mod->get_nChannels())
	+ ", InputView"
	", THREADS_PER_BLOCK"
	", " + to_string(mod->get_nThreads())
	+ ", " + to_string(mod->get_inputLimit())
	+ ", " + mod->get_name() // for CRTP
	+ ">";
    }
  
  return baseType;
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
  string baseType = "Mercator::" + genDeviceModuleBaseType(mod);
  
  struct Arg  { string type; string name; };
  struct Init { string name; string initExpr; };
  
  vector<Arg>  args;
  vector<Init> inits;
  vector<string> baseArgs;
  
  args.push_back({"Mercator::RefCountedArena *", "parentArena"});
  baseArgs.push_back("parentArena");
  
  if (mod->isEnumerate())
    {
      args.push_back({"unsigned int", "enumId"});
      baseArgs.push_back("enumId");
    }
  
  // modules with per-node parameters have a node parameter accessor
  if (mod->hasNodeParams())
    {
      string paramsType = app->name + "::" + mod->get_name() +
	"::NodeParams";
      
      args.push_back({"const " + paramsType + "*", "inodeParams"});
      
      inits.push_back({"nodeParams", "inodeParams"});
    }
  
  // modules with per-module parameters have a module parameter accessor
  if (mod->hasModuleParams())
    {
      string paramsType = app->name + "::" + mod->get_name() +
	"::ModuleParams";
      
      args.push_back({"const " + paramsType + "*", "imoduleParams"});
      
      inits.push_back({"moduleParams", "imoduleParams"});
    }
  
  // in apps with params, all modules have app parameter accessor
  if (app->hasParams())
    {
      string paramsType = app->name + "::" + "AppParams";
      args.push_back({"const " + paramsType + "*", "iappParams"});
      
      inits.push_back({"appParams", "iappParams"});
    }
  
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
  f.add("{}");
}


//
// @brief generate a device-side NodeFunction class
//
// @param mod node type for which we are generating class
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

  f.add("template <template<typename> typename InputView>");
  f.add("class " + classHeader + " {");
  f.indent();
  
  f.add("public:", true);
  
  if (mod->get_nChannels() > 0)
    {
      f.add("// enum defining output channels");
      genDeviceModuleOutChannelEnum(mod, f);
      f.add("");
    }
  
  // constructor
  genDeviceModuleConstructor(app, mod, f);
  f.add("");
  
  if (mod->isUser())
    {
      // run function (public because of CRTP)
      f.add("__device__");
      f.add(genFcnHeader("void",
			 "run", 
			 genDeviceModuleRunFcnParams(mod)) + ";");
      f.add("");
    }

  
  if (mod->isEnumerate())
    {
      string inputType = mod->get_inputType()->name;
      
      f.add("__device__");
      f.add(genFcnHeader("unsigned int",
			 "findCount", 
			 "const " + inputType + " &parent") + " const;");
      
      f.add("");
    }
  
  f.add("private:", true);
  f.add("");
  
  if (mod->isUser())
    {
      // generate reflectors for user code to learn about this module
      f.add("using Mercator::" + baseType + "::getNumActiveThreads;");
      f.add("using Mercator::" + baseType + "::getThreadGroupSize;");
      f.add("using Mercator::" + baseType + "::isThreadGroupLeader;");
      f.add("using Mercator::" + baseType + "::push;");
      
      f.add("");
    }

  const DataType *fromType = 
    (mod->isSink()
     ? nullptr
     : mod->get_inputType()->from);
  
  // If module has a from type for its input, it must be in
  // an enumeration region -- generate functions specific to 
  // enumerated modules (begin(), end(), getParent())
  if (fromType)
    {
      f.add("public:");
      
      f.add("__device__");
      f.add(genFcnHeader("void",
			 "begin",
			 "") +";");
      f.add("");
      f.add("__device__");
      f.add(genFcnHeader("void",
			 "end", 
			 "") +";");
      f.add("");
      
      f.add("private:");
      
      f.add("__device__");
      f.add(genFcnHeader(fromType->name + "*",
			 "getParent",
			 "") + " const");
      f.add("{");
      f.indent();
      
      string pbType = 
	"Mercator::ParentBuffer<" + fromType->name + ">";
      
      f.add(pbType + " *pb = static_cast<" + 
	    pbType + " *>(this->parentArena);");
      f.add("return pb->get(this->parentIdx);");
      
      f.unindent();
      f.add("}");
      f.add("");
    }
    
  if (mod->hasNodeParams())
    {
      // generate node parameter storage
      
      string paramsType = app->name + "::" + mod->get_name() + 
	"::NodeParams";
      
      f.add("const " + paramsType + "* const nodeParams;");
      f.add("");
      
      // generate node parameter accessor
      
      f.add("__device__");
      f.add(genFcnHeader("const " + paramsType + "*",
			 "getParams",
			 "") + " const");
      f.add("{ return nodeParams; }");
      f.add("");
    }
  
  if (mod->hasModuleParams())
    {
      // generate module parameter storage
      
      string paramsType = app->name + "::" + mod->get_name() + 
	"::ModuleParams";
      
      f.add("const " + paramsType + "* const moduleParams;");
      f.add("");
      
      // generate module parameter accessor
      
      f.add("__device__");
      f.add(genFcnHeader("const " + paramsType + "*",
			 "getModuleParams",
			 "") + " const");
      f.add("{ return moduleParams; }");
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
      // generate node state structure
      
      f.add("struct NodeState {");
      f.indent();
      
      for (const DataItem *var : mod->nodeState)
	{
	  f.add(var->type->name + " " + var->name + ";");
	}
      
      f.unindent();
      f.add("};");
      
      f.add("");
      
      // generate state accessor
      
      f.add("__device__");
      f.add(genFcnHeader("NodeState*",
			 "getState",
			 ""));
      f.add("{ return &state; }");
      
      f.add("");
      
      // generate state storage
      f.add("NodeState state;");
      
      f.add("");
      
      f.add("public:");
      
      // generate hook functions (filled by user)
      f.add("__device__ void init(); // called once per block before run()");
      f.add("__device__ void cleanup(); // called once per block after run()");
    }
  
  if (mod->isSink())
    {
      f.add("public:");
      
      // initialize the sink from the data passed down from the host
      f.add("__device__");
      f.add(genFcnHeader("void", "init", ""));
      f.add("{");
      f.indent();

      f.add("if (threadIdx.x == 0)");
      f.add("{");
      f.indent();
      
      f.add("this->sink.init(nodeParams->sinkData.bufferData);");
      
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
    bool needsUser      = false;
    bool needsBuffered  = false;
    bool needsEnumerate = false;
    
    for (const ModuleType *mod : app->modules)
      {
	if (mod->isSink())
	  continue;
	else if (mod->isEnumerate())
	  needsEnumerate = true;
	else if (!mod->get_useAllThreads())
	  needsBuffered = true;
	else
	  needsUser = true;
      }
    
    if (needsUser)
      f.add(genUserInclude("deviceCode/NodeFunction_User.cuh"));
    
    if (needsBuffered)
      f.add(genUserInclude("deviceCode/NodeFunction_Buffered.cuh"));
    
    if (needsEnumerate)
      f.add(genUserInclude("deviceCode/NodeFunction_Enumerate.cuh"));
    
    f.add(genUserInclude("deviceCode/NodeFunction_Sink.cuh"));
    
    f.add(genUserInclude("deviceCode/Node_Source.cuh"));
    f.add(genUserInclude("deviceCode/Node.cuh"));
    f.add("");
  }
  
  f.add(genUserInclude("deviceCode/Scheduler_impl.cuh"));
  f.add("");
  
  // begin device app class
  f.add("class " + DeviceAppClass +
	" : public Mercator::DeviceApp<" 
	+ to_string(app->nodes.size()) + ","
	+ to_string(app->threadWidth) + "," 
	+ to_string(options.deviceStackSize) + ","
	+ to_string(options.deviceHeapSize)
	+ "> {");
  f.indent();
  
  // generate class def for each module type and its nodes
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

  f.add("");
  f.add("#define __MDECL__ \\");
  f.add("   template <template<typename T> typename InputView> __device__ ");
  f.add("");
	
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
			  const vector<App *> apps)
{
  Formatter f;

  for (const App *app : apps)
    {
      string DeviceAppClass = app->name + "_dev";
      
      // user's skeleton should include the codegen'd class declaration
      f.add(genUserInclude(DeviceAppClass + ".cuh"));
    }
  
  f.add("");
  
  for (const App *app : apps)
    {
      string DeviceAppClass = app->name + "_dev";
      
      // generate class def for each module type
      for (const ModuleType *mod : app->modules)
	{
	  string ModClass = mod->get_name() + "<InputView>";
	  
	  if (mod->isSink())
	    continue;
	  
	  if (mod->hasState())
	    {
	      // generate init function 
	      f.add("__MDECL__");
	      f.add(genFcnHeader("void",
				 DeviceAppClass + "::\n" + 
				 ModClass + "::init", 
				 ""));
	      
	      f.add("{");
	      f.add("");
	      f.add("}");
	      
	      f.add("");
	    }
	  
	  
	  if (mod->get_inputType()->from)
	    {
	      //generate begin function
	      f.add("__MDECL__");
	      f.add(genFcnHeader("void",
				 DeviceAppClass + "::\n" + 
				 ModClass + "::begin", 
				 ""));
	      
	      f.add("{");
	      f.add("");
	      f.add("}");
	      
	      f.add("");
	    }
	  
	  // non-enumerate regular modules get run functions
	  if (!mod->isEnumerate())
	    {
	      // generate run function
	      f.add("__MDECL__");
	      f.add(genFcnHeader("void",
				 DeviceAppClass + "::\n" + 
				 ModClass + "::run", 
				 genDeviceModuleRunFcnParams(mod)));
	      
	      f.add("{");
	      f.indent();
	      
	      f.add("");
	      
	      f.unindent();
	      f.add("}");
	      
	      f.add("");
	    }
	  
	  // enumerate modules get findCount()
	  if (mod->isEnumerate())
	    {
	      string fromType = mod->get_inputType()->name;
	      
	      //generate findCount function
	      f.add("__MDECL__");
	      f.add(genFcnHeader("unsigned int",
				 DeviceAppClass + "::\n" + 
				 ModClass + "::findCount", 
				 "const " + fromType + " &parent") + " const");
	      
	      f.add("{");
	      f.add("\treturn 0;\t//Replace this return with the number of elements found for this enumeration.");
	  f.add("}");
	  
	  f.add("");
	    }
	  
	  if (mod->get_inputType()->from)
	    {
	      //generate end function
	      f.add("__MDECL__");
	      f.add(genFcnHeader("void",
				 DeviceAppClass + "::\n" + 
				 ModClass + "::end", 
				 ""));
	      
	      f.add("{");
	      f.add("");
	      f.add("}");
	      
	      f.add("");
	    }
	  
	  if (mod->hasState())
	    {
	      // generate cleanup function
	      f.add("__MDECL__");
	      f.add(genFcnHeader("void",
				 DeviceAppClass + "::\n" + 
				 ModClass + "::cleanup", 
				 ""));
	      
	      f.add("{");
	      f.add("");
	      f.add("}");
	      
	      f.add("");
	    }
	}
    }
  
  f.emit(skeletonFileName);
}
