//
// PARSE.CC
// Parser interface for spec files
//
// MERCATOR
// Copyright (C) 2018 Washington University in St. Louis; all rights reserved.
//

#include "parse.h"

#include <unordered_set>

#include "parser-driver.h"
#include "typecheck.h"

using namespace std;

//#define DEBUG_PARSING
//#define DEBUG_SCANNING

// local prototypes
static
ASTContainer *
scrapeTypes(const input::AppSpec *app,
	    const vector<string> &references,
	    const vector<string> &includePaths);


vector<input::AppSpec *> 
parseInput(const string &sourceFile,
	   const vector<string> &typecheckIncludePaths,
	   vector<string> &references)
{
  mercator_driver driver;
  
#ifdef DEBUG_PARSING
  driver.set_trace_parsing(true);
#endif

#ifdef DEBUG_SCANNING
  driver.set_trace_scanning(true);
#endif

  if (driver.parse(sourceFile)) // parse failed
    abort();
  
  //
  // Obtain source file's directory, which is automatically
  // part of the search path for referenced includes
  // without a fully-qualified pathname.
  //
  string base;
  size_t endPosn = sourceFile.find_last_of('/');
  if (endPosn == string::npos)
    base = "./";
  else
    base = sourceFile.substr(0, endPosn + 1);
  
  vector<string> incPaths = typecheckIncludePaths;
  incPaths.push_back(base);
  
  for (input::AppSpec *app : driver.apps)
    {  
      app->typeInfo = scrapeTypes(app, driver.refs, incPaths);
    }
  
  references = driver.refs;
  return driver.apps;
}


static
ASTContainer *
scrapeTypes(const input::AppSpec *app,
	    const vector<string> &references,
	    const vector<string> &includePaths)
{
  // strings of the types present in the current app
  unordered_set<string> typeStrings;
  
  // get types from modules
  for (const input::ModuleTypeStmt *module : app->modules)
    {
      // get input type
      typeStrings.insert(module->inputType->name);
      if (module->inputType->from) 
	typeStrings.insert(module->inputType->from->name);
      
      // get types from channels
      for (const input::ChannelSpec *channel : module->channels)
	{
	  typeStrings.insert(channel->type->name);
	  if (channel->type->from) 
	    typeStrings.insert(channel->type->from->name);
	}
    }
  
  // get types from source or sink nodes
  for (const input::NodeStmt *node : app->nodes)
    {
      if (node->type->kind == input::NodeType::isSource ||
	  node->type->kind == input::NodeType::isSink)
	{
	  typeStrings.insert(node->type->dataType->name);
	}
    }
  
  // get types from app, module, and node params and state
  for (const input::DataStmt *var : app->vars)
    {
      typeStrings.insert(var->type->name);
      assert (!var->type->from);  // these are never from types
    }
  
  //Construct the ASTContainer
  ASTContainer *a = ASTContainer::create();
  
  vector<string> typeStringVec;
  for (const string &s : typeStrings)
    typeStringVec.push_back(s);
  
  //Build the AST with all the instantiated types found 
  a->findTypes(typeStringVec, references, includePaths);
  
  return a;
}
