//
// MERCATOR-PARSER.YY
// Parser for MERCATOR spec files
//
// MERCATOR
// Copyright (C) 2018 Washington University in St. Louis; all rights reserved.

%skeleton "lalr1.cc" /* -*- C++ -*- */
%require "3.0.4"

%defines

%define parser_class_name {mercator_parser}
//%define api.prefix {yy}

%define api.token.constructor
%define api.value.type variant

%code requires
{
#include <string>

#include "inputspec.h"

class mercator_driver;
}

// The parsing context
%param { mercator_driver &driver }

%locations
%initial-action
{
  // Initialize the initial location.
  @$.begin.filename = @$.end.filename = &driver.file;
};

%define parse.assert
%define parse.trace
%define parse.error verbose

%code
{
#include "parser-driver.h"
}

%printer { yyoutput << $$; } <*>;

//////////////////////////////////////////////////////////////////
// TOKENS
//////////////////////////////////////////////////////////////////

%define api.token.prefix {TOK_}

// special symbols
%token
  END  0  "end of file"
  BANG    "!"
  COLON   ":"
  COMMA   ","
  GOESTO  "->"
  GT      ">"
  LT      "<"
  SCOPE   "::"
  SEMICOLON ";"
  STAR    "*"
;

// keywords
%token
  AGGREGATE "aggregate"
  ALLTHREADS "allthreads"
  APPLICATION "application"
  EDGE    "edge"
  ENUMERATE "enumerate"
  FROM    "from"
  ILIMIT  "ilimit"
  MAPPING "mapping"
  MODULE  "module"
  NODE    "node"
  NODEPARAM "nodeparam"
  NODESTATE "nodestate"
  PARAM    "param"
  REFERENCE "reference"
  SINK    "sink"
  SOURCE  "source"
  VOID    "void"
;

// complex tokens
%token <std::string> TYPE   "typeliteral"
%token <int>         NUMBER "number"
%token <std::string> ID     "identifier"
%token <std::string> STRING "string"

// types of nonterminals

%type <std::string> typename_string channelname modulename sourcefilename
%type <std::string> appname nodename varscope edgechannelspec
%type <input::NodeType *> nodetype
//%type <int> maxoutput mappingspec qualifier
%type <int> maxoutput mappingspec
%type <input::DataType *> typename basetypename fromtypename inputtype vartype
%type <input::ChannelSpec *> channel 
%type <std::vector<input::ChannelSpec *> *> channels
%type <input::OutputSpec *> outputtype
%type <input::ModuleTypeStmt *> moduletype
%type <input::DataStmt *> varname scoped_varname;

//////////////////////////////////////////////////////////////////
// GRAMMAR RULES
// NB: C++ variant parsers do NOT have default rule behavior;
// it is necessary to say { $$ = $1; } explicitly to propagate
// a value.
//////////////////////////////////////////////////////////////////

%%

%start stmts;

// a MERCATOR spec is a sequence of statements
stmts:
  %empty
| stmt stmts
;

//////////////
// STATEMENTS
//////////////

stmt:
  referencestmt
| applicationstmt
| modulestmt
| allthreadsstmt
| ilimitstmt
| nodestmt
| edgestmt
| mappingstmt
| paramstmt
| nodeparamstmt
| nodestatestmt
;

// reference stmt: include a C++/CUDA source file to define types
referencestmt:
"reference" sourcefilename ";"           
{ driver.refs.push_back($2); };

sourcefilename: "string" 
{ $$ = $1; };

// application stmt: declare a new MERCATOR application, whose structure
// is defined by subsequent statements
applicationstmt:
"application" appname ";" 
{ driver.createApp($2); };

appname: "identifier" 
{ $$ = $1; };

// module stmt: name a module and define its type
modulestmt:
"module" modulename ":" moduletype ";" 
{ 
  $4->name = $2;
  if (!driver.currApp())
   {
      error(yyla.location, "ModuleType statement outside app context");
      exit(EXIT_FAILURE);
   }
  driver.currApp()->modules.push_back($4);
};

modulename: "identifier"
{ $$ = $1; };

// ilimit stmt: limit maximum number of concurrent inputs to a module
ilimitstmt:
"ilimit" modulename "number" ";"
{
  input::ILimitStmt limit($2, $3);
  if (!driver.currApp())
   {
      error(yyla.location, "ILimit statement outside app context");
      exit(EXIT_FAILURE);
   }
  driver.currApp()->ilimits.push_back(limit);
};

allthreadsstmt:
"allthreads" modulename ";"
{
  input::AllThreadsStmt at($2);
  if (!driver.currApp())
   {
      error(yyla.location, "AllThreads statement outside app context");
      exit(EXIT_FAILURE);
   }
  driver.currApp()->allthreads.push_back(at);
};

mappingstmt:
"mapping" modulename mappingspec ";"
{
  input::MappingStmt mapping($2, std::abs($3), ($3 < 0));
  if (!driver.currApp())
   {
      error(yyla.location, "Mapping statement outside app context");
      exit(EXIT_FAILURE);
   }
  driver.currApp()->mappings.push_back(mapping);
};

mappingspec:
  "number"      { $$ =  $1; } // multiple inputs / thread
| ":" "number"  { $$ = -$2; } // NB: negative means multiple threads / input
;

// node stmt: declare a node as instance of a given module
nodestmt:
"node" nodename ":" nodetype ";"
{ 
  input::NodeStmt *node = new input::NodeStmt($2, $4);
  if (!driver.currApp())
   {
      error(yyla.location, "Node statement outside app context");
      exit(EXIT_FAILURE);
   }
  driver.currApp()->nodes.push_back(node);
};

nodename: "identifier"
{ $$ = $1; };

nodetype: 
  "identifier"                    { $$ = new input::NodeType($1); }
| "source" "<" basetypename ">"  
       { $$ = new input::NodeType(input::NodeType::isSource, $3); }
| "sink" "<" basetypename ">"  
         { $$ = new input::NodeType(input::NodeType::isSink, $3); }
;

// edge stmt: declare an edge from a channel out of one node into another
edgestmt:
"edge" nodename edgechannelspec "->" nodename ";" 
{
  input::EdgeStmt edge($2, $3, $5);
  if (!driver.currApp())
   {
      error(yyla.location, "Edge statement outside app context");
      exit(EXIT_FAILURE);
   }
  driver.currApp()->edges.push_back(edge);
};

edgechannelspec:
  "::" channelname { $$ = $2; }
| %empty           { $$ = ""; }
;
 
// per-app or per-module parameter
paramstmt:
"param" varname ":" vartype ";" 
{
   $2->type = $4;
   $2->isParam = true;
   $2->isPerNode = false;
   if (!driver.currApp())
   {
      error(yyla.location, "Param statement outside app context");
      exit(EXIT_FAILURE);
   }
   driver.currApp()->vars.push_back($2);
}

// per-node parameter
nodeparamstmt:
"nodeparam" scoped_varname ":" vartype ";" 
{
   $2->type = $4;
   $2->isParam = true;
   $2->isPerNode = true;
   if (!driver.currApp())
   {
      error(yyla.location, "NodeParam statement outside app context");
      exit(EXIT_FAILURE);
   }
   driver.currApp()->vars.push_back($2);
}

// per-node state
nodestatestmt:
"nodestate" scoped_varname ":" vartype ";" 
{
   $2->type = $4;
   $2->isParam = false;
   $2->isPerNode = true;
   if (!driver.currApp())
   {
      error(yyla.location, "NodeParam statement outside app context");
      exit(EXIT_FAILURE);
   }
   driver.currApp()->vars.push_back($2);
}

varname:
  "identifier"
    { $$ = new input::DataStmt($1, ""); }
| scoped_varname
    { $$ = $1; }
;

scoped_varname: 
varscope "::" "identifier" 
{ $$ = new input::DataStmt($3, $1); };

varscope: 
 %empty        { $$ = ""; }
| "identifier" { $$ = $1; }
;

vartype: basetypename 
{ $$ = $1; };

/////////////////////////////
// MODULE TYPES AND CHANNELS
/////////////////////////////

//qualifier inputtype "->" outputtype 
moduletype:
inputtype "->" outputtype 
{ 
  $$ = new input::ModuleTypeStmt($1, $3); 
  $$->flags |= 0;
}
| "enumerate" inputtype "->" outputtype
{ 
  $$ = new input::ModuleTypeStmt($2, $4); 
  $$->flags |= input::ModuleTypeStmt::isEnumerate;
}
;

//qualifier:
//  %empty                 { $$ = 0; }
//| "enumerate"            { $$ = input::ModuleTypeStmt::isEnumerate; }
//;
//| "aggregate"            { $$ = input::ModuleTypeStmt::isAggregate; }

inputtype:
 typename                { $$ = $1; }
;

outputtype:
 "void"                  
{ $$ = new input::OutputSpec(input::OutputSpec::isVoid);  }
| typename maxoutput
{ 
  // a single channel does not need a name
  auto v = new std::vector<input::ChannelSpec *>;
  auto c = new input::ChannelSpec("__out", $1, std::abs($2), ($2 > 0), 0);
  v->push_back(c);
  $$ = new input::OutputSpec(v);
}
| channels               
{ $$ = new input::OutputSpec($1); }
;

channels:
  channel           
   { $$ = new std::vector<input::ChannelSpec *>; $$->push_back($1); }
| channels "," channel 
   { $$ = $1; $$->push_back($3); } 
;

// an output channel of a module has a name and a type
channel:
channelname "<" typename  maxoutput ">" 
{ $$ = new input::ChannelSpec($1, $3, std::abs($4), ($4 > 0), 0); }
| "aggregate" channelname "<" typename  maxoutput ">" 
{ $$ = new input::ChannelSpec($2, $4, std::abs($5), ($5 > 0), 1); };

maxoutput:
  %empty              { $$ =   1; }
| ":" "number"        { $$ =  $2; }
| ":" "!" "number"    { $$ = -$3; } // NB: negative indicates fixed
;
 
channelname: "identifier"
{ $$ = $1; };

//////////////
// DATA TYPES
//////////////

typename:
  basetypename   {$$ = $1;}
| fromtypename   {$$ = $1;}
;

fromtypename: basetypename "from" typename 
{
   $1->from = $3;
   $$ = $1;
};

basetypename: typename_string
{ $$ = new input::DataType($1); };

typename_string:
  "identifier"
    { $$ = $1; }
| "identifier" "*"
    { $$ = $1 + " *"; }
| "typeliteral"
    { $$ = $1; }
;

%%

void
yy::mercator_parser::error(const location_type &l,
			   const std::string &m)
{			   
  driver.error(l, m);
}
