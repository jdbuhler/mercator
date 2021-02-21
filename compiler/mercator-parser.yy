//
// MERCATOR-PARSER.YY
// Parser for MERCATOR spec files
//
// MERCATOR
// Copyright (C) 2018 Washington University in St. Louis; all rights reserved.

%skeleton "lalr1.cc" /* -*- C++ -*- */
%require "3.3"

%defines

%define api.parser.class {mercator_parser}
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
  COLON   ":"
  COMMA   ","
  GOESTO  "->"
  GT      ">"
  LT      "<"
  SCOPE   "::"
  SEMICOLON ";"
  STAR    "*"
  AT      "@"
;

// keywords
%token
  AGGREGATE "aggregate"
  APPLICATION "application"
  BUFFER "buffer"
  EDGE    "edge"
  ENUMERATE "enumerate"
  FROM    "from"
  FUNCTION "function"
  ILIMIT  "ilimit"
  MODULE  "module"
  NODE    "node"
  NODEPARAM "nodeparam"
  NODESTATE "nodestate"
  PARAM    "param"
  REFERENCE "reference"
  SIMPLEMODULE "simplemodule"
  SINK    "sink"
  SOURCE  "source"
  THREADWIDTH "threadwidth"
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
%type <input::SourceStmt::SourceKind> sourcetype
%type <unsigned int> maxoutput inputsperthread
%type <input::DataType *> typename basetypename fromtypename inputtype vartype
%type <input::ChannelSpec *> channel simplechannel
%type <std::vector<input::ChannelSpec *> *> channels implicitchannel 
%type <input::OutputSpec *> outputtype 
%type <input::ModuleTypeStmt *> moduletype simplemoduletype
%type <input::DataStmt *> varname scoped_varname
%type <bool> modulekind

// expected shift-reduce conflicts:
//   node <nodename> : <modulename> vs node <nodename> : <moduletype> 
//%expect 1

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
| ilimitstmt
| threadwidthstmt
| nodestmt
| sourcestmt
| edgestmt
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
modulekind modulename ":" moduletype ";" 
{ 
  $4->name = $2;

  if ($1)
   $4->setSimple();
    
  if (!driver.currApp())
   {
      error(yyla.location, "ModuleType statement outside app context");
      exit(EXIT_FAILURE);
   }
  driver.currApp()->modules.push_back($4);
};

modulekind:
  "module"       { $$ = false; }
| "simplemodule" { $$ = true; };

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

threadwidthstmt:
"threadwidth" "number" ";"
{
   if (!driver.currApp())
    {
       error(yyla.location, "ThreadWdith statment outside app context");
       exit(EXIT_FAILURE);
    }
   driver.currApp()->threadWidth = $2;
};


// node stmt: declare a node as instance of a given module
nodestmt:
"node" nodename ":" nodetype ";"
{ 
  input::NodeStmt *node;
  
  if ($4->kind == input::NodeType::isGensym)
  {
    // module type was implicitly defined; give it a name and this
    // name to record the type of the node
    
     std::string gensymType = $2 + "_type";
     $4->mt->name = gensymType;
     driver.currApp()->modules.push_back($4->mt);
     
     node = new input::NodeStmt($2, new input::NodeType(gensymType));
     delete $4;
  }
  else
  {
     node = new input::NodeStmt($2, $4);
  }

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
| "sink" "<" basetypename ">"  
         { $$ = new input::NodeType(input::NodeType::isSink, $3); }
| moduletype                      { $$ = new input::NodeType($1); };


sourcestmt:
"source" nodename sourcetype ";"
{
   input::SourceStmt src($2, $3);
   
   if (!driver.currApp())
   {
      error(yyla.location, "Edge statement outside app context");
      exit(EXIT_FAILURE);
   }
   driver.currApp()->sources.push_back(src);
};


sourcetype:
 %empty       { $$ = input::SourceStmt::SourceIdx; }
| "function"  { $$ = input::SourceStmt::SourceFunction; }
| "buffer"    { $$ = input::SourceStmt::SourceBuffer; };


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

moduletype:
simplemoduletype { $$ = $1; }
| "enumerate" simplemoduletype
{
  $2->setEnumerate();
  $$ = $2;
};

simplemoduletype:
inputtype inputsperthread "->" outputtype 
{
  $$ = new input::ModuleTypeStmt($1, $2, $4);
};

inputtype:
 typename { $$ = $1; }
;

inputsperthread:
  %empty       { $$ = 1; }
| "@" "number" { $$ = $2; }
;

outputtype:
 "void"                  
{ $$ = new input::OutputSpec(input::OutputSpec::isVoid);  }
| implicitchannel
{ $$ = new input::OutputSpec($1); }
| "aggregate" implicitchannel
{ (*$2)[0]->isAggregate = true; $$ = new input::OutputSpec($2); }
| channels               
{ $$ = new input::OutputSpec($1); }
;

implicitchannel : typename maxoutput
{ 
  // a single channel does not need a name
  auto v = new std::vector<input::ChannelSpec *>;
  auto c = new input::ChannelSpec("__out", $1, $2, 0);
  v->push_back(c);
  $$ = v;
}

channels:
  channel           
   { $$ = new std::vector<input::ChannelSpec *>; $$->push_back($1); }
| channels "," channel 
   { $$ = $1; $$->push_back($3); } 
;

// an output channel of a module has a name and a type
channel:
simplechannel { $$ = $1; }
|
"aggregate" simplechannel
{
  $2->isAggregate = true;
  $$ = $2;
};

simplechannel:
channelname "<" typename  maxoutput ">" 
{ $$ = new input::ChannelSpec($1, $3, $4, false); }

maxoutput:
  %empty              { $$ =   1; }
| ":" "number"        { $$ =  $2; }
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
