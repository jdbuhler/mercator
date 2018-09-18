//
// PARSER-DRIVER.H
// Parser driver for MERCATOR spec filess
//
// MERCATOR
// Copyright (C) 2018 Washington University in St. Louis; all rights reserved.
//

#ifndef PARSER_DRIVER_H
#define PARSER_DRIVER_H

#include <string>
#include <vector>

#include "inputspec.h"

#include "mercator-parser.tab.hh"

// Tell Flex the lexer's prototype ...
#define YY_DECL	\
  yy::mercator_parser::symbol_type yylex (mercator_driver& driver)
// ... and declare it for the parser's sake.
YY_DECL;

// parse a spec file into data structures
class mercator_driver
{
public:
  mercator_driver() 
    : trace_scanning(false),
      trace_parsing(false),
      _currApp(nullptr)
  {}
  
  virtual ~mercator_driver() 
  {}
  
  // specify whether to trace lexing
  void set_trace_scanning(bool v)
  {
    trace_scanning = v;
  }
  
  // specify whether to trace parsing
  void set_trace_parsing(bool v)
  {
    trace_parsing = v;
  }
  
  // parse a file f; returns parser's return code
  // (which is 0 iff successful)
  int parse(const std::string &f);
  
 
  std::vector<input::AppSpec *> apps;  // apps defined by parsed spec
  std::vector<std::string>      refs;  // list of file references
  
  void createApp(const std::string &name)
  {
    _currApp = new input::AppSpec(name);
    apps.push_back(_currApp);
  }
  
  input::AppSpec *currApp() { return _currApp; }
  
private:
  
  // control debug tracing
  bool trace_scanning;
  bool trace_parsing;
  
  // lexical scanner setup/teardown
  void scan_begin (const std::string &f, bool trace_scanning);
  void scan_end ();

  input::AppSpec *_currApp;
  
public:

  // these bits are exposed to the parser but are uninteresting otherwise
 
  // name of file being parsed
  std::string file;

  // error handling
  void error (const yy::location &l, const std::string &m) const;
  void error (const std::string &m) const;
};

#endif
