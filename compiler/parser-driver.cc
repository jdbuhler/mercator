//
// PARSER-DRIVER.CC
// Parser driver for MERCATOR spec files
//
// MERCATOR
// Copyright (C) 2018 Washington University in St. Louis; all rights reserved.
//

#include <iostream>

#include "parser-driver.h"

using namespace std;

int
mercator_driver::parse(const string &f)
{
  file = f; // set for location tracking in parser
  
  // prepare scanner to begin lexing
  scan_begin(f, trace_scanning);
  
  // create a new parser context
  yy::mercator_parser parser(*this);
  parser.set_debug_level(trace_parsing);
  
  // do the parsing job
  int res = parser.parse();
  
  // clean up after the lexer
  scan_end();
  
  return res;
}

void
mercator_driver::error (const yy::location &l, const string &m) const
{
  cerr << l << ": " << m << endl;
}

void
mercator_driver::error (const string &m) const
{
  cerr << m << endl;
}
