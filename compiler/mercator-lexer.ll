%{ /* -*- C++ -*- */

// MERCATOR-LEXER.LL
// Lexer for MERCATOR specification parser
//
// MERCATOR
// Copyright (C) 2018 Washington University in St. Louis; all rights reserved.
//

#include <cerrno>
#include <climits>
#include <cstdlib>
#include <string>

#include "parser-driver.h"
#include "mercator-parser.tab.hh"

// Work around an incompatibility in flex (at least versions
// 2.5.31 through 2.5.33): it generates code that does
// not conform to C89.  See Debian bug 333231
// <http://bugs.debian.org/cgi-bin/bugreport.cgi?bug=333231>.
#undef yywrap
#define yywrap() 1

// The location of the current token.
static yy::location loc;
%}

%option noyywrap nounput batch debug noinput

cxxcomment  \/\/[^\n]*\n

int      [0-9]+
blank    [ \t]
type     \{[^\}\n]+\}
string   \"[^\"]*\"

id       [a-zA-Z_][a-zA-Z_0-9]*


%{
  // Code run each time a pattern is matched.
  # define YY_USER_ACTION  loc.columns (yyleng);
%}

%%

%{
  // Code run each time yylex is called.
  // Next few lines adjust the location when
  // consuming strings of symbols with no effect.
  loc.step ();
%}

{blank}+     loc.step();
[\n]+        loc.lines(yyleng); loc.step();
{cxxcomment} loc.columns(yyleng); loc.lines(1); loc.step();

%{
  // special symbols
%}

":"      return yy::mercator_parser::make_COLON(loc);
","      return yy::mercator_parser::make_COMMA(loc);
"->"     return yy::mercator_parser::make_GOESTO(loc);
">"      return yy::mercator_parser::make_GT(loc);
"<"      return yy::mercator_parser::make_LT(loc);
"!"      return yy::mercator_parser::make_BANG(loc);
"::"     return yy::mercator_parser::make_SCOPE(loc);
";"      return yy::mercator_parser::make_SEMICOLON(loc);
"*"      return yy::mercator_parser::make_STAR(loc);

%{
  // keywords are case-insensitive
%}

(?i:"aggregate")   return yy::mercator_parser::make_AGGREGATE(loc);
(?i:"allthreads")  return yy::mercator_parser::make_ALLTHREADS(loc);
(?i:"application") return yy::mercator_parser::make_APPLICATION(loc);
(?i:"buffer")      return yy::mercator_parser::make_BUFFER(loc);
(?i:"edge")        return yy::mercator_parser::make_EDGE(loc);
(?i:"enumerate")   return yy::mercator_parser::make_ENUMERATE(loc); 
(?i:"from")        return yy::mercator_parser::make_FROM(loc); 
(?i:"function")    return yy::mercator_parser::make_FUNCTION(loc); 
(?i:"ilimit")      return yy::mercator_parser::make_ILIMIT(loc); 
(?i:"mapping")     return yy::mercator_parser::make_MAPPING(loc);
(?i:"module")      return yy::mercator_parser::make_MODULE(loc);
(?i:"node")        return yy::mercator_parser::make_NODE(loc);
(?i:"nodeparam")   return yy::mercator_parser::make_NODEPARAM(loc);
(?i:"nodestate")   return yy::mercator_parser::make_NODESTATE(loc);
(?i:"param")       return yy::mercator_parser::make_PARAM(loc);
(?i:"reference")   return yy::mercator_parser::make_REFERENCE(loc);
(?i:"source")      return yy::mercator_parser::make_SOURCE(loc);
(?i:"sink")        return yy::mercator_parser::make_SINK(loc);
(?i:"threadwidth") return yy::mercator_parser::make_THREADWIDTH(loc);
(?i:"void")        return yy::mercator_parser::make_VOID(loc);

{int}      {
  errno = 0;
  unsigned long n = strtoul(yytext, nullptr, 10);
  if (! (n <= INT_MAX && errno != ERANGE))
    {
      driver.error (loc, "integer is out of range");
      exit(EXIT_FAILURE);
    }
  return yy::mercator_parser::make_NUMBER(n, loc);
}

{id}       return yy::mercator_parser::make_ID(yytext, loc);

{type}     {
  // trim first and last character {}
  std::string s(yytext);
  std::string t = s.substr(1, s.size() - 2);
  return yy::mercator_parser::make_TYPE(t.c_str(), loc);
}

{string}   {
  // trim first and last character ""
  std::string s(yytext);
  std::string t = s.substr(1, s.size() - 2);
  return yy::mercator_parser::make_STRING(t.c_str(), loc);
}

.          {
   driver.error (loc, "invalid character");
   exit(EXIT_FAILURE);
}

<<EOF>>    return yy::mercator_parser::make_END(loc);

%%

void
  mercator_driver::scan_begin(const std::string &f,
                              bool trace_scanning)
{
  yy_flex_debug = trace_scanning;
  if (f.empty() || f == "-")
    yyin = stdin;
  else if (!(yyin = fopen (f.c_str(), "r")))
    {
      error("cannot open " + f + ": " + strerror(errno));
      exit(EXIT_FAILURE);
    }
}

void
mercator_driver::scan_end ()
{
  fclose(yyin);
}
