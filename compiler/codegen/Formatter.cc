//
// FORMATTER.CC
// Collect lines of a source code file, maintaining running indentation
// throughout.
//
// MERCATOR
// Copyright (C) 2018 Washington University in St. Louis; all rights reserved.
//

#include <iostream>
#include <fstream>
#include <cassert>

#include "Formatter.h"

using namespace std;

//
// add()
// Add a line to the end of the formatted uoutput
// if backup is true, undo the last indent for
// this line only.
//
void Formatter::add(const string &s, bool backup)
{
  int space = totalSpace;
  if (backup)
    {
      assert(levels.size() > 0);
      space -= levels.back();
    }
  
  string line = string(space, ' ') + s;
  
  lines.push_back(line);
}


//
// extend()
// append to the end of the previous line
//
void Formatter::extend(const string &s)
{
  assert(lines.size() > 0);
  
  string &lastLine = lines.back();
  lastLine.append(s);
}


//
// emit()
// print the formatted outut on the specified ostream
//
void Formatter::emit(const string &fileName) const
{
  ofstream os(fileName);
  
  for (const string &line : lines)
    os << line << endl;
}

//
// indent()
// Add the specified amount to the current indentation.
//
void Formatter::indent(int amount)
{
  levels.push_back(amount);
  totalSpace += amount;
}


//
// indentAfter()
// Indent so that the next line begins delta characters after
// the first occurrence of char c on the previous line.
// If c does not occur, do nothing.
//
void Formatter::indentAfter(char c, int delta)
{
  assert(lines.size() > 0);
  
  const string &prevLine = lines.back();
  size_t pos = prevLine.find_first_of(c);
  if (pos != string::npos)
    {
      int diff = (pos + delta) - totalSpace;
      indent(diff);
    }
}


//
// unindent()
// Undo the most recent level of indentation.
//
void Formatter::unindent()
{
  assert(levels.size() > 0);
  
  int amount = levels.back();
  totalSpace -= amount;
  
  levels.pop_back();
}
