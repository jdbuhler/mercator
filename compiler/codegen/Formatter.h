//
// FORMATTER.H
// Collect lines of a source code file, maintaining running indentation
// throughout.
//
// MERCATOR
// Copyright (C) 2018 Washington University in St. Louis; all rights reserved.
//

#ifndef __FORMATTER_H
#define __FORMATTER_H

#include <string>
#include <vector>

class Formatter {
public:  

  Formatter(int idefaultIndent = 3)
    : totalSpace(0),
      defaultIndent(idefaultIndent)
  {}
  
  //
  // add()
  // Add a line to the end of the formatted uoutput
  // if backup is true, undo the last indent for
  // this line only.
  //
  void add(const std::string &s, bool backup = false);
  
  //
  // extend()
  // append to the end of the previous line
  //
  void extend(const std::string &s);
  
  //
  // emit()
  // print the formatted outut to the specified file
  //
  void emit(const std::string &fileName) const;

  //
  // indent()
  // Add the default amount to the current indentation.
  //
  void indent() { indent(defaultIndent); }
  
  //
  // indent()
  // Add the specified amount to the current indentation.
  //
  void indent(int amount);
  
  //
  // indentAfter()
  // Indent the specified number of characters (default 1)
  // after the first occurrence of char c on the previous line.
  // If c does not occur, indent by delta.
  //
  void indentAfter(char c, int delta = 1);
  
  //
  // unindent()
  // Undo the most recent level of indentation.
  void unindent();
  
private:

  std::vector<std::string> lines;
  
  std::vector<int> levels;
  int totalSpace;
  int defaultIndent;
};

#endif
