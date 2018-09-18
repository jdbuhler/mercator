#ifndef __TYPECHECK_H
#define __TYPECHECK_H

//
// TYPECHECK.H
// Interface to LLVM type checker
//
// MERCATOR
// Copyright (C) 2018 Washington University in St. Louis; all rights reserved.
//

#include <vector>
#include <string>

/*
 * @brief Base virtual class for the ASTContext.
 */
class ASTContainer {
public:
  static ASTContainer *create();
  
  virtual
  ~ASTContainer() {}
  
  virtual
  void findTypes(const std::vector<std::string> &typeStrings,
		 const std::vector<std::string> &references,
		 const std::vector<std::string> &includePaths) = 0;
    
  virtual
  bool queryType(const std::string &typeName) const = 0;

  virtual
  unsigned long typeId(const std::string &typeName) const = 0;
  
  virtual
  bool compareTypes(const std::string &firstName, 
		    const std::string &secondName) const = 0;
};

#endif
