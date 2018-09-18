//
// PARSE.H
// Parser interface for spec files
//
// MERCATOR
// Copyright (C) 2018 Washington University in St. Louis; all rights reserved.
//

#ifndef __PARSE_H
#define __PARSE_H

#include <string>
#include <vector>

#include "inputspec.h"

std::vector<input::AppSpec *> 
parseInput(const std::string &sourceFile,
	   const std::vector<std::string> &typecheckIncludePaths,
	   std::vector<std::string> &references);

#endif
