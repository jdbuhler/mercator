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
