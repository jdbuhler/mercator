#ifndef CODEGEN_UTILS_H
#define CODEGEN_UTILS_H

//
// CODEGEN_UTILS.H
// Utilities to assist with code generation
//
// MERCATOR
// Copyright (C) 2018 Washington University in St. Louis; all rights reserved.
//

#include <string>
#include <cassert>

/**
 * @brief Generate \#include statement for system file.
 *
 * Format: \#include \< fileName \>
 *
 * @param fileName Name of file to be included
 *
 */
inline
std::string genSystemInclude(const std::string &fileName)
{
  return "#include <" + fileName + ">";
}

/**
 * @brief Generate \#include statement for user file.
 *
 * Format: \#include "fileName"
 *
 * @param fileName Name of file to be included
 *
 */
inline
std::string genUserInclude(const std::string &fileName)
{
  return "#include \"" + fileName + "\"";
}

/**
 * @brief Turn a name into an include guard string
 */
inline
std::string genIncludeGuardName(const std::string &name)
{
  std::string headerBase = name;
  size_t dot_pos = headerBase.find_last_of('.');
  if(dot_pos != std::string::npos)
    headerBase = headerBase.replace(dot_pos, 1, "_"); 
  return "__" + headerBase + "_CUH__";
}


/**
 * @brief Generate string giving nondefining fcn declaration.
 *
 * @param retType String representing function return type
 * @param fcnName String representing function name
 * @param param Full string representing fcn params 
 * @return string representing fcn header
 */
inline
std::string genFcnHeader(const std::string &retType,
			 const std::string &fcnName,
			 const std::string &param)
{
  return retType + (retType == "" ? "" : " ") + fcnName + "(" + param + ")";
}

#endif
