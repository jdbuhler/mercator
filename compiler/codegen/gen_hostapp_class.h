#ifndef GEN_HOSTAPP_CLASS_H
#define GEN_HOSTAPP_CLASS_H

//
// @file gen_hostapp_class.h
// @brief code generator for MERCATOR app class on host
//
// MERCATOR
// Copyright (C) 2018 Washington University in St. Louis; all rights reserved.
//

#include <vector>
#include <string>

class App;  

//
// @brief generate the entire host-side header for a MERCATOR app
//
// @param hostClassFileName name of header file
// @param app Application for which to generate output
// @param userIncludes list of include files supplied by user's
//    reference directives, whicht must be included in the header file
//
void genHostAppHeader(const std::string &fileName,
		      const App *app,
		      const std::vector<std::string> &userIncludes);


void genHostAppConstructor(const std::string &fileName,
			   const App *app);

#endif
