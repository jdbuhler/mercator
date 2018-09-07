#ifndef GEN_HOSTAPP_CLASS_H
#define GEN_HOstAPP_CLASS_H

//
// @file gen_hostapp_class.h
// @brief code generator for MERCATOR app class on host
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
