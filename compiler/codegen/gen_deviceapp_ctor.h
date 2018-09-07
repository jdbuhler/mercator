#ifndef GEN_DEVICEAPP_CTOR_H
#define GEN_DEVICEAPP_CTOR_H

//
// @file gen_deviceapp_ctor.h
// @brief code gneerator for device-side app constructor
//

#include <string>

#include "Formatter.h"

class App;
  
//
// @brief Code-gen device-side app constructor
//
// @param app to be codegen'd
// @param f Formatter to receive generated code
//
void genDeviceAppConstructor(const App *app,
			     Formatter &f);

#endif
