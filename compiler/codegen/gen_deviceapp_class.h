#ifndef GEN_DEVICEAPP_CLASS_H
#define GEN_DEVICEAPP_CLASS_H

//
// @file gen_deviceapp_class.h
// @brief code generator for MERCATOR app class on device
//

#include <string>

class App;  

//
// @brief generate the entire device-side header for a MERCATOR app
// @param fileName name of file to generate
// @param app application to be codegen'd
//
void genDeviceAppHeader(const std::string &fileName,
			const App *app);


//
// @brief generate the device-side skeleton file with run() functions
// @param fileName name of file to generate
// @param app application to be codegen'd
// 
void genDeviceAppSkeleton(const std::string &fileName,
			  const App *app);

#endif
