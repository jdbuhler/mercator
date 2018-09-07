#ifndef __BUILDAPP_H
#define __BUILDAPP_H

#include "inputspec.h"
#include "app.h"


//
// buildApp()
// Convert an app spec into an internal reprsentation of the application 
//
App *buildApp(const input::AppSpec *appSpec);

#endif
