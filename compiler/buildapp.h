//
// BUILDAPP.H
// Build an internal application representation from a parsed spec
//
// MERCATOR
// Copyright (C) 2018 Washington University in St. Louis; all rights reserved.
//

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
