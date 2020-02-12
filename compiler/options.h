//
// OPTIONS.H
// Command-line option parser
//
// MERCATOR
// Copyright (C) 2018 Washington University in St. Louis; all rights reserved.

#ifndef __OPTIONS_H
#define __OPTIONS_H

#include <string>
#include <vector>

struct CommandOptions {
  
  bool emitDeps;            // run in emit dependency mode
  bool generateSkeletons;   // run in generate skeleton mode
  
  unsigned int threadsPerBlock; // # threads per GPU block
  unsigned int deviceStackSize; // size of device stack
  unsigned int deviceHeapSize;  // size of device heap
  
  // multiplier used for minimum viable queue sizes
  // to determine sizes actually built
  unsigned int queueScaler;
  
  std::string outputPath;  // where to write outputs
  std::string appToBuild;  // name of app if multiple in file
  
  // list of paths to use to resolve filenames in 
  // reference directives
  std::vector<std::string> typecheckIncludePaths;

  // list of spec files to compile
  std::vector<std::string> sourceFiles;
  
  // set default options
  CommandOptions()
  {
    emitDeps = false;
    generateSkeletons = false;
    
    threadsPerBlock = 128;
    deviceStackSize = (1024 * 8);          // 8 KB

    //deviceHeapSize  = (1024 * 1024 * 200); // 100 MB
    deviceHeapSize  = (1024 * 1024 * 500); // 500MB 
    
    queueScaler = 4;
    
    outputPath = "";
    appToBuild = "";
  }
};

extern CommandOptions options;

bool parseCommandLine(int argc, char **argv);


#endif
