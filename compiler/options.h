//
// OPTIONS.H
// Command-line option parser
//

#ifndef __OPTIONS_H
#define __OPTIONS_H

#include <string>
#include <vector>

struct CommandOptions {
  
  bool emitDeps;
  bool generateSkeletons;
  
  unsigned int threadsPerBlock;
  unsigned int deviceStackSize;
  unsigned int deviceHeapSize;
  
  std::string outputPath;
  std::string appToBuild;
  
  std::vector<std::string> typecheckIncludePaths;
  
  std::vector<std::string> sourceFiles;
  
  // set default options
  CommandOptions()
  {
    emitDeps = false;
    generateSkeletons = false;
    
    threadsPerBlock = 128;
    deviceStackSize = (1024 * 8);          // 8 KB
    deviceHeapSize  = (1024 * 1024 * 100); // 100 MB
    
    outputPath = "";
    appToBuild = "";
  }
};

extern CommandOptions options;

bool parseCommandLine(int argc, char **argv);


#endif
