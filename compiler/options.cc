//
// OPTIONS.CC
// Command-line option parser
//
// MERCATOR
// Copyright (C) 2018 Washington University in St. Louis; all rights reserved.
//

#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <cassert>
#include <unistd.h>

#include "options.h"

#include "version.h"

using namespace std;

const char OptionList[] = ":a:DhH:I:K:o:q:S:t:v";

CommandOptions options;

static void printUsage()
{
  cerr << "USAGE: mercator [options] <sourcefile> [ <sourcefile> ...]\n" 
       << "\n"
       << "OPTIONS:\n"
       << " -a <app>\n"
       << "   build only the specified app, ignoring others in spec file\n"
       << "\n"
       << " -I <path>\n"
       << "   add <path> to search path for parsing files included\n"
       << "   in a spec file by a reference statement\n"
       << "   (by default, check system includes, CUDA includes, and\n"
       << "    the directory where the spec file is located)\n"
       << "\n"
       << " -K <path>\n"
       << "   generate a skeleton user stub files for apps at <path>\n"
       << "    (default: generate runtime support code for apps)\n"
       << "\n"
       << " -o <path>\n"
       << "   write the output files to <path>\n"
       << "   (defaults to current directory)\n"
       << "\n"
       << "-q <#>\n"
       << "   queue size scale factor for generated application\n"
       << "   (default " << options.queueScaler << ")\n"
       << "\n"
       << " -t <#>\n"
       << "   number of threads per block for the generated application\n"
       << "    (default " << options.threadsPerBlock << ")\n"
       << "\n"
       << " -H <#>\n"
       << "   size of the device heap in MEGAbytes\n"
       << "    (default " << options.deviceHeapSize/(1024*1024) << " MB)\n"
       << "\n"
       << " -S <#>\n"
       << "   size of the device stack in KILObytes\n"
       << "    (default " << options.deviceStackSize/(1024) << " KB)\n"
       << "\n"
       << " -D\n"
       << "   do not generate code, but print a list of the output files produced\n"
       << "   from each source file in the form of a make dependency\n"
       << "\n"
       << "-v\n"
       << "   print version information\n"
       << " -h\n"
       << "   print this help message\n";
    
  exit(1);
}


bool parseCommandLine(int argc, char **argv)
{
  if (argc == 1) // no arguments
    {
      cerr << "For help, say \"mercator -h\"" << endl;
      exit(1);
    }
  
  int c;
  try 
    {
      while ((c = getopt(argc, argv, OptionList)) != -1)
	{
	  switch (c)
	    {
	    case 'a':
	      options.appToBuild = optarg;
	      break;
	      
	    case 'I':
	      options.typecheckIncludePaths.push_back(optarg);
	      break;
	      
	    case 'K':
	      options.generateSkeletons = true;
	      options.skeletonFileName = optarg;
	      break;
	      
	    case 'o':
	      options.outputPath = optarg;
	      break;
	      
	    case 'q':
	      options.queueScaler = stoi(optarg);
	      if (options.queueScaler < 1)
		{
		  cerr << "ERROR: queue scaler must be >= 1"
		       << endl;
		  return false;
		}
	      break;
	      
	    case 't':
	      options.threadsPerBlock = stoi(optarg);
	      if (options.threadsPerBlock < 1)
		{
		  cerr << "ERROR: threads per block must be >= 1" 
		       << endl;
		  return false;
		}
	      break;
	      
	    case 'S':
	      options.deviceStackSize = stoi(optarg) * 1024;
	      if (options.deviceStackSize < 1024)
		{
		  cerr << "ERROR: device stack must be >= 1 KB"
		       << endl;
		  return false;
		}
	      break;
	      
	    case 'H':
	      options.deviceHeapSize = stoi(optarg) * 1024 * 1024;
	      if (options.deviceHeapSize < 1024 * 1024)
		{
		  cerr << "ERROR: device heap must be >= 1 MB"
		       << endl;
		  return false;
		}
	      break;
	      
	    case 'D':
	      options.emitDeps = true;
	      break;
	      
	    case 'v':
	      cout << "MERCATOR " 
		   << MERCATOR_MAJOR << '.' 
		   << MERCATOR_MINOR << '.'
		   << MERCATOR_PATCHLEVEL << "\n";
	      exit(1);
	    break;
	    
	    case 'h':
	      printUsage(); // does not return
	      break;
	      
	    case '?':
	      cerr << "  (use -h to get help)" << endl;
	      return false;
	      
	    case ':':
	      cerr << "ERROR: option '" << optopt << "' requires argument" << endl;
	      cerr << "  (use -h to get help)" << endl;
	      return false;
	      
	    default:
	      // we should never get here!
	      assert(false);
	    }
	}
    }
  catch (const invalid_argument &ia) 
    {
      cerr << "ERROR: invalid numerical argument '" << optarg << "' to option -" << (char) c << endl;
      cerr << "  (use -h to get help)" << endl;
      return false;
    }
  
  for (int j = optind; j < argc; j++)
    options.sourceFiles.push_back(argv[j]);
  
  //
  // make sure the output path ends with a path separator
  //
  
  if (options.outputPath != "" && 
      options.outputPath.back() != '/')
    options.outputPath += "/";
  
  return true;
}
    
