//
// AGGENUMDRIVER.CU
// Aggregation and Enumeration functionality test
//
// MERCATOR
// Copyright (C) 2018 Washington University in St. Louis; all rights reserved.
//

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>

#include "Taxi.cuh"
#include "StrFunc.cuh"

using namespace std;

Position *computeCPU(const char *text,
		     const vector<Line> &lines,
		     unsigned int *len)
  
{
  vector<Position> posns;
  
  unsigned int lineNo = 1;
  for (const Line &line : lines)
    {
      const char *lineText = text + line.start;
      unsigned int len     = line.length;
      
      for (unsigned int charIdx = 0; charIdx < len; charIdx++)
	{
	  if (lineText[charIdx] == '[' && lineText[charIdx + 1] != '[')
	    {
	      char *end;
	      
	      double lon = d_strtod(&lineText[charIdx + 1], &end);
	      double lat = d_strtod(end + 1, 0);
	      
	      Position p(lineNo, lat, lon);
	      posns.push_back(p);
	    }
	}
      
      lineNo++;
    }

  Position *posnsRaw = new Position [posns.size()];
  memcpy(posnsRaw, posns.data(), posns.size() * sizeof(Position));

  *len = posns.size();
  return posnsRaw;
}
		     
bool operator==(const Position &p1, const Position &p2)
{
  const double EPS = 1.0e-6;
  
  return (p1.tag == p2.tag &&
	  fabs((p1.latitude - p2.latitude) < EPS * fabs(p1.latitude)) &&
	  fabs((p1.longitude - p2.longitude) < EPS * fabs(p1.longitude)));
}

bool posnComp(const Position &p1, const Position &p2)
{
  const double EPS = 1.0e-6;
  
  return (p1.tag < p2.tag ||
	  (p1.tag == p2.tag &&
	   (p1.latitude < p2.latitude - EPS ||
	    (!(p1.latitude > p2.latitude + EPS) &&
	     p1.longitude < p2.longitude))));
}

bool posnEq(const Position &p1, const Position &p2)
{
  const double EPS = 1.0e-6;
  return (p1.tag == p2.tag &&
	  fabs(p1.latitude - p2.latitude) < EPS &&
	  fabs(p1.longitude - p2.longitude) < EPS);
}
	     
ostream &operator<<(ostream &os, const Position &p)
{
  os << '[' << p.tag << ", " << p.latitude << ", " << p.longitude << ']';
  return os;
}


void verifyOutput(Position *posns1, unsigned int nPosns1,
		  Position *posns2, unsigned int nPosns2)
{
  if (nPosns1 != nPosns2)
    {
      cerr << "ERROR: size mismatch: CPU " 
	   << nPosns1 << " != GPU " << nPosns2 << endl;
      //return;
    }
  
  std::sort(posns1, posns1 + nPosns1, posnComp);
  std::sort(posns2, posns2 + nPosns2, posnComp);
  
  for (unsigned int j = 0; j < min(nPosns1,nPosns2); j++)
    {
#if 1
      if (!posnEq(posns1[j], posns2[j]))
	{
	  cerr << "DIFFERENCE AT ENTRY " << j << ':' << endl;
	  cerr << "CPU: " << posns1[j] << endl;
	  cerr << "GPU: " << posns2[j] << endl;
	  cerr << posns1[j].latitude - posns2[j].latitude << endl;
	  return;
      	}
      
#else
      cout << posns1[j] << ' ' << posns2[j] << endl;
#endif
    }
  
  cout << "Outputs match!" << endl;
}


int main(int argc, char** argv)
{
  const unsigned int NVALUES = 100000000;
  
  if (argc < 2)
    {
      cerr << "Syntax: TaxiApp <input file>" << endl;
      exit(1);
    }
  
  ifstream file(argv[1]);
  if (!file)
    {
      cerr << "Error: could not open file " << argv[1] << endl;
      exit(1);
    }
  
  // skip the header
  file.ignore(numeric_limits<streamsize>::max(), '\n');
  size_t fileStart = file.tellg();
  
  //
  // read the rest of the file into a buffer
  //
  
  file.seekg(0, file.end);
  size_t fileEnd = file.tellg();
  
  size_t fileSize = fileEnd - fileStart;
  
  char* h_string = new char [fileSize];
  
  file.seekg(fileStart, file.beg);
  file.read(h_string, fileSize);
  
  //
  // compute the lengths of each line in the file
  //
  
  vector<Line> lines;
  
  size_t lineStart = 0;
  unsigned int lineNum = 1;
  for (size_t j = 0; j < fileSize; j++)
    {
      if (h_string[j] == '\n')
	{
	  lines.push_back(Line(lineNum++, lineStart, j - lineStart + 1));
	  lineStart = j + 1;
	}
    }

  // last line might not terminate in EOL
  if (h_string[fileSize - 1] != '\n')
    lines.push_back(Line(lineNum, lineStart, fileSize - lineStart));
  
  char* d_string;
  cudaMalloc(&d_string, fileSize);
  cudaMemcpy(d_string, h_string, fileSize, cudaMemcpyHostToDevice);
  
  Mercator::Buffer<Line> inputBuffer(lines.size());
  Mercator::Buffer<Position> outputBuffer(NVALUES);
  
  inputBuffer.set(lines.data(), lines.size());
  
  Taxi efapp;
  
  efapp.getParams()->text = d_string;
  efapp.setSource(inputBuffer);
  efapp.snk.setSink(outputBuffer);
  
  cout << "RUNNING APP..." << endl;
  efapp.run(inputBuffer.size());
  cout << "APP FINISHED!" << endl;
  
  // get data out of the output buffer
  unsigned int gpuOutputSize = outputBuffer.size();
  Position *gpuOutput = new Position [gpuOutputSize];
  outputBuffer.get(gpuOutput, gpuOutputSize);
  
  unsigned int cpuOutputSize;
  Position *cpuOutput = computeCPU(h_string, lines, &cpuOutputSize);
  
  verifyOutput(cpuOutput, cpuOutputSize,
	       gpuOutput, gpuOutputSize);
  
  return 0;
}
