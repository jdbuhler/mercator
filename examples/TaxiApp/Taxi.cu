#include "Taxi_dev.cuh"
#include "StrFunc.cuh"

__device__
void Taxi_dev::
BracketFind::begin()
{}

__device__
void Taxi_dev::
BracketFind::run(const unsigned int& charIdx, unsigned int nInputs)
{
  const Line* line = getParent();
  const char *text = getAppParams()->text + line->start;

  bool foundBracket = false;
  if (threadIdx.x < nInputs)
    {
      if (text[charIdx] == '[')
	foundBracket = true;
    }
  
  push(charIdx, foundBracket); 
}

__device__
void Taxi_dev::
BracketFind::end()
{}

__device__
void Taxi_dev::
CoordinateSwap::begin()
{}

__device__
void Taxi_dev::
CoordinateSwap::run(const unsigned int& charIdx, unsigned int nInputs)
{
  const Line *line = getParent();
  const char *text = getAppParams()->text + line->start;
      
  Position p;
    
  bool foundPosition = false;
  if (threadIdx.x < nInputs)
    {
      if (text[charIdx + 1] != '[') 
	{
	  foundPosition = true;
	  
	  char* end;
	  
	  double lon = d_strtod(&text[charIdx + 1], &end);
	  double lat = d_strtod(end + 1, nullptr);
	  
	  p = Position(line->tag, lat, lon); // Swap coords for output
	}
    }
  
  push(p, foundPosition);
}

__device__
void Taxi_dev::
CoordinateSwap::end()
{}

__device__
unsigned int Taxi_dev::
__enumerateFor_BracketFind::findCount(const Line &parent) const
{
  return parent.length;
}
