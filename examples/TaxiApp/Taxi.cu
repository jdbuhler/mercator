#include "Taxi_dev.cuh"
#include "StrFunc.cuh"

__MDECL__
void Taxi_dev::
BracketFind<InputView>::begin()
{}

__MDECL__
void Taxi_dev::
BracketFind<InputView>::run(const unsigned int& charIdx, unsigned int nInputs)
{
  bool foundBracket = false;
  
  if (threadIdx.x < nInputs)
    {
      const Line* line = getParent();
      const char *text = getAppParams()->text + line->start;
      
      foundBracket = (threadIdx.x < nInputs && 
		      text[charIdx] == '[' &&
		      text[charIdx + 1] != '[');
    }
  
  push(charIdx, foundBracket);
}

__MDECL__
void Taxi_dev::
BracketFind<InputView>::end()
{}

__MDECL__
void Taxi_dev::
CoordinateSwap<InputView>::begin()
{}

__MDECL__
void Taxi_dev::
CoordinateSwap<InputView>::run(const unsigned int& charIdx, unsigned int nInputs)
{
  Position p;
    
  if (threadIdx.x < nInputs)
    {
      const Line *line = getParent();
      const char *text = getAppParams()->text + line->start;
      
      char* end;
      
      double lon = d_strtod(&text[charIdx + 1], &end);
      double lat = d_strtod(end + 1, &end);
      
      p = Position(line->tag, lat, lon); // Swap coords for output
    }
  
  push(p, threadIdx.x < nInputs);
}

__MDECL__
void Taxi_dev::
CoordinateSwap<InputView>::end()
{}

__MDECL__
unsigned int Taxi_dev::
__enumerateFor_BracketFind<InputView>::findCount(const Line &parent) const
{
  return parent.length;
}
