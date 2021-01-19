#include "Taxi_dev.cuh"
#include "StrFunc.cuh"

__MDECL__
void Taxi_dev::
BracketFind<InputView>::begin()
{}

__MDECL__
void Taxi_dev::
BracketFind<InputView>::run(unsigned int const & charIdx, unsigned int nInputs)
{
  const Line* line = getParent();
  const char *text = getAppParams()->text + line->start;
  bool foundBracket = false;
  
  if (threadIdx.x < nInputs)
    {
      foundBracket = (threadIdx.x < nInputs && 
		      text[charIdx] == '[' &&
		      text[charIdx + 1] != '[');
    }
  
  push(text + charIdx, foundBracket);
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
CoordinateSwap<InputView>::run(char* const &text,
			       unsigned int nInputs)
{
  Position p;
    
  if (threadIdx.x < nInputs)
    {
      const Line* line = getParent();
      char* end;
      
      double lon = d_strtod(text + 1, &end);
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
__enumerateFor_BracketFind<InputView>::findCount(Line const &parent) const
{
  return parent.length;
}
