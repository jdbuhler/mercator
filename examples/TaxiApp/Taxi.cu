#include "Taxi_dev.cuh"
#include "StrFunc.cuh"

__device__
void Taxi_dev::
BracketFind::begin()
{
#if 0
  const Line *line = getParent();
  
  if (threadIdx.x == 0)
    printf("%d %d\n", blockIdx.x, line->tag);
#endif
}

__device__
void Taxi_dev::
BracketFind::run(const unsigned int& charIdx)
{
  const Line* line = getParent();
  const char *text = getAppParams()->text + line->start;
  
  if (text[charIdx] == '[')
    push(charIdx); 
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
CoordinateSwap::run(const unsigned int& charIdx)
{
  const Line *line = getParent();
  const char *text = getAppParams()->text + line->start;
  
  if (text[charIdx + 1] != '[') 
    {
      char* end;
      
      double lon = d_strtod(&text[charIdx + 1], &end);
      double lat = d_strtod(end + 1, NULL);
      
      push(Position(line->tag, lat, lon)); // Swap coords for output
    }
}

__device__
void Taxi_dev::
CoordinateSwap::end()
{}

__device__
unsigned int Taxi_dev::
__enumerateFor_BracketFind::findCount(const Line &parent)
{
  return parent.length;
}
