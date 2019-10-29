#include "Taxi_dev.cuh"
#include "StrFunc.cuh"

__device__
void Taxi_dev::
BracketFind::begin(InstTagT nodeIdx)
{

}

__device__
void Taxi_dev::
BracketFind::run(const unsigned int& inputItem, InstTagT nodeIdx)
{
  Line* l = getParent(nodeIdx);
  if(l[0].startPointer[inputItem] == '[')
	push(inputItem, nodeIdx); 
}

__device__
void Taxi_dev::
BracketFind::end(InstTagT nodeIdx)
{

}

__device__
void Taxi_dev::
CoordinateSwap::begin(InstTagT nodeIdx)
{

}

__device__
void Taxi_dev::
CoordinateSwap::run(const unsigned int& inputItem, InstTagT nodeIdx)
{
  Line* l = getParent(nodeIdx);
  double lon, lat;
  char* end;
  if(l[0].startPointer[inputItem + 1] != '[') {
	lon = d_strtod(l[0].startPointer + inputItem + 1, &end);
	lat = d_strtod(end + 1, NULL);
	Position p = Position(l[0].tag, lat, lon);	//Swap coordinated for output
	//printf("[%d, %d]\t\t%lf, %lf\n", blockIdx.x, l[0].tag, lat, lon);
	push(p, nodeIdx);
  }
}

__device__
void Taxi_dev::
CoordinateSwap::end(InstTagT nodeIdx)
{

}

__device__
unsigned int Taxi_dev::
__enumerateFor_BracketFind::findCount(InstTagT nodeIdx)
{
	Line* l = getParent(nodeIdx);
	//printf("[%d] %d\n", blockIdx.x, l[0].length);
	return l[0].length;
	//return 0;	//Replace this return with the number of elements found for this enumeration.
}

