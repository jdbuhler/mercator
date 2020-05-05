#include "Taxi_dev.cuh"
#include "StrFunc.cuh"

__device__
void Taxi_dev::
BracketFind::begin()
{

}

__device__
void Taxi_dev::
BracketFind::run(const unsigned int& inputItem)
{
  Line* l = getParent();
  //printf("%s\n", l[0].startPointer[inputItem]);
  //if(l[0].tag > 0)
	//printf("Tag: %d\t\tinputItem: %d\t\tchar: %c\n", l[0].tag, inputItem, l[0].startPointer[inputItem]);
  if(l[0].startPointer[inputItem] == '[')
	push(inputItem); 
}

__device__
void Taxi_dev::
BracketFind::end()
{

}

__device__
void Taxi_dev::
CoordinateSwap::begin()
{

}

__device__
void Taxi_dev::
CoordinateSwap::run(const unsigned int& inputItem)
{
  Line* l = getParent();
  double lon, lat;
  char* end;
  if(l[0].startPointer[inputItem + 1] != '[') {
	lon = d_strtod(l[0].startPointer + inputItem + 1, &end);
	lat = d_strtod(end + 1, NULL);
	Position p = Position(l[0].tag, lat, lon);	//Swap coordinated for output
	//printf("[%d, %d]\t\t%lf, %lf\n", blockIdx.x, l[0].tag, lat, lon);
	push(p);
  }
}

__device__
void Taxi_dev::
CoordinateSwap::end()
{

}

__device__
unsigned int Taxi_dev::
__enumerateFor_BracketFind::findCount()
{
	Line* l = getParent();
	//if(l[0].tag > 0)
	//	printf("[%d] %d, %d\n", blockIdx.x, l[0].length, l[0].tag);
	return l[0].length;
	//return 0;	//Replace this return with the number of elements found for this enumeration.
}

