#include "OtherTaxi_dev.cuh"
#include "StrFunc.cuh"

__device__
void OtherTaxi_dev::
BracketFind::begin(InstTagT nodeIdx)
{

}

__device__
void OtherTaxi_dev::
BracketFind::run(const unsigned int& inputItem, InstTagT nodeIdx)
{
  Line* l = getParent(nodeIdx);
  if(l[0].startPointer[inputItem] == '[')
	push(TaggedPosition(l[0].tag, l[0].startPointer + inputItem), nodeIdx); 
}

__device__
void OtherTaxi_dev::
BracketFind::end(InstTagT nodeIdx)
{

}

__device__
void OtherTaxi_dev::
CoordinateSwap::run(const TaggedPosition& inputItem, InstTagT nodeIdx)
{
  //Line* l = getParent(nodeIdx);
  double lon, lat;
  char* end;
  if(inputItem.pos[1] != '[') {
	lon = d_strtod(inputItem.pos + 1, &end);
	lat = d_strtod(end + 1, NULL);
	Position p = Position(inputItem.tag, lat, lon);	//Swap coordinated for output
	//printf("[%d, %d]\t\t%lf, %lf\n", blockIdx.x, l[0].tag, lat, lon);
	push(p, nodeIdx);
  }
}

__device__
unsigned int OtherTaxi_dev::
__enumerateFor_BracketFind::findCount(InstTagT nodeIdx)
{
	Line* l = getParent(nodeIdx);
	return l[0].length;
}

