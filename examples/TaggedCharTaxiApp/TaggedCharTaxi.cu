#include "TaggedCharTaxi_dev.cuh"
#include "StrFunc.cuh"

__device__
void TaggedCharTaxi_dev::
CharEnumerate::begin(InstTagT nodeIdx)
{

}

__device__
void TaggedCharTaxi_dev::
CharEnumerate::run(const unsigned int& inputItem, InstTagT nodeIdx)
{
  Line* l = getParent(nodeIdx);
  push(TaggedPosition(l[0].tag, l[0].startPointer + inputItem), nodeIdx);
}

__device__
void TaggedCharTaxi_dev::
CharEnumerate::end(InstTagT nodeIdx)
{

}

__device__
void TaggedCharTaxi_dev::
BracketFind::run(const TaggedPosition& inputItem, InstTagT nodeIdx)
{
  if(inputItem.pos[0] == '[')
	push(inputItem, nodeIdx); 
}

__device__
void TaggedCharTaxi_dev::
CoordinateSwap::run(const TaggedPosition& inputItem, InstTagT nodeIdx)
{
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
unsigned int TaggedCharTaxi_dev::
__enumerateFor_CharEnumerate::findCount(InstTagT nodeIdx)
{
	Line* l = getParent(nodeIdx);
	return l[0].length;
}

