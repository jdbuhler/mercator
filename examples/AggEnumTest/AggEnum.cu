#include "AggEnum_dev.cuh"

__device__
void AggEnum_dev::
EnumModule::begin(InstTagT nodeIdx)
{
  //if(threadIdx.x == 0)
//	printf("[%d, %d] CALLED BEGIN ENUM MODULE\n", blockIdx.x, threadIdx.x); 
}

__device__
void AggEnum_dev::
EnumModule::run(const unsigned int& inputItem, InstTagT nodeIdx)
{
   unsigned int* z = getParent(nodeIdx);
   //if(threadIdx.x == 0)
   //	printf("ENUM_MODULE Z = %p\n", z);
   push(inputItem, nodeIdx);
}

__device__
void AggEnum_dev::
EnumModule::end(InstTagT nodeIdx)
{
  //if(threadIdx.x == 0)
	//printf("[%d, %d] CALLED END ENUM MODULE\n", blockIdx.x, threadIdx.x); 
}

__device__
void AggEnum_dev::
Filter::begin(InstTagT nodeIdx)
{
  //if(threadIdx.x == 0)
//	printf("[%d, %d] CALLED BEGIN FILTER MODULE\n", blockIdx.x, threadIdx.x); 
}

__device__
void AggEnum_dev::
Filter::run(const unsigned int& inputItem, InstTagT nodeIdx)
{
   unsigned int* z = getParent(nodeIdx);
  // if(threadIdx.x == 0)
   //	printf("FILTER Z = %p\n", z);
   push(inputItem, nodeIdx);
}

__device__
void AggEnum_dev::
Filter::end(InstTagT nodeIdx)
{
  //if(threadIdx.x == 0)
//	printf("[%d, %d] CALLED END FILTER MODULE\n", blockIdx.x, threadIdx.x); 
}

__device__
void AggEnum_dev::
AggModule::begin(InstTagT nodeIdx)
{

}

__device__
void AggEnum_dev::
AggModule::run(const unsigned int& inputItem, InstTagT nodeIdx)
{
   unsigned int* z = getParent(nodeIdx);
   //if(threadIdx.x == 0)
   //	printf("AGG_MODULE Z = %p\n", z);
   push(inputItem, nodeIdx);
}

__device__
void AggEnum_dev::
AggModule::end(InstTagT nodeIdx)
{

}

__device__
unsigned int AggEnum_dev::
__enumerateFor_EnumModule::findCount(InstTagT nodeIdx)
{
  //if(threadIdx.x == 0)
	//printf("CALLED FINDCOUNT\n"); 
  //if(threadIdx.x == 0)
  //	push(1, nodeIdx);
  //return 6000;
  //return 5888;
  //unsigned int* z = AggEnum_dev::__enumerateFor_EnumModule::getParent(nodeIdx);
  unsigned int* z = getParent(nodeIdx);
  //printf("PARENT = %d\n", z[0]);

  //return 3;
  return z[0];
}

      //__device__
      //void AggEnum_dev::__enumerateFor_EnumModule::run(const unsigned int& inputItem, InstTagT nodeIdx) { push(inputItem, nodeIdx); }
