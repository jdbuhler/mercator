#include "Stat_dev.cuh"

#define NSIZE 2048

__device__
void Stat_dev::
EnumModule::begin(InstTagT nodeIdx)
{

}

__device__
void Stat_dev::
EnumModule::run(const unsigned int& inputItem, InstTagT nodeIdx)
{
  Composite* c = getParent(nodeIdx);
  //push the value to be enumerated from the array, starting from
  //startPointer of the current composite object.
  //push(c[0].startPointer[inputItem], nodeIdx); 
  push(inputItem, nodeIdx); 
}

__device__
void Stat_dev::
EnumModule::end(InstTagT nodeIdx)
{

}

__device__
void Stat_dev::
Filter::begin(InstTagT nodeIdx)
{

}

__device__
void Stat_dev::
Filter::run(const unsigned int& inputItem, InstTagT nodeIdx)
{
  push(inputItem, nodeIdx); 
}

__device__
void Stat_dev::
Filter::end(InstTagT nodeIdx)
{

}

__device__
void Stat_dev::
AggModule::init()
{
	__shared__ unsigned int* s;
	if(threadIdx.x == 0) {
		s = new unsigned int[NSIZE];
	}

	__syncthreads();

	if(threadIdx.x == 0) {
		for(unsigned int i = 0; i < NSIZE; ++i) {
			s[i] = 0;
		}
	}

	__syncthreads();

	if(threadIdx.x == 0) {
		getState()->avgState[0] = s;
	}
}

__device__
void Stat_dev::
AggModule::begin(InstTagT nodeIdx)
{
  //printf("PUSHED AN AGGREGATE (BEGIN)\n");
  //push(10.0, nodeIdx); 
  //printf("RESET AGGREGATE STATE\n");
  for(unsigned int i = 0; i < NSIZE; ++i)
  	getState()->avgState[0][i] = 0;
}

__device__
void Stat_dev::
AggModule::run(const unsigned int& inputItem, InstTagT nodeIdx)
{
  //push(inputItem, nodeIdx); 
  Composite* c = getParent(nodeIdx);
  //printf("InputPointer[%d]: %d, %d\n", threadIdx.x, inputItem, c[0].length);
 
  getState()->avgState[0][inputItem] = c[0].startPointer[inputItem];
}

__device__
void Stat_dev::
AggModule::end(InstTagT nodeIdx)
{
  //printf("PUSHED AN AGGREGATE\t\t%f\n", getState()->avgState[0][0]);
  //if(threadIdx.x == 0) //Would be required if signal handler was parallelized
  Composite* c = getParent(nodeIdx);
  unsigned int sum = 0;
  for(unsigned int i = 0; i < c[0].length; ++i) {
	//printf("AVGSTATE: %d\n", getState()->avgState[0][i]);
	sum += getState()->avgState[0][i];
  }
  double d = sum;
  //d /= c[0].length;
  pushAggregate(d, nodeIdx); 
  //printf("PUSHED AN AGGREGATE\t\t%f\n", d);
}

__device__
void Stat_dev::
AggModule::cleanup()
{
	if(threadIdx.x == 0)
		delete [] getState()->avgState[0];
}

__device__
unsigned int Stat_dev::
__enumerateFor_EnumModule::findCount(InstTagT nodeIdx)
{
	Composite* c = getParent(nodeIdx);
	//printf("SIZE: %d\n", c[0].length);
	return c[0].length;
	//return 1;
	//return 0;	//Replace this return with the number of elements found for this enumeration.
}

