#include "LiteAggEnum_dev.cuh"

__device__
void LiteAggEnum_dev::
EnumModule::run(const unsigned int& inputItem, InstTagT nodeIdx)
{
  push(inputItem, nodeIdx); 
}

__device__
void LiteAggEnum_dev::
Filter::run(const unsigned int& inputItem, InstTagT nodeIdx)
{
  push(inputItem, nodeIdx); 
}

__device__
void LiteAggEnum_dev::
AggModule::run(const unsigned int& inputItem, InstTagT nodeIdx)
{
  push(inputItem, nodeIdx); 
}

