#include "InterruptSim_dev.cuh"


#define MAX_EXPAND 40

__device__
unsigned int munge(unsigned int key)
{
  key = (key << 15) - key - 1;
  key = key ^ (key >> 12);
  key = key + (key << 2);
  key = key ^ (key >> 4);
  key = key * 2057;
  key = key ^ (key >> 16);
  return key;
}


__MDECL__
void InterruptSim_dev::
filter<InputView>::run(size_t const & inputItem)
{
  unsigned int v;
  
  v = munge((unsigned int) inputItem);
  
  // If no channel is specified, push sends a value to the module's
  // first output channel.
  if(v % 2 == 1)
    push(v); // defaults to pushing to Out::accept
}


__MDECL__
void InterruptSim_dev::
otherFilter<InputView>::run(unsigned int const & inputItem)
{
  unsigned int v;
  
  v = munge((unsigned int) inputItem);
  
  // If no channel is specified, push sends a value to the module's
  // first output channel.
  if(v % 2 == 0)
	  push(v);
}


__MDECL__
void InterruptSim_dev::
expand<InputView>::init()
{
  const int threadWidth = getNumActiveThreads();

  if (threadIdx.x == 0)
    getState()->i = 0;
  
  __syncthreads(); // protect state update
}


__MDECL__
unsigned int InterruptSim_dev::
expand<InputView>::run(unsigned int const & inputItem, unsigned int nInputs)
{
  unsigned int tid = threadIdx.x;
  unsigned int i = getState()->i;
  
  unsigned int v = (tid < nInputs ? munge(inputItem) % MAX_EXPAND : 0);

  bool canContinue;
  do
    {
      canContinue = push(v + i, tid < nInputs && i < v);
      
      i++;
    }
  while (i < MAX_EXPAND && canContinue);

  if (threadIdx.x == 0)
    getState()->i = (i < MAX_EXPAND ? i : 0);

  __syncthreads(); // protect state update
  
  //if (threadIdx.x == 0)
  //  printf("%d SET RESTORE = %d\n", blockIdx.x, i);
  
  return (i < MAX_EXPAND ? 0 : nInputs);
}

__MDECL__
void InterruptSim_dev::
expand<InputView>::cleanup()
{
}
