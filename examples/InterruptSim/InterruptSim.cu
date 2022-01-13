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
	if(threadIdx.x == 0) {
		getState()->restoreArray = new unsigned int[threadWidth];
	}
	__syncthreads();

	if(threadIdx.x < threadWidth) {
		getState()->restoreArray[threadIdx.x] = 0;
	}

	__syncthreads();
}

__MDECL__
void InterruptSim_dev::
expand<InputView>::run(unsigned int const & inputItem, unsigned int nInputs)
{
  	unsigned int tid = threadIdx.x;
	unsigned int i = getState()->restoreArray[threadIdx.x];
	restoreComplete();
	bool saveState = false;

	unsigned int v = (tid < nInputs ? munge(inputItem) : 0);
	do {

		saveState = pushAndCheck(v + i, tid < nInputs && i < v % MAX_EXPAND);

		if(!saveState) {
			++i;
		}

	} while(i < MAX_EXPAND && !saveState);
	__syncthreads();

	if(tid < nInputs) {
		if(saveState) {
			getState()->restoreArray[threadIdx.x] = i;
			printf("[%d, %d] SET RESTORE = %d\n", blockIdx.x, threadIdx.x, i);
		}
		else {
			getState()->restoreArray[threadIdx.x] = 0;
			restoreComplete();
			printf("[%d, %d] RESET RESTORE\n", blockIdx.x, threadIdx.x);
		}
	}

	if(tid == 0) {
		printf("[%d, %d] NINPUTS = %d\n", blockIdx.x, threadIdx.x, nInputs);
	}
}

__MDECL__
void InterruptSim_dev::
expand<InputView>::cleanup()
{
	if(threadIdx.x == 0) {
		delete [] getState()->restoreArray;
	}
}

