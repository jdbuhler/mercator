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
//filter<InputView>::run(size_t const & inputItem, unsigned int nInputs)
{
  //unsigned int tid = threadIdx.x;
  unsigned int v;
  
  //if (tid < nInputs)
    v = munge((unsigned int) inputItem);
  
  // If no channel is specified, push sends a value to the module's
  // first output channel.
  //push(v, tid < nInputs && (v % 2) == 1); // defaults to pushing to Out::accept
  //push(v, (v % 2) == 1); // defaults to pushing to Out::accept
  if(v % 2 == 1)
  	push(v); // defaults to pushing to Out::accept
}

__MDECL__
void InterruptSim_dev::
otherFilter<InputView>::run(unsigned int const & inputItem)
//otherFilter<InputView>::run(unsigned int const & inputItem, unsigned int nInputs)
{
  //unsigned int tid = threadIdx.x;
  unsigned int v;
  
  //if (tid < nInputs)
    v = munge((unsigned int) inputItem);
  
  // If no channel is specified, push sends a value to the module's
  // first output channel.
  //push(v, tid < nInputs && (v % 2) == 0); // defaults to pushing to Out::accept
  //push(v, (v % 2) == 0); // defaults to pushing to Out::accept
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
	//bool saveState = false;
	__shared__ bool saveState[THREADS_PER_BLOCK];
	__shared__ bool continueSaving;
	//if(threadIdx.x == 0) {
	//	saveState = false;
	//}
	//__syncthreads();

	unsigned int v = (tid < nInputs ? munge(inputItem) : 0);
	do {

		//v = munge(inputItem);

		saveState[tid] = pushAndCheck(v + i, tid < nInputs && i < v % MAX_EXPAND);

		__syncthreads();

		if(IS_BOSS()) {
			continueSaving = false;
			for(unsigned int j = 0; j < nInputs && !continueSaving; ++j) {
				if(saveState[j]) {
					continueSaving = true;
				}
			}
		}

		__syncthreads();

		//if(!saveState[tid]) {
		if(!continueSaving) {
			++i;
		}

	} while(i < MAX_EXPAND && continueSaving);
	//} while(i < MAX_EXPAND && !saveState[tid]);
	//} while(i < threadIdx.x && !saveState);
	__syncthreads();

	/*
	if(IS_BOSS()) {
		continueSaving = false;
		for(unsigned int j = 0; j < nInputs && !continueSaving; ++j) {
			if(saveState[j]) {
				continueSaving = true;
			}
		}
	}

	__syncthreads();
	*/

	//if(saveState && !(i == v % 4)) {
	if(tid < nInputs) {
		if(continueSaving) {
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
	__syncthreads();
}

__MDECL__
void InterruptSim_dev::
expand<InputView>::cleanup()
{
	if(threadIdx.x == 0) {
		delete [] getState()->restoreArray;
	}
}

