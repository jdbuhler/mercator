#include "SameTypePipe_1to1map_dev.cuh"

#include <curand_kernel.h>

#include "BlackScholes/BlackScholes_device.cuh"
#define UPPERBOUND 3000000
#define LOWERBOUND 2500000
__device__
void 
SameTypePipe_1to1map_dev::
A::init()
{
  
  const int threadWidth = getNumActiveThreads();
  const int nInstances  = getNumInstances();
  // allocate a random state vector for each node
  __shared__ curandState_t *randState;
  if (threadIdx.x == 0)
    randState = new curandState_t [threadWidth];
  
  __syncthreads(); // all threads must see randState

  // all nodes share the same random state vector.  We
  // are not able to guarantee deterministic behavior even
  // if we give them separate vectors, so who cares.
  if (threadIdx.x < nInstances)
    getState()->randState[threadIdx.x] = randState;
  
  // initialize teh random state vector
  if (threadIdx.x < threadWidth)
    {
      curand_init(getAppParams()->seed,
		  blockIdx.x * threadWidth + threadIdx.x,
		  0, &randState[threadIdx.x]);
    }

#ifdef INSTRUMENT_FG_TIME
 //set upperbound for data collection
    if(IS_BOSS()){
    setFGContainerBounds((unsigned long long)LOWERBOUND, (unsigned long long)UPPERBOUND);
    }
  __syncthreads(); // all threads must see updates to the bounds
#endif  
}

__device__
void 
SameTypePipe_1to1map_dev::
A::run(const PipeEltT &inputItem, InstTagT nodeIdx)
{ 
  PipeEltT myItem = inputItem;
  
  ///// computation:
  //  do NUM_OPTS rounds of Black-Scholes calculations
  int num_opts = myItem.get_workIters();
  
  //  parameters for BlackScholes fcn call;
  //   both hold results (passed by value)
  float callResult = 1.0f;
  float putResult  = 1.0f;
  
  // item's state for PRNG -- cache it for efficiency
  curandState_t randState = getState()->randState[nodeIdx][threadIdx.x];
  
  // call BlackScholes fcn
  doBlackScholes_fast(callResult, putResult, num_opts, randState);
  
  //save updated random state
  getState()->randState[nodeIdx][threadIdx.x] = randState;
  
  // store result in item (addition combo is arbitrary)
  myItem.set_floatResult(callResult + putResult);
  
  ///// filtering:
  //  filter out item if its ID is in upper filterRate-fraction of
  //  bounded (node-specific) range; else push item downstream
  auto params = getParams();
  
  float fr = params->filterRate[nodeIdx];
  int ub   = params->upperBound[nodeIdx];
  
  // border b/t filtered (above) and forwarded (below) items
  int thresh = (1.0f - fr) * ub;
  
  bool passThrough = (myItem.get_ID() < thresh);
  if (passThrough)
    push(myItem, nodeIdx);
}

__device__
void
SameTypePipe_1to1map_dev::
A::cleanup()
{
  const int nInstances = getNumInstances();
  
  if (threadIdx.x == 0)
    delete [] getState()->randState[0]; // only delete once
}
