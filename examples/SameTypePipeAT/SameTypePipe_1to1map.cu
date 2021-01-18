#include "SameTypePipe_1to1map_AT_dev.cuh"

#include <curand_kernel.h>

#include "BlackScholes/BlackScholes_device.cuh"

__MDECL__
void 
SameTypePipe_1to1map_AT_dev::
A<InputView>::init()
{
  const int threadWidth = getNumActiveThreads();
  
  // all nodes share the same random state vector.  We
  // are not able to guarantee deterministic behavior even
  // if we give them separate vectors, so who cares.
  if (threadIdx.x == 0)
    getState()->randState = new curandState_t [threadWidth];
  
  __syncthreads();
  
  // initialize the random state vector
  if (threadIdx.x < threadWidth)
    {
      curand_init(getAppParams()->seed,
		  blockIdx.x * threadWidth + threadIdx.x,
		  0, &getState()->randState[threadIdx.x]);
    }
}

__MDECL__
void 
SameTypePipe_1to1map_AT_dev::
A<InputView>::run(const PipeEltT& inputItem, unsigned int nInputs)
{ 
  unsigned int tid = threadIdx.x;

  bool passThrough = false;
  PipeEltT myItem;
  
  if (tid < nInputs)
    {
      myItem = inputItem;
      
      ///// computation:
      //  do NUM_OPTS rounds of Black-Scholes calculations
      int num_opts = myItem.get_workIters();
      
      //  parameters for BlackScholes fcn call;
      //   both hold results (passed by value)
      float callResult = 1.0f;
      float putResult  = 1.0f;
      
      // item's state for PRNG -- cache it for efficiency
      curandState_t randState = getState()->randState[threadIdx.x];
      
      // call BlackScholes fcn
      doBlackScholes_fast(callResult, putResult, num_opts, randState);
      
      //save updated random state
      getState()->randState[threadIdx.x] = randState;
      
      // store result in item (addition combo is arbitrary)
      myItem.set_floatResult(callResult + putResult);
      
      ///// filtering:
      //  filter out item if its ID is in upper filterRate-fraction of
      //  bounded (node-specific) range; else push item downstream
      auto params = getParams();
      
      float fr = params->filterRate;
      int ub   = params->upperBound;
      
      // border b/t filtered (above) and forwarded (below) items
      int thresh = (1.0f - fr) * ub;
      
      passThrough = (myItem.get_ID() < thresh);
    }
  
  push(myItem, passThrough);
}

__MDECL__
void
SameTypePipe_1to1map_AT_dev::
A<InputView>::cleanup()
{
  if (threadIdx.x == 0)
    delete [] getState()->randState;
}
