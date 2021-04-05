/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
 * Modified by Steve Cole (coles) Spring 2017.
 */

#ifndef __BLACKSCHOLES_DEVICE_CUH
#define __BLACKSCHOLES_DEVICE_CUH

// for random numbers on device
#include <curand.h>
#include <curand_kernel.h>

////////////////////////////////////////////////////////////////////////////////
// Helper function, returning uniformly distributed
// random float in [0, 1] range
////////////////////////////////////////////////////////////////////////////////
__device__
float RandFloat_01(int seed)
{
  curandState_t state;
  
  curand_init(seed,
              threadIdx.x,  /*  sequence num */
              0,  /*  offset in seq */
              &state);

  return curand_uniform(&state);
}

//////////////////////////////////////////////////////////////////////////////
// Helper function, returning uniformly distributed
// random float in [low, high] range
//////////////////////////////////////////////////////////////////////////////
__device__
float RandFloat(float low, float high, unsigned int seed)
{
  assert(high > 0 && low >= 0  && high > low);

  float float01 = RandFloat_01(seed);

  float range = high - low;

  return float01 * range + low;
}

// faster version: take already-init'd curand state
__device__
float RandFloat_fast(float low, float high, curandState_t &state)
{
  assert(high > 0 && low >= 0  && high > low);

  float float01 = curand_uniform(&state);

  float range = high - low;

  return float01 * range + low;
}

/////////////////////////////////////////////////////////////////////////////
// Polynomial approximation of cumulative normal distribution function
/////////////////////////////////////////////////////////////////////////////
__device__ inline float cndGPU(float d)
{
    const float       A1 = 0.31938153f;
    const float       A2 = -0.356563782f;
    const float       A3 = 1.781477937f;
    const float       A4 = -1.821255978f;
    const float       A5 = 1.330274429f;
    const float RSQRT2PI = 0.39894228040143267793994605993438f;

    float
      K = __fdividef(1.0f, (1.0f + 0.2316419f * fabsf(d)));
    
    float
    cnd = RSQRT2PI * __expf(- 0.5f * d * d) *
          (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));

    if (d > 0)
        cnd = 1.0f - cnd;

    return cnd;
}


///////////////////////////////////////////////////////////////////////////////
// Black-Scholes formula for both call and put
///////////////////////////////////////////////////////////////////////////////
__device__ inline void BlackScholesBodyGPU(
    float &CallResult,
    float &PutResult,
    float S, //Stock price
    float X, //Option strike
    float T, //Option years
    float R, //Riskless rate
    float V  //Volatility rate
)
{
    float sqrtT, expRT;
    float d1, d2, CNDD1, CNDD2;

    sqrtT = sqrtf(T);
    d1 = __fdividef(__logf(S / X) + (R + 0.5f * V * V) * T, V * sqrtT);
    d2 = d1 - V * sqrtT;

    CNDD1 = cndGPU(d1);
    CNDD2 = cndGPU(d2);

    //Calculate Call and Put simultaneously
    expRT = __expf(- R * T);
    CallResult = S * CNDD1 - X * expRT * CNDD2;
    PutResult  = X * expRT * (1.0f - CNDD2) - S * (1.0f - CNDD1);
}

//////////////////////////////////////////////////////
// Create and process specified number of options on GPU
// Upon completion, arguments will hold values of
//   last option calculated.
//   --adapted from SDK test-harness code by coles
//////////////////////////////////////////////////////
__device__
void doBlackScholes(
  float& callResult, 
  float& putResult,
  int numOpts
  )
{
  constexpr float      RISKFREE = 0.02f;
  constexpr float    VOLATILITY = 0.30f;

  for(int i=0; i < numOpts; ++i)
  {
    // make result depend on input
    assert(callResult >= 0);

    int seed = (int)callResult;

    float stockPrice    = RandFloat(5.0f, 30.0f, seed);
    float optionStrike  = RandFloat(1.0f, 100.0f, seed);
    float optionYears   = RandFloat(0.25f, 10.0f, seed);

    BlackScholesBodyGPU(
      callResult,
      putResult,
      stockPrice,
      optionStrike,
      optionYears,
      RISKFREE,
      VOLATILITY
    );
  }

}

// does not init curand state
// --but also does not afford possibility of chaining input/output
//   via init seed
__device__
void doBlackScholes_fast(
  float& callResult, 
  float& putResult,
  int numOpts,
  curandState_t &state
  )
{
  constexpr float      RISKFREE = 0.02f;
  constexpr float    VOLATILITY = 0.30f;

  for(int i=0; i < numOpts; ++i)
  {
    // make result depend on input
    
    float stockPrice    = RandFloat_fast( 5.0f,  30.0f, state);
    float optionStrike  = RandFloat_fast( 1.0f, 100.0f, state);
    float optionYears   = RandFloat_fast( 0.25f, 10.0f, state);

    BlackScholesBodyGPU(
      callResult,
      putResult,
      stockPrice,
      optionStrike,
      optionYears,
      RISKFREE,
      VOLATILITY
    );
  }

}
#endif  // #ifndef __BLACKSCHOLES_KERNEL_CUH
