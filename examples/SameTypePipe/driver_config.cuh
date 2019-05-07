#ifndef __DRIVER_CONFIG_CUH
#define __DRIVER_CONFIG_CUH

/**
 * @file driver_config.cuh
 * @brief Define runtime execution and app-specific parameters 
 *         used by MERCATOR examples.
 *
 */

/////////////////////////////////////////////////////////////////////////
// GENERAL PARAMETERS
/////////////////////////////////////////////////////////////////////////

// select desired mapping, expressed in terms of <elts>TO<threads>
#define MAPPING_1TO1
// MANY_THREADS
//#define MAPPING_1TO2
//#define MAPPING_1TO4
// MANY_ITEMS
//#define MAPPING_2TO1
//#define MAPPING_4TO1

// input buffer size (synthetic apps currently fill input buffer)
// Good default values:
//   debug mode: 256K
//   release mode: 1M
//#define NUM_INPUTS 128
//#define NUM_INPUTS 256
//#define NUM_INPUTS 512
//#define NUM_INPUTS 1024
//#define NUM_INPUTS 4096
//#define NUM_INPUTS 8192
//#define NUM_INPUTS 8704
//#define NUM_INPUTS 4100
//#define NUM_INPUTS (1 << 15)    // 32K
//#define NUM_INPUTS (1 << 18)      // 256K
//#define NUM_INPUTS (1 << 19)      // 512K
//#define NUM_INPUTS (1 << 19 | 1 << 18)      // 512K + 256K
//#define NUM_INPUTS (1 << 20)    // 1M
//#define NUM_INPUTS 10 * (1 << 20)    // 10M
//#define NUM_INPUTS 100 * (1 << 20)    // 100M
//#define NUM_INPUTS 10 * (1 << 20)    // 10M
#define NUM_INPUTS 100 * (1 << 20)    // 100M

// use random input stream?
// if 1, select each input uniformly at random
// if 0, use sequential IDs for inputs
#define USE_RANDOM_INPUTS

// use deterministic and repeatable input stream (if random)?
// intention: yes for debugging, no for testing
// -- toggles different seeding options for PRNG that generates inputs
// -- NB: only used if USE_RANDOM_INPUTS set to 1
#define USE_REPEATABLE_INPUTS

// filtering rate for BlackScholes work
// NB: Must be of form "FILTER_RATE" for runSynthExps.sh 
//      config script to work
#define FILTER_RATE 0.5

// Number of iterations of dummy work to be done at each node or pseudo-node.
// NB: release-scale experiments (176 blocks, 1M inputs): 5k-25k iters
// debug-scale experiments: 20-100 iters
//#define WORK_ITERS_4THREADS 12500
#define WORK_ITERS_4THREADS 12500

//--- others derived from num iters when many threads are mapped
//---  to each element
//--- x4 normalizes for up to 1:4 elts:threads ratio
#define WORK_ITERS_1TO1MAP (4 * WORK_ITERS_4THREADS)
//--- Num iters for ManyItems case is same as for 1-to-1 case 
//     since WORK_ITERS specifies iters per thread PER ITEM
#define WORK_ITERS_MANYITEMS (WORK_ITERS_1TO1MAP)

// set dummy work iterations for all mapping options
#if defined(MAPPING_1TO1)
  #define WORK_ITERS (WORK_ITERS_1TO1MAP)
#elif defined(MAPPING_1TO2) 
  #define WORK_ITERS (2 * WORK_ITERS_4THREADS)
#elif defined(MAPPING_1TO4)
  #define WORK_ITERS (WORK_ITERS_4THREADS)
#elif defined(MAPPING_2TO1) 
  #define WORK_ITERS (WORK_ITERS_MANYITEMS)
#elif defined(MAPPING_4TO1)
  #define WORK_ITERS (WORK_ITERS_MANYITEMS)
#else
  #define WORK_ITERS 0
#endif

#endif
