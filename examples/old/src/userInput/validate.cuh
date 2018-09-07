#ifndef __VALIDATE_CUH
#define __VALIDATE_CUH

#include "PipeEltT.cuh"   

/**
 * @brief Validate output of a low-pass filtering app based on 
 *         app inputs and outputs.
 *
 * @param inputs Input array
 * @param inSize Size of input array
 * @param outputs Output array
 * @param outSize Size of output array
 * @param filterThresh Value below which inputs should be accepted
 * @param gamma Replication factor of inputs to outputs; i.e.,
 *              num copies of each passing input that should exist 
 *              in output array
 *
 * @return true if i/o arrays are consistent; false otherwise
 */
bool validate_lowpassFilterApp_outputs(const PipeEltT* inputs,
				       int inSize,
				       const PipeEltT* outputs,
				       int outSize,
				       int filterThresh,
				       int gamma);

bool validate_lowpassFilterApp_outputs_inorder(const PipeEltT* inputs,
					       int inSize,
					       const PipeEltT* outputs,
					       int outSize,
					       int filterThresh,
					       int gamma);

#endif
