#ifndef __APPDRIVER_CUH
#define __APPDRIVER_CUH

//
// @file AppDriver.cuh
// @brief User-facing host object used to set up and run the app
//
// MERCATOR
// Copyright (C) 2018 Washington University in St. Louis; all rights reserved.
//

#include <cstddef>
#include <iostream>

// profiling
#ifdef PROFILE_TIME
#include <cuda_profiler_api.h>
#endif

#include "AppDriverBase.cuh"

#include "support/util.cuh"

#include "options.cuh"

#include "kernels/initKernel.cuh"
#include "kernels/mainKernel.cuh"
#include "kernels/cleanupKernel.cuh"

#ifdef INSTRUMENT_TIME_HOST
#include "instrumentation/host_timer.h"
#endif

namespace Mercator  {

  //
  // @class AppDriver
  // @brief host-side driver for a MERCATOR application
  //
  // @tparam HostParamsT type of host-side parameter struct passed to device
  // @tparam DevApp type of device-side application
  //
  template<typename HostParamsT, typename DevApp>
  class AppDriver : public AppDriverBase<HostParamsT> {
    
  public:
    
    // @brief get number of active blocks for this app
    //
    int getNBlocks() const { return nBlocks; }
    
    
    void setAppName(char* name, int size) { 
        //memcopy(appName, name, size%20);
        //doing by hand to not load extra includes
        nameSet=true;
        for (int i=0;i<size%20; i++){
          appName[i]=name[i];
        }
    }
    // @brief constructor sets up the device context
    //
    // @param stream CUDA stream in which to run
    // @param deviceId CUDA device on which to run
    //
    AppDriver(cudaStream_t istream,
	      int ideviceId)
      : stream(istream),
	deviceId(ideviceId)
    {
      using namespace std;
      
      // switch to our device
      int prevDeviceId = switchDevice(deviceId);
      
      //
      // Make sure we have adequate stack and heap size for the app
      //
      
      size_t stackSize;      
      cudaDeviceGetLimit(&stackSize, cudaLimitStackSize);
      
      if (stackSize < DevApp::DEVICE_STACK_SIZE)
	{
	  cudaDeviceSetLimit(cudaLimitStackSize, DevApp::DEVICE_STACK_SIZE);
	  gpuErrchk( cudaPeekAtLastError() );
	}
      
#ifdef PRINT_MEM_USAGE
      // print new size to double-check setting
      cudaDeviceGetLimit(&stackSize, cudaLimitStackSize);
      cout << "CUDA stack size: "
	   << stackSize
	   << endl;
#endif
      
      // increase deviceheap size
      size_t devHeapSize;
      cudaDeviceGetLimit(&devHeapSize, cudaLimitMallocHeapSize);
      
      if (devHeapSize < DevApp::DEVICE_HEAP_SIZE)
	{
	  cudaDeviceSetLimit(cudaLimitMallocHeapSize, DevApp::DEVICE_HEAP_SIZE);
	  gpuErrchk( cudaPeekAtLastError() );
	}
      
#ifdef PRINT_MEM_USAGE
      // print new size to double-check setting
      cudaDeviceGetLimit(&devHeapSize, cudaLimitMallocHeapSize);
      cout << "New CUDA device heap size: "
	   << devHeapSize
	   << endl;
#endif
      
      //
      // OCCUPANCY CALCULATION
      //
      // Let CUDA suggest a number of active blocks/SM, and use that to
      // determine the number of blocks to launch.  I'm not sure
      // this is actually going to be feasible -- it depends on how
      // well the occupancy calculator can navigate our virtual function
      // usage.  But there should be no harm in launching with more
      // blocks than we can actually use at once.
      //
      
      int nBlocksPerSM_suggested;
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nBlocksPerSM_suggested,
						    mainKernel<DevApp>,
						    DevApp::THREADS_PER_BLOCK,
						    0);
      
      if (nBlocksPerSM_suggested == 0) // unable to launch at all
	{
	  cerr << "ERROR: app kernel cannot launch with the requested"
	       << " # of threads per block (" 
	       << DevApp::THREADS_PER_BLOCK 
	       << ")" << endl;
	}
      
      // assume we are using the first device in the host (for now)
      CUdevice dev;
      cuDeviceGet(&dev, deviceId);
      
      // get # of SMs on the device
      int numSMs;
      cudaDeviceGetAttribute(&numSMs, 
			     cudaDevAttrMultiProcessorCount,
			     dev);
      
    #ifdef USE_MAX_BLOCKS
      nBlocks = nBlocksPerSM_suggested * numSMs;
    #endif
      
    #ifdef USE_ONE_BLOCKS
      nBlocks = 1;
    #endif

    #ifdef USE_SM_BLOCKS
      nBlocks= numSMs;
    #endif
    
    #ifdef USE_X_BLOCKS 
      nBlocks = USE_X_BLOCKS;
    #endif
    

      // allocate space on device for passing host parameter struct
      cudaMalloc(&hostParams, sizeof(HostParamsT));
      gpuErrchk( cudaPeekAtLastError() );
      
      // allocate space on device for source's tail pointer
      cudaMalloc(&sourceTailPtr, sizeof(size_t));
      gpuErrchk( cudaPeekAtLastError() );
      
      // allocate space on device to hold all per-block app objects
      cudaMalloc(&deviceAppObjs, sizeof(DevApp *) * nBlocks);
      gpuErrchk( cudaPeekAtLastError() );
      
#ifdef INSTRUMENT_TIME_HOST
      elapsedTime_init = 0.0;
      elapsedTime_main = 0.0;
      elapsedTime_cleanup = 0.0;
#endif

#ifdef INSTRUMENT_TIME_HOST
      timer.start(stream);
#endif
      
      // initialize the device-side structures, including setting
      // internal pointers to our (as-yet uninitialized)
      // shared tail pointer and host parameter struct
      initKernel<<<nBlocks, 1, 0, stream>>>(sourceTailPtr, 
					    hostParams, 
					    deviceAppObjs);
      
      // synchronize to make sure the initialization was successful
      gpuErrchk( cudaStreamSynchronize(stream) );
#ifdef INSTRUMENT_TIME_HOST
      timer.stop(stream);
      elapsedTime_init = timer.elapsed();
#endif
      
      // switch back to caller's device
      switchDevice(prevDeviceId);
    }
    
    
    //
    // @brief launch a MERCATOR app from the host side
    //  Will return after kernel is launched, but possibly
    //  before it is finished.  Use join() to wait for kernel.
    //
    // @param params host-side parameters for run
    //
    void runAsync(const HostParamsT *params)
    {
      // switch to our device
      int prevDeviceId = switchDevice(deviceId);
      
      
      // make a copy of the current parameter data so that the user
      // can change this structure safely after we return.
      HostParamsT *tmpParams = new HostParamsT;
      memcpy(tmpParams, params, sizeof(HostParamsT));
    
      // copy the current parameter data to the device
      cudaMemcpyAsync(hostParams, tmpParams, 
		      sizeof(HostParamsT), cudaMemcpyHostToDevice,
		      stream);
      gpuErrchk( cudaPeekAtLastError() );
      
      // make sure we clean up our copy of the parameters after
      // cudaMemcpyAsync() is done using it.
      cudaStreamAddCallback(stream, freeCallback, tmpParams, 0);
      gpuErrchk( cudaPeekAtLastError() );
      
      // reset the source's tail pointer
      cudaMemsetAsync(sourceTailPtr, 0, sizeof(size_t), stream);
      gpuErrchk( cudaPeekAtLastError() );
      
      gpuErrchk( cudaStreamSynchronize(stream) );
#ifdef INSTRUMENT_TIME_HOST
      timer.start(stream);
#endif
      mainKernel<<<nBlocks, DevApp::THREADS_PER_BLOCK, 0, stream>>>(deviceAppObjs);
      gpuErrchk( cudaPeekAtLastError() );
      
#ifdef INSTRUMENT_TIME_HOST
      timer.stop(stream);
#endif
      // switch back to caller's device
      switchDevice(prevDeviceId);
    }
    
    
    //
    // @brief wait for the most recent run to complete
    //  Will block the CPU thread until the run finishes
    //
    void join()
    {
      // switch to our device
      int prevDeviceId = switchDevice(deviceId);
      
      // wait for the ops in the current stream to finish
      gpuErrchk( cudaStreamSynchronize(stream) );
      
#ifdef INSTRUMENT_TIME_HOST
      elapsedTime_main += timer.elapsed();
#endif
      
      // switch back to caller's device
      switchDevice(prevDeviceId);
    }
    
    //
    // @brief destructor calls cleanup kernel and prints any
    // host performance data
    //
    ~AppDriver()
    {
      using namespace std;

      // switch to our device
      int prevDeviceId = switchDevice(deviceId);
      
#ifdef INSTRUMENT_TIME_HOST
      timer.start(stream);
#endif
      
      cleanupKernel<<<nBlocks, 1, 0, stream>>>(deviceAppObjs);
      
      // make sure we have caught any errors from this app
      gpuErrchk( cudaStreamSynchronize(stream) );

#ifdef INSTRUMENT_TIME_HOST
      timer.stop(stream);
      elapsedTime_cleanup = timer.elapsed();
#endif

      cudaFree(deviceAppObjs);
      cudaFree(hostParams);
      cudaFree(sourceTailPtr);
      
#ifdef INSTRUMENT_TIME
      // print clock rate
      cudaDeviceProp prop;
      cudaGetDeviceProperties(&prop, deviceId);
      cout << "GPU clock rate (cycles/ms): "
	   << prop.clockRate    
	   << endl;
#endif
      
#ifdef INSTRUMENT_TIME_HOST
      float totalTime = 
	elapsedTime_init + 
	elapsedTime_main + 
	elapsedTime_cleanup;
      
      cout.setf(ios::fixed, ios::floatfield);
      int oldPrec = cout.precision(2);
      if (nameSet)
        printf("*** %s runtimes (ms):\n", appName);
      else
        cout << "***Application runtimes (ms):" << endl;
      cout << "\tinit: " 
	   << elapsedTime_init << "ms ("
	   << elapsedTime_init*100.0/totalTime
	   << "%)" << endl;
      cout << "\tmain: " 
	   << elapsedTime_main << "ms ("
	   << elapsedTime_main*100.0/totalTime
	   << "%)" << endl;
      cout << "\tcleanup: " 
	   << elapsedTime_cleanup << "ms ("
	   << elapsedTime_cleanup*100.0/totalTime
	   << "%)" << endl;
      cout << "\tTotal time: "
	   << totalTime << "ms" << endl;
      cout.precision(oldPrec);
#endif    // if INSTRUMENT_TIME_HOST

      // switch back to caller's device
      switchDevice(prevDeviceId);
    }
    
    
    //
    // @brief change the app's binding to a new stream
    //  Completes any pending work on the old stream before rebinding,
    //  and synchronizes with host.
    //
    // @param newStream new stream to bind to
    void bindToStream(cudaStream_t newStream)
    {
      int prevDeviceId = switchDevice(deviceId);
      
      // finish all ops on old stream and check for errors
      gpuErrchk( cudaStreamSynchronize(stream) );
      
      stream = newStream;
      
      switchDevice(prevDeviceId);
    }
    
  private:


    cudaStream_t stream;
    int deviceId;
    
    DevApp **deviceAppObjs;
    HostParamsT *hostParams;
    int nBlocks;
    bool nameSet=false;
    char appName[20];   

    size_t *sourceTailPtr;
    
#ifdef INSTRUMENT_TIME_HOST    
    GpuTimer timer;
    
    double elapsedTime_init;
    double elapsedTime_main;
    double elapsedTime_cleanup;
#endif
    
    //
    // @brief change the active CUDA device to the specified value
    // and return the previous active device
    //
    int switchDevice(int device)
    {
      int oldDevice;
      cudaGetDevice(&oldDevice);
      cudaSetDevice(device);
      return oldDevice;
    }
    
    //
    // @brief callback to delete temporary copy of params allocated
    //  in runAsync()
    //
    static void freeCallback(cudaStream_t stream, cudaError_t status,
			     void *tmpParams)
    {
      delete (HostParamsT *) tmpParams;
    }

  };

}// end Mercator namespace

#endif
