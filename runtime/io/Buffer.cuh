#ifndef __BUFFER_CUH
#define __BUFFER_CUH

//
// @file Buffer.cuh
// Specification of typed device buffers used for I/O to MERCATOR apps
//
// MERCATOR
// Copyright (C) 2018 Washington University in St. Louis; all rights reserved.
//

#include <cstddef>
#include <cassert>

namespace Mercator {
  
  // POD type holding the essential properties of a Buffer, for
  // transfer between host and device
  template <typename T>
  struct BufferData {
    T *data;          // device storage used to hold items
    size_t capacity;  // size of allocated space
    size_t size;      // number of elements actually present
  };
  
  
  template <typename T>
  class Buffer {
    
  public:
    
    //
    // @brief constructor allocates a managed BufferData object to
    // facilitiate host-device communication about the buffer state,
    // plus a chunk of device memory (not managed) for the actual
    // buffer contents.
    //
    // @param capacity -- size of buffer
    //
    Buffer(size_t capacity)
    {
      cudaMallocManaged(&bufferData, sizeof(BufferData<T>));
      
      bufferData->capacity = capacity;
      cudaMalloc(&bufferData->data, capacity * sizeof(T));
      
      bufferData->size = 0;
    }
    
    //
    // @brief copy constructor makes a new buffer and transfers
    // an existing buffer's data via device-to-device copy.
    //
    // @param other -- existing buffer to copy
    //
    Buffer(const Buffer<T> &other)
    {
      cudaMallocManaged(&bufferData, sizeof(BufferData<T>));
      
      copyContents(bufferData, other.bufferData);
    }

    //
    // @brief operator= behaves as copy constructor, except that
    // it destroys the destination buffer's contents first.
    //
    // @param other -- existing buffer to copy
    //
    Buffer<T> &operator=(const Buffer<T> &other)
    {
      cudaFree(bufferData->data);
     
      copyContents(bufferData, other.bufferData);
      
      return *this;
    }
    
    
    //
    // @brief destructor cleans up allocations
    //
    ~Buffer()
    { 
      cudaFree(bufferData->data); 
      cudaFree(bufferData);
    }
    
    size_t capacity() const 
    { return bufferData->capacity; }
    
    size_t size() const 
    { return bufferData->size; }
    
    //
    // @brief copy data from a buffer to the host
    //
    // @param hostData target host buffer
    // @param nElts number of elements to read
    // @param offset starting offset for read in device buffer
    //
    void get(T *hostData, size_t nElts,
	     size_t offset = 0) const
    {
      cudaMemcpy(hostData, bufferData->data + offset * sizeof(T),
		 nElts * sizeof(T), cudaMemcpyDeviceToHost);
    }
    
    
    //
    // @brief asynchronous version of get() for streams
    //
    // @param hostData target host buffer
    // @param nElts number of elements to read
    // @param offset starting offset for read in device buffer
    // @param stream stream in which to perform the copy
    //
    void getAsync(T *hostData, size_t nElts,
		  size_t offset = 0,
		  cudaStream_t stream = 0) const
    {
      cudaMemcpyAsync(hostData, bufferData->data + offset * sizeof(T),
		      nElts * sizeof(T), cudaMemcpyDeviceToHost, stream);
    }
    

    //
    // @brief copy data from the host to a buffer,
    //   destroying any existing contents
    //
    // @param hostData source host buffer
    // @param nElts number of elements to copy to device
    //
    void set(const T *hostData, size_t nElts)
    {
      assert(nElts <= bufferData->capacity);
      
      cudaMemcpy(bufferData->data, hostData, 
		 nElts * sizeof(T), cudaMemcpyHostToDevice);
      bufferData->size = nElts;
    }

    
    //
    // @brief asynchronous version of put() for streams
    //
    // @param hostData source host buffer
    // @param nElts number of elements to copy to device
    // @param stream stream in which to perform the copy
    //
    void setAsync(const T *hostData, size_t nElts,
		  cudaStream_t stream = 0)
    {
      assert(nElts <= bufferData->capacity);
      
      cudaMemcpyAsync(bufferData->data, hostData, 
		      nElts * sizeof(T), cudaMemcpyHostToDevice, stream);
      
      SizeUpdate *su = new SizeUpdate(bufferData, nElts);
      cudaLaunchHostFunc(stream, setSizeCallback, su);
    }
    
    
    //
    // @brief copy data to our buffer from another buffer,
    //   destroying any existing contents.  Do not involve
    //   the host in the copy.
    //
    // @param other pointer to other buffer to copy from
    // @param nElts number of elements to copy
    void copy(const Buffer<T> *other, size_t nElts)
    {
      assert(nElts <= bufferData->capacity);
      
      cudaMemcpy(bufferData->data, other->bufferData->data,
		 nElts * sizeof(T), cudaMemcpyDeviceToDevice);
      bufferData->size = nElts;
    } 
    
    //
    // @brief asynchronous version of copy() for streams
    //  (does not require special memory, but will not overlap
    //   with operations in other streams on the source/target devices)
    //
    // @param other pointer to other buffer to copy from
    // @param nElts number of elements to copy
    // @param stream stream in which to perform the copy
    //
    void copyAsync(const Buffer<T> *other, size_t nElts,
		   cudaStream_t stream = 0)
    {
      assert(nElts <= bufferData->capacity);
      
      cudaMemcpyAsync(bufferData->data, other->bufferData->data,
		      nElts * sizeof(T), cudaMemcpyDeviceToDevice,
		      stream);
      
      SizeUpdate *su = new SizeUpdate(bufferData, nElts);
      cudaLaunchHostFunc(stream, setSizeCallback, su);
    } 
    
    
    //
    // @brief empty out a buffer
    // To conform with get() and set(), which contain cudaMemcpy calls
    // that implicitly synchronize the default stream, this operation
    // synchronizes the default stream explicitly.
    //
    void clear()
    { 
      cudaStreamSynchronize(0);
      bufferData->size = 0; 
    }
    
    
    //
    // @brief empty out a buffer asynchronously, AFTER all prior
    // operations in the current stream have finished.
    //
    void clearAsync(cudaStream_t stream = 0)
    {
      cudaLaunchHostFunc(stream, clearCallback, bufferData);
    }
    
    //
    // @brief extract the managed data object from the buffer
    //
    BufferData<T> *getData() const { return bufferData; }
    
  private:
    
    // information about an asynchronous size update to the buffer
    struct SizeUpdate {
      BufferData<T> *bufferData;
      size_t newSize;
      
      SizeUpdate(BufferData<T> *ibufferData, size_t inewSize)
	: bufferData(ibufferData), newSize(inewSize)
      {}
    };
    
    BufferData<T> *bufferData;
    
    //
    // @brief copy contents fom source to a new destination buffer
    // entirely on device, without involving the host.  Because
    // this operation involves a cudaMalloc(), it might as well be
    // synchronous.
    //
    // @param dst -- target buffer
    // @param src -- source buffer
    //
    void copyContents(BufferData<T> &dst,
		      const BufferData<T> &src)
    {
      dst.capacity = src.capacity;
      
      cudaMalloc(&dst.data, dst.capacity * sizeof(T));
      
      cudaMemcpy(dst.data, src.data,
		 src.size * sizeof(T),
		 cudaMemcpyDeviceToDevice);
      
      dst.size = src.size;
    }

    
    //
    // @brief callback to execute an asynchronous clear of a buffer.
    // This operation executes on the host only after all previous
    // operations in the same stream on the device are finished.
    //
    static void clearCallback(void *userData)
    {
      BufferData<T> *bufferData = (BufferData<T> *) userData;
      bufferData->size = 0;
    }

    //
    // @brief callback to execute an asynchronous clear of a buffer.
    // This operation executes on the host only after all previous
    // operations in the same stream on the device are finished.
    //
    static void setSizeCallback(void *userData)
    {
      SizeUpdate *su = (SizeUpdate *) userData;
      
      su->bufferData->size = su->newSize;
      
      delete su;
    }
    
  };
}

#endif
