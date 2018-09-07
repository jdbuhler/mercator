#ifndef TEST_DATATYPES
#define TEST_DATATYPES

#define MATCH_SCORE 1
#define MISMATCH_SCORE -3

#define STRING_BUFF 8

#define THRESH_UNGAPPED 3
 
#define WINDOW_SIZE 8

#include "support/Managed.cuh"

#include "./blastData.cuh"

class point {
public:
	int db, query;
	__host__ __device__ point() {
		db = 2;
		query = 40;
	}
	__host__ __device__ point(int x, int y) {
		db = x;
		query = y;
	}
};

class MyModuleData : public CudaManaged {     
  public:
    int factor;

  __host__ __device__ MyModuleData() { factor = 0; }
  __host__ __device__ MyModuleData(int ifact) { factor = ifact; }
};

class MyNodeData : public CudaManaged {     
  public:
    int factor;

  __host__ __device__ MyNodeData() { factor = 0; }
  __host__ __device__ MyNodeData(int ifact) { factor = ifact; }
};

class UserDataExt : public CudaManaged {     
  public:
    int x;

  __host__ __device__ UserDataExt() { x = -1; }
  __host__ __device__ UserDataExt(int ix) { x = ix; }
};

#endif
