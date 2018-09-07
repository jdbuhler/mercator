#ifndef BLAST_DATA
#define BLAST_DATA

#include "support/Managed.cuh"

typedef unsigned char Base;

class BlastData : public CudaManaged {
public:
  Base* query;
  Base* database;
  int* qHits;
  int* qHash;
  
  BlastData() 
  {}
  
  BlastData(int* qHi, int *qHa, int bsize, int csize, 
	    Base* q, Base* d, int qsize, int dsize) 
  {
    cudaMalloc((void**) &qHits, bsize * sizeof(int));
    cudaMemcpy(qHits, qHi, bsize * sizeof(int), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
		
    cudaMalloc((void**) &qHash, csize * sizeof(int));
    cudaMemcpy(qHash, qHa, csize * sizeof(int), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
		
    printf("SIZEOF Q (in bytes == bases): %d\r\n", qsize);
    printf("SIZEOF D (in bytes == bases/4): %d\r\n", dsize);
		
    cudaMalloc((void**) &query, qsize * sizeof(Base));
    cudaMemcpy(query, q, qsize * sizeof(Base), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
		
    cudaMalloc((void**) &database, dsize * sizeof(Base));
    cudaMemcpy(database, d, dsize * sizeof(Base), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
  }
  
  ~BlastData()
  {
    cudaDeviceSynchronize();
    cudaFree(qHits);
    cudaFree(qHash);
    cudaFree(query);
    cudaFree(database);
  }
};

#endif
