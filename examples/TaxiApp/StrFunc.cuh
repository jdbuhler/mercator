#ifndef STRFUNC_CUH
#define STRFUNC_CUH

//Skip Whitespaces
__host__ __device__ 
const char* skipwhite(const char *q);


__host__ __device__
double d_strtod(const char* str, char** end);

#endif
