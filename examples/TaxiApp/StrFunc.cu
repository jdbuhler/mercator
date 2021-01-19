#include "StrFunc.cuh"

/*
 * Modified copy of strtod implementation by Yasuhiro Matsumoto.
 * https://gist.github.com/mattn/1890186
 */

#define isdigit(c) ((c) >= '0' && (c) <= '9')

__host__ __device__
double d_strtod(const char *p, char **end)
{  
  const unsigned int MAXNUMSIZE = 17;
  static const double shift[] = 
    {
      1e-0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 
      1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13,
      1e-14, 1e-15, 1e-16 
    };
  
  while (!isdigit(*p))
    p++;
    
  bool isNegative = (*p == '-');
  if (isNegative)
    p++;
  
  const char *decimalPosn = nullptr;
  double d = 0.0;
  
  for (unsigned int j = 0; j < MAXNUMSIZE; j++, p++)
    {
      if (isdigit(*p))
	d = d * 10.0 + (double) (*p - '0');
      else if (*p == '.')
	decimalPosn = p;
      else
	break;
    }
  
  if (decimalPosn)
    d *= shift[p-1 - decimalPosn];
  
  *end = (char *) p;
  return (isNegative ? -d : d);
}
