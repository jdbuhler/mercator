#include "StrFunc.cuh"

/*
 * Modified copy of strtod implementation by Yasuhiro Matsumoto.
 * https://gist.github.com/mattn/1890186
 */

#define isdigit(c) ((c) >= '0' && (c) <= '9')
#define isspace(c) ((c) == ' ')

//Skip Whitespaces
__host__ __device__
const char* skipwhite(const char *q) 
{
  const char *p = q;
  while(isspace(*p))
    ++p;
  return p;
}

__host__ __device__
double d_strtod(const char* str, char** end) 
{
  double d = 0.0;
  int sign;
  int n = 0;
  const char *p, *a;
  
  a = p = str;
  p = skipwhite(p);
  
  /* decimal part */
  sign = 1;
  
  if(*p == '-') 
    {
      sign = -1;
      ++p;
    }
  else if (*p == '+')
    {
      ++p;
    }
  
  if(isdigit(*p)) 
    {
      d = (double)(*p++ - '0');
      //printf("d = %f\n", d);
      while(*p && isdigit(*p)) 
	{
	  d = d * 10.0 + (double)(*p - '0');
	  ++p;
	  ++n;
	  //printf("d = %f\n", d);
	}
      a = p;
    }
  else if(*p != '.')
    goto done;
  
  d *= sign;
  
  
  /* fraction part */
  if (*p == '.') 
    {
      double f = 0.0;
      double base = 0.1;
      ++p;
      
      //printf("%c\n", *p);
      if (isdigit(*p)) 
	{
	  while(*p && isdigit(*p)) 
	    {
	      f += base * (*p - '0');
	      base /= 10.0;
	      ++p;
	      ++n;
	      //printf("%c, %lf\n", *p, f);
	    }
	}
      d += f * sign;
      a = p;
    }
  
  /* exponential part */
  //Omitted because unused
  
 done:
  if(end)
    *end = (char*) a;
  
  return d;
}
