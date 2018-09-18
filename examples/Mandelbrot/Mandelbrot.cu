//
// MANDELBROT.CU
// Generate a Mandelbrot fractal
//
// MERCATOR
// Copyright (C) 2018 Washington University in St. Louis; all rights reserved.
//

#include "Mandelbrot_dev.cuh"

const uint32_t MAXITER = 255; // max number of iterations


__device__
void Mandelbrot_dev::
MakePixel::run(const uint32_t& idx, InstTagT nodeIdx)
{
  auto params = getAppParams();
  
  // convert linear pixel index to X and Y coordinates
  uint32_t y = idx / params->nX;
  uint32_t x = idx % params->nX;

  // oompute complex plane coordinates of this pixel
  float r = params->rMin + (params->rMax - params->rMin)/params->nX * x;
  float c = params->cMin + (params->cMax - params->cMin)/params->nY * y;
  
  float q = (r - 0.25) * (r - 0.25) + c * c;
  if (q * (q + r - 0.25) < 0.25 * c * c ||  // point is in main cardioid
      (r + 1) * (r + 1) + c * c < 1/16.0)   // point is in period-2 bulb
    {
      params->image[idx] = MAXITER; // point will not diverge 
    }
  else // do iterative convergence test
    {
      Pixel p;
      p.idx = idx;
      p.r = r;
      p.c = c;
      
      p.iter = 0;
      p.rCurr = 0.0;
      p.cCurr = 0.0;
      
      push(p, nodeIdx);
    }
}

__device__
void Mandelbrot_dev::
IteratePixel::run(const Pixel& pixel, InstTagT nodeIdx)
{
  float rsq = pixel.rCurr * pixel.rCurr;
  float csq = pixel.cCurr * pixel.cCurr;
  
  // check if this pixel has diverged yet
  if (rsq + csq < 2*2 && pixel.iter < MAXITER)
    {
      Pixel p = pixel;
      
      p.iter++;
      
      // iterate and continue
      p.rCurr = rsq - csq + p.r;
      p.cCurr = 2 * pixel.rCurr * pixel.cCurr + p.c;
      
      push(p, nodeIdx, Out::repeat);
    }
  else
    {
      // pixel has diverged, or we are out of iterations
      push(pixel, nodeIdx, Out::accept);
    }
}


__device__
void Mandelbrot_dev::
WritePixel::run(const Pixel& pixel, InstTagT nodeIdx)
{
  auto params = getAppParams();
  
  params->image[pixel.idx] = pixel.iter;
  
  // nothing pushed
}

