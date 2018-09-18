//
// PIXEL.H
// Pixel class for Mandelbrot renderer
//
// MERCATOR
// Copyright (C) 2018 Washington University in St. Louis; all rights reserved.
//

#ifndef _PIXEL_H
#define _PIXEL_H

#include <cstdint>

struct Pixel {
  uint32_t idx;
  float    r, c;
  
  float    rCurr, cCurr;
  uint8_t  iter;
};

#endif
