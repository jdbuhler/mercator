//
// MANDELDRIVER.CU
// Driver for Mandelbrot set renderer
//
// MERCATOR
// Copyright (C) 2018 Washington University in St. Louis; all rights reserved.
//

#include <iostream>
#include <fstream>
#include <string>

#include "Mandelbrot.cuh"

// image resolution
const uint32_t NX = 3840*4;
const uint32_t NY = 2160*4;

// image palette (from the UltraFractal program)
const unsigned int NCOLORS = 16;
const uint8_t palette[NCOLORS][3] = {
  {  66,  30,  15 },
  {  25,   7,  26 },
  {   9,   1,  47 },
  {   4,   4,  73 },
  {   0,   7, 100 },
  {  12,  44, 138 },
  {  24,  82, 177 },
  {  57, 125, 209 },
  { 134, 181, 229 },
  { 211, 236, 248 },
  { 241, 233, 191 },
  { 248, 201,  95 },
  { 255, 170,   0 },
  { 204, 128,   0 },
  { 153,  87,   0 },
  { 106,  52,   3 }
};

const uint8_t zero[3] = { 0, 0, 0 };

using namespace std;

int main(int argc, char *argv[])
{
  string filename;
  
  if (argc > 1)
    filename = argv[1];
  else
    filename = "fractal.ppm";
  
  uint8_t *image;
  
  cudaMalloc(&image, NX * NY * sizeof(uint8_t));
  
  Mercator::Range<uint32_t> range(0, NX * NY, 1);
  
  Mandelbrot mandelApp;
  
  mandelApp.src.setSource(range);
  
  auto appParams = mandelApp.getParams();
  
  appParams->image = image;
  appParams->nX = NX;
  appParams->nY = NY;
  
  appParams->rMin = -2.5;
  appParams->rMax =  1.0;

  appParams->cMin = -1.0;
  appParams->cMax =  1.0;
  
  mandelApp.run();

  uint8_t *myImage = new uint8_t [NX * NY];
  cudaMemcpy(myImage, image, NX * NY * sizeof(uint8_t), 
	     cudaMemcpyDeviceToHost);
  
  ofstream os(filename);
  
  os << "P6" << endl;
  os << NX << ' ' << NY << endl;
  os << 255 << endl;
  
  for (unsigned int j = 0; j < NX * NY; j++)
    {
      uint8_t v = myImage[j];
      if (v == 255)
	os.write((char *) zero, 3);
      else
	os.write((char *) palette[v % NCOLORS], 3);
    }
  
  return 0;
}
