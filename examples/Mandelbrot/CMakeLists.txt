#
# CMake build rules for MERCATOR app
#

project(Mandelbrot
	LANGUAGES CUDA CXX)

cmake_minimum_required(VERSION 3.8)

add_mercator_app(TARGET Mandelbrot
		 SPECFILE Mandelbrot.mtr
		 SOURCES Mandelbrot.cu)

add_mercator_executable(TARGET MandelDriver
			APPS Mandelbrot
			SOURCES MandelDriver.cu)
