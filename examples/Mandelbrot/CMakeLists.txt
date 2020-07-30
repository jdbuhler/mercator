#
# CMake build rules for MERCATOR app
#
# MERCATOR
# Copyright (C) 2018 Washington University in St. Louis; all rights reserved.

project(Mandelbrot
	LANGUAGES CUDA CXX)

add_mercator_app(TARGET Mandelbrot
		 SPECFILE Mandelbrot.mtr
		 SOURCES Mandelbrot.cu)

add_mercator_executable(TARGET MandelDriver
			APPS Mandelbrot
			SOURCES MandelDriver.cu)
