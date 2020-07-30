#
# CMake build rules for MERCATOR app
#
# MERCATOR
# Copyright (C) 2018 Washington University in St. Louis; all rights reserved.

project(SameTypePipe
	LANGUAGES CUDA CXX)

add_mercator_app(TARGET LoopFilter
		 SPECFILE LoopFilter.mtr
		 SOURCES LoopFilter.cu)

add_mercator_executable(TARGET LoopFilterTest
			APPS LoopFilter
			SOURCES LFDriver.cu)

