#
# CMake build rules for MERCATOR app
#

project(SameTypePipe
	LANGUAGES CUDA CXX)

cmake_minimum_required(VERSION 3.8)

add_mercator_app(TARGET LoopFilter
		 SPECFILE LoopFilter.mtr
		 SOURCES LoopFilter.cu)

add_mercator_executable(TARGET LoopFilterTest
			APPS LoopFilter
			SOURCES LFDriver.cu)

