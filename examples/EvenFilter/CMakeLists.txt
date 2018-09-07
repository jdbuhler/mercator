#
# CMake build rules for MERCATOR app
#

project(SameTypePipe
	LANGUAGES CUDA CXX)

cmake_minimum_required(VERSION 3.8)

add_mercator_app(TARGET EvenFilter
		 SPECFILE EvenFilter.mtr
		 SOURCES EvenFilter.cu)

add_mercator_executable(TARGET EvenFilterTest
			APPS EvenFilter
			SOURCES EFDriver.cu)

