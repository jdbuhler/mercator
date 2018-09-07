#
# CMake build rules for MERCATOR app
#

project(SameTypePipe
	LANGUAGES CUDA CXX)

cmake_minimum_required(VERSION 3.8)

add_mercator_app(TARGET AsyncFilter
		 SPECFILE AsyncFilter.mtr
		 SOURCES AsyncFilter.cu)

add_mercator_executable(TARGET AsyncFilterTest
			APPS AsyncFilter
			SOURCES AFDriver.cu)
