#
# CMake build rules for MERCATOR app
#

project(SameTypePipe
	LANGUAGES CUDA CXX)

cmake_minimum_required(VERSION 3.8)

add_mercator_app(TARGET SameTypePipe_1to1map
		 SPECFILE SameTypePipe_1to1map.mtr
		 SOURCES SameTypePipe_1to1map.cu)

add_mercator_executable(TARGET SameTypePipe
			APPS SameTypePipe_1to1map
			SOURCES SameTypePipe_driver.cu validate.cu)

