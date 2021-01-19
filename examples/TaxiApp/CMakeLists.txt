#
# CMake build rules for MERCATOR app
#
# MERCATOR
# Copyright (C) 2021 Washington University in St. Louis; all rights reserved.

project(Taxi
	LANGUAGES CUDA CXX)

# flags for MERCATOR compiler
set (MERCATOR_FLAGS -q 48 -t 96)

set (CMAKE_CUDA_FLAGS_RELEASE
     "${CMAKE_CUDA_FLAGS_RELEASE} --maxrregcount=32")

add_mercator_app(TARGET Taxi
		 SPECFILE TaxiApp.mtr
		 SOURCES Taxi.cu StrFunc.cu)

add_mercator_executable(TARGET TaxiApp
			APPS Taxi
			SOURCES TaxiDriver.cu)
