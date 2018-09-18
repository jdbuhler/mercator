#
# CMake build rules for MERCATOR app
#
# MERCATOR
# Copyright (C) 2018 Washington University in St. Louis; all rights reserved.

project(SameTypePipe
	LANGUAGES CUDA CXX)

cmake_minimum_required(VERSION 3.8)

add_mercator_app(TARGET ModFilter
		 SPECFILE ModFilter.mtr
		 SOURCES ModFilter.cu)

add_mercator_executable(TARGET ModFilterTest
			APPS ModFilter
			SOURCES MFDriver.cu)

add_mercator_executable(TARGET ModFilterRangeTest
			APPS ModFilter
			SOURCES MFRangeDriver.cu)

