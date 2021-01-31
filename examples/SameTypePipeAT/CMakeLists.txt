#
# CMake build rules for MERCATOR app
#
# MERCATOR
# Copyright (C) 2018 Washington University in St. Louis; all rights reserved.

project(SameTypePipe
	LANGUAGES CUDA CXX)

add_mercator_app(TARGET SameTypePipe_1to1map_AT
		 SPECFILE SameTypePipe_1to1map.mtr
		 SOURCES SameTypePipe_1to1map.cu)

add_mercator_executable(TARGET SameTypePipeAT
			APPS SameTypePipe_1to1map_AT
			SOURCES SameTypePipe_driver.cu validate.cu)

