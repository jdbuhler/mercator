#
# CMake build rules for MERCATOR app
#
# MERCATOR
# Copyright (C) 2018 Washington University in St. Louis; all rights reserved.

project(SameTypePipe
	LANGUAGES CUDA CXX)

add_mercator_app(TARGET InterruptSim
		 SPECFILE InterruptSim.mtr
		 SOURCES InterruptSim.cu)

add_mercator_executable(TARGET InterruptSimTest
			APPS InterruptSim
			SOURCES InterruptSimDriver.cu)

