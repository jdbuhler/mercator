# MERCATOR
# Copyright (C) 2018 Washington University in St. Louis; all rights reserved.


project(MERCATOR_EXAMPLES
	LANGUAGES CUDA CXX)

cmake_minimum_required(VERSION 3.18)

#set(CMAKE_BUILD_TYPE debug)
set(CMAKE_BUILD_TYPE release)

#set(CMAKE_VERBOSE_MAKEFILE on)

######################################################################
# PACKAGE DEPENDENCIES
######################################################################

set(MERCATOR_ROOT_DIR /project/cassini/mblast/mercator)

# Uncomment this variable to build MERCATOR's examples inside
# its development source tree.
set(BUILDING_IN_TREE true)

# flags for MERCATOR compiler
set (MERCATOR_FLAGS "-q 24")

#######################################################################
# CUDA COMPILER FLAGS
#######################################################################

#set (CMAKE_CUDA_ARCHITECTURES 52)
set (CMAKE_CUDA_ARCHITECTURES 61)
#set (CMAKE_CUDA_ARCHITECTURES 72)

set (CMAKE_CXX_FLAGS_DEBUG   
     "${CMAKE_CXX_FLAGS_DEBUG} -UNDEBUG -Wall")

set (CMAKE_CUDA_FLAGS_DEBUG   
     "${CMAKE_CUDA_FLAGS_DEBUG} -G -UNDEBUG")

set (CMAKE_CXX_FLAGS_RELEASE 
     "${CMAKE_CXX_FLAGS_RELEASE} -UNDEBUG")

set (CMAKE_CUDA_FLAGS_RELEASE
     "${CMAKE_CUDA_FLAGS_RELEASE} -lineinfo -UNDEBUG")

set(WARNFLAGS -Xcompiler -Wall -Xcudafe --diag_suppress=unsigned_compare_with_zero -Xptxas --suppress-stack-size-warning)

string(JOIN " " NVCCFLAGS ${WARNFLAGS})

string(APPEND CMAKE_CUDA_FLAGS ${NVCCFLAGS})

######################################################################

include(${MERCATOR_ROOT_DIR}/mercator-rules.txt)

if(${BUILDING_IN_TREE})
  # Override default path to MERCATOR compiler to allow build from
  # non-installed version
  set(MERCATOR_CC ${MERCATOR_ROOT_DIR}/build/compiler/mercator)
  include_directories(AFTER ${MERCATOR_ROOT_DIR}/build/runtime)
endif()

add_subdirectory(SameTypePipe)

add_subdirectory(SameTypePipeAT)

add_subdirectory(EvenFilter)

add_subdirectory(ModFilter)

#add_subdirectory(SplitFilter)

#add_subdirectory(LoopFilter)

#add_subdirectory(AsyncFilter)

#add_subdirectory(Mandelbrot)

add_subdirectory(TaxiApp)
