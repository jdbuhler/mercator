# MERCATOR
# Copyright (C) 2018 Washington University in St. Louis; all rights reserved.


project(MERCATOR_EXAMPLES
	LANGUAGES CUDA CXX)

cmake_minimum_required(VERSION 3.8)

#set(CMAKE_BUILD_TYPE debug)
set(CMAKE_BUILD_TYPE release)

#set(CMAKE_VERBOSE_MAKEFILE on)

######################################################################
# PACKAGE DEPENDENCIES
######################################################################

#set(MERCATOR_ROOT_DIR /project/cassini/m-node/mercator)
set(MERCATOR_ROOT_DIR /home/research/planot/mercator)

set(CUB_ROOT_DIR /project/cassini/cub)

# Uncomment this variable to build MERCATOR's examples inside
# its development source tree.
set(BUILDING_IN_TREE true)

#######################################################################
# CUDA COMPILER FLAGS
#######################################################################

set (CMAKE_CXX_FLAGS_DEBUG   
     "${CMAKE_CXX_FLAGS_DEBUG} -UNDEBUG -Wall")

set (CMAKE_CUDA_FLAGS_DEBUG   
     "${CMAKE_CUDA_FLAGS_DEBUG} -G -UNDEBUG")

set (CMAKE_CXX_FLAGS_RELEASE 
     "${CMAKE_CXX_FLAGS_RELEASE} -DNDEBUG")

set (CMAKE_CUDA_FLAGS_RELEASE
     "${CMAKE_CUDA_FLAGS_RELEASE} -lineinfo -DNDEBUG")

set(ARCHFLAGS -arch compute_61 -code sm_61)
set(WARNFLAGS -Xcompiler -Wall -Xcudafe --diag_suppress=unsigned_compare_with_zero -Xptxas --suppress-stack-size-warning)

string(JOIN " " NVCCFLAGS ${ARCHFLAGS} ${WARNFLAGS})

string(APPEND CMAKE_CUDA_FLAGS ${NVCCFLAGS})

######################################################################

include(${MERCATOR_ROOT_DIR}/mercator-rules.txt)

#env Q_SIZE=X H_SIZE=Y S_SIZE=Z cmake3 .
set(MERCATOR_FLAGS )
#set queue size
if(DEFINED ENV{Q_SIZE})
set(MERCATOR_FLAGS -Q$ENV{Q_SIZE})
endif()
#set heap size
if(DEFINED ENV{H_SIZE})
set(MERCATOR_FLAGS ${MERCATOR_FLAGS} -H$ENV{H_SIZE})
endif()
#set stack size
if(DEFINED ENV{S_SIZE})
set(MERCATOR_FLAGS ${MERCATOR_FLAGS} -S$ENV{S_SIZE})
endif()

if(${BUILDING_IN_TREE})
  # Override default path to MERCATOR compiler to allow build from
  # non-installed version
  set(MERCATOR_CC ${MERCATOR_ROOT_DIR}/compiler/mercator)
  include_directories(AFTER ${MERCATOR_ROOT_DIR}/runtime)
endif()

add_subdirectory(SameTypePipe)

add_subdirectory(EvenFilter)

add_subdirectory(ModFilter)

add_subdirectory(SplitFilter)

add_subdirectory(LoopFilter)

add_subdirectory(AsyncFilter)

add_subdirectory(Mandelbrot)
