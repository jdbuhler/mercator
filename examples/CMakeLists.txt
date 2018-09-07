project(MERCATOR_EXAMPLES
	LANGUAGES CUDA CXX)

cmake_minimum_required(VERSION 3.8)

#set(CMAKE_BUILD_TYPE debug)
set(CMAKE_BUILD_TYPE release)

#set(CMAKE_VERBOSE_MAKEFILE on)

######################################################################
# PACKAGE DEPENDENCIES
######################################################################

set(MERCATOR_ROOT_DIR /project/cassini/mercator/mercator-head)

set(CUB_ROOT_DIR /project/cassini/cub)

# We are building using a non-installed version of MERCATOR
set(BUILDING_IN_TREE true)
set(MERCATOR_BUILD_DIR ${MERCATOR_ROOT_DIR}/build)

#######################################################################
# CUDA COMPILER FLAGS
#######################################################################

set (CMAKE_CXX_FLAGS_DEBUG   
     "${CMAKE_CXX_FLAGS_DEBUG} -O0 -UNDEBUG -Wall -DVERBOSE")

set (CMAKE_CXX_FLAGS_RELEASE 
     "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

set(ARCHFLAGS -arch compute_61 -code sm_61)
set(WARNFLAGS -Xcompiler -Wall -Xcudafe --diag_suppress=unsigned_compare_with_zero -Xptxas --suppress-stack-size-warning)

string(JOIN " " NVCCFLAGS ${ARCHFLAGS} ${WARNFLAGS} -lineinfo)

string(APPEND CMAKE_CUDA_FLAGS ${NVCCFLAGS})

######################################################################

include(../mercator-rules.txt)

if(${BUILDING_IN_TREE})
  # Override default path to MERCATOR compiler to allow build from
  # non-installed version
  set(MERCATOR_CC ${MERCATOR_BUILD_DIR}/compiler/mercator)
endif()

add_subdirectory(SameTypePipe)

add_subdirectory(EvenFilter)

add_subdirectory(ModFilter)

add_subdirectory(SplitFilter)

add_subdirectory(LoopFilter)

add_subdirectory(AsyncFilter)
