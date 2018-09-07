project(MERCATOR
        LANGUAGES CUDA CXX)

cmake_minimum_required(VERSION 3.8)

######################################################################
# INSTALL TARGETS
######################################################################

set(CMAKE_INSTALL_PREFIX /project/cassini/mercator-0.9.0)

######################################################################
# PACKAGE DEPENDENCIES
######################################################################

# Point to the LLVM installation's path (required) 
set(LLVM_PATH /project/cassini/llvm-6.0.1)

######################################################################
# MAKE CONFIGURATION
######################################################################

#set(CMAKE_BUILD_TYPE debug)
set(CMAKE_BUILD_TYPE release)

#set(CMAKE_VERBOSE_MAKEFILE on)

#####################################################################

add_subdirectory(compiler)

add_subdirectory(doc)

install(FILES     mercator-rules.txt DESTINATION .)
install(DIRECTORY runtime            DESTINATION .)
