# MERCATOR
# Copyright (C) 2018 Washington University in St. Louis; all rights reserved.

# for VERSION
cmake_policy(SET CMP0048 NEW)

project(MERCATOR
        VERSION 0.9.1
        LANGUAGES CUDA CXX)

cmake_minimum_required(VERSION 3.8)

######################################################################
# INSTALL TARGETS
######################################################################

#set(CMAKE_INSTALL_PREFIX /project/cassini/mercator-${PROJECT_VERSION})
set(CMAKE_INSTALL_PREFIX /export/project/hondius/stimcheck/ymercator/mercator-${PROJECT_VERSION})

######################################################################
# PACKAGE DEPENDENCIES
######################################################################

# Point to the LLVM installation's path (required) 
set(LLVM_PATH /project/cassini/llvm)

######################################################################
# MAKE CONFIGURATION
######################################################################

#set(CMAKE_BUILD_TYPE debug)
set(CMAKE_BUILD_TYPE release)

#set(CMAKE_VERBOSE_MAKEFILE on)

#####################################################################

configure_file(runtime/version.h.in
               runtime/version.h)

add_subdirectory(compiler)

add_subdirectory(doc)

install(FILES     mercator-rules.txt DESTINATION .)

install(DIRECTORY runtime            DESTINATION .
        PATTERN   "*.h.in"           EXCLUDE
        PATTERN   "*~"               EXCLUDE)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/runtime/version.h  
        DESTINATION runtime)

