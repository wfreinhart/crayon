#
# CMakeLists.txt
# CMake directives for compiling crayon
#
# Copyright (c) 2018 Wesley Reinhart.
# This file is part of the crayon project, released under the Modified BSD License.

cmake_minimum_required(VERSION 3.1)

project(crayon)
set(crayon_VERSION_MAJOR 0)
set(crayon_VERSION_MINOR 1)
set(crayon_VERSION_PATCH 0)

set(CMAKE_MODULE_PATH
	"${CMAKE_CURRENT_SOURCE_DIR}/CMake"
	"${CMAKE_MODULE_PATH}"
	)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -Wall")

if( ${APPLE} )
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -framework Python")
endif()

if( ${CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT} )
    set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR} CACHE PATH "Install path prefix, prepended onto install directories." FORCE)
    message("-- Default install prefix detected, will install to default location ${CMAKE_INSTALL_PREFIX}")
else()
    message("-- Will install to user-specified location ${CMAKE_INSTALL_PREFIX}")
endif()

find_package(PythonInterp REQUIRED)
find_package(PythonLibsNew REQUIRED)
find_package(Boost REQUIRED)

include_directories(${PYTHON_INCLUDE_DIR})
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/pybind11/include)
include_directories(${PROJECT_SOURCE_DIR}/eigen)
include_directories(${PROJECT_SOURCE_DIR}/libgraphlet/include)
include_directories(${PROJECT_SOURCE_DIR}/libgraphlet/libgraph/include)
include_directories(${PROJECT_SOURCE_DIR}/libgraphlet/src)
include_directories(${PROJECT_SOURCE_DIR}/voro++)

set(CRAYON_DEST "${CMAKE_INSTALL_PREFIX}/crayon")
SET(CMAKE_SKIP_BUILD_RPATH FALSE)
SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
SET(CMAKE_INSTALL_RPATH "${CRAYON_DEST}/lib")
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
LIST(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CRAYON_DEST}/lib" isSystemDir)
IF("${isSystemDir}" STREQUAL "-1")
   SET(CMAKE_INSTALL_RPATH "${CRAYON_DEST}/lib")
ENDIF("${isSystemDir}" STREQUAL "-1")

# build voro++ library
file(GLOB SRC_VOROPP "voro++/voro++/voro++.cc")
add_library(voropp SHARED ${SRC_VOROPP})

# build orca library
file(GLOB SRC_ORCA "libgraphlet/src/orca/Orca.cpp")
add_library(orca SHARED ${SRC_ORCA})

# build libgraphlet library without GPU support
file(GLOB SRC_GRAPHLET "libgraphlet/src/libgraphlet/*.cpp")
file(GLOB SRC_GPU "libgraphlet/src/libgraphlet/*GPU.cpp")
list(REMOVE_ITEM SRC_GRAPHLET ${SRC_GPU})
add_library(graphlet SHARED ${SRC_GRAPHLET})
target_link_libraries(graphlet
                      orca)

# build crayon library
file(GLOB SRC_CRAYON "src/*.cc")
add_library(crayon SHARED ${SRC_CRAYON})
target_link_libraries(crayon
                      voropp
                      orca
                      graphlet
)
set_target_properties(crayon PROPERTIES PREFIX "_")
set_target_properties(crayon PROPERTIES SUFFIX ".so")

# install C++ libraries
install(TARGETS voropp DESTINATION "${CRAYON_DEST}/lib")
install(TARGETS orca DESTINATION "${CRAYON_DEST}/lib")
install(TARGETS graphlet DESTINATION "${CRAYON_DEST}/lib")
install(TARGETS crayon DESTINATION ${CRAYON_DEST})
# install python module
file(GLOB SRC_PY "src/py/*.py")
install(FILES ${SRC_PY} DESTINATION ${CRAYON_DEST})

#==================================================
# testing suite
#--------------------------------------------------
enable_testing()
# generate list of tests
execute_process(COMMAND "${PYTHON_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/test/list_test.py"
                OUTPUT_VARIABLE STR_TESTS
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_STRIP_TRAILING_WHITESPACE)
# split the list
separate_arguments(TEST_LIST UNIX_COMMAND ${STR_TESTS})
# save for 'make test'
foreach(TEST ${TEST_LIST})
    add_test(${TEST} "${PYTHON_EXECUTABLE}" ${TEST})
endforeach(TEST)
