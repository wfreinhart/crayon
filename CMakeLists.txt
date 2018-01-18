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

message("cmake_module_path: ${CMAKE_MODULE_PATH}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -Wall")

find_package(PythonInterp REQUIRED)
find_package(PythonLibsNew REQUIRED)
find_package(Boost REQUIRED)

include_directories(${PYTHON_INCLUDE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/pybind11/include)
include_directories(${PROJECT_SOURCE_DIR}/eigen)
include_directories(${PROJECT_SOURCE_DIR}/libgraphlet/include)
include_directories(${PROJECT_SOURCE_DIR}/libgraphlet/libgraph/include)
include_directories(${PROJECT_SOURCE_DIR}/libgraphlet/src)

# build orca library
file(GLOB SRC_ORCA "libgraphlet/src/orca/Orca.cpp")
add_library(orca SHARED ${SRC_ORCA})

# build libgraphlet library without GPU support
file(GLOB SRC_GRAPHLET "libgraphlet/src/libgraphlet/*.cpp")
file(GLOB SRC_GPU "libgraphlet/src/libgraphlet/*GPU.cpp")
list(REMOVE_ITEM SRC_GRAPHLET ${SRC_GPU})
add_library(graphlet SHARED ${SRC_GRAPHLET})

# build crayon library
file(GLOB SRC_CRAYON "src/*.cc")
add_library(crayon SHARED ${SRC_CRAYON})
target_link_libraries(crayon
                      orca
                      graphlet
)

set_target_properties(crayon PROPERTIES PREFIX "")
set_target_properties(crayon PROPERTIES SUFFIX ".so")

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
