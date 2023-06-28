cmake_minimum_required(VERSION 3.15)

# set the project name and version
project(BLASExt LANGUAGES C VERSION 1.0 DESCRIPTION "Some pseudo-BLAS function for matrix operation not supported in canonical BLAS")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "Host processor : ${CMAKE_HOST_SYSTEM_PROCESSOR}")
message(STATUS "Architecture: ${ARCHITECTURE}")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
#set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_C_FLAGS_DEBUG "-g -march=native")
set(CMAKE_C_FLAGS_RELEASE "-O2 -march=native -ffast-math -ftree-vectorize -funroll-loops")

#------------------
# Define target 
#------------------
add_library(BLASExt SHARED
    src/dvdvt.c
    src/zvdvh.c
)
set_target_properties(BLASExt PROPERTIES VERSION ${PROJECT_VERSION} PREFIX "" SUFFIX ".so")

#-----------------------------------------------
# Find supported instruction set
#-----------------------------------------------
set( CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/external" ${CMAKE_MODULE_PATH} )
#include(FindAVX2)
set(AVX2_FOUND TRUE)
if (AVX2_FOUND)
    add_compile_definitions(HAVE_AVX2)
    message(STATUS "Processor instructions set: AVX2")
endif()


#----------------------------------------
# Add benchmark
#----------------------------------------

if (BUILD_BENCHMARKS)
  include(FindLAPACK)
  message(STATUS "LAPACK lib: ${LAPACK_LIBRARIES}")

  add_subdirectory(bench)
endif()

