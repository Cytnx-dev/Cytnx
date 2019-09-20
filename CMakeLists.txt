###
#  @file CMakeLists.txt
#  @license
#    Copyright (c) 2013-2017
#    National Taiwan University
#    National Tsing-Hua University
#
#    This file is part of Cytnx.
#
#    Cytnx is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    cytnx is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with cytnx.  If not, see <http://www.gnu.org/licenses/>.
#  @endlicense
#  @brief Main specification file for CMake
#  @author Ying-Jer Kao
#  @date 2014-05-06
#  @since 0.1.0
###


######################################################################
### CMAKE VERSION
######################################################################
cmake_minimum_required(VERSION 3.1)

include(cmake/target_sources_local.cmake)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")
set(CMAKE_CXX_STANDARD 11)

######################################################################
### DISABLING IN-SOURCE BUILD
### (DO NOT MODIFY)
######################################################################
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)
if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  message(FATAL_ERROR
    "
     In-source builds not allowed.\n
     Please\n
       1) make a build directory by:
            mkdir -p path/to/my/build/directory\n
       2) run CMake in that build directory:
            cd path/to/my/build/directory
            cmake path/to/my/src/directory

     You have to first remove the file \"CMakeCache.txt\" by:
         rm CMakeCache.txt
    "
  )
endif()



if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

string(TOLOWER "${CMAKE_BUILD_TYPE}" cmake_build_type_tolower)
if(    NOT cmake_build_type_tolower STREQUAL "debug"
   AND NOT cmake_build_type_tolower STREQUAL "release"
   AND NOT cmake_build_type_tolower STREQUAL "relwithdebinfo")
  message(FATAL_ERROR "Unknown build type \"${CMAKE_BUILD_TYPE}\". Allowed values are Debug, Release, RelWithDebInfo (case-insensitive).")
endif()
######################################################################
# Version information
######################################################################
set(CYTNX_VERSION_MAJOR 0)
set(CYTNX_VERSION_MINOR 7)
set(CYTNX_VERSION_PATCH 0)
set(CYTNX_VERSION COUNT 1)
set(CYTNX_VERSION
  ${CYTNX_VERSION_MAJOR}.${CYTNX_VERSION_MINOR}.${CYTNX_VERSION_PATCH}
)



######################################################################
### COMMAND LINE OPTIONS
######################################################################

option(USE_ICPC "Build Cytnx with Intel Compilers" OFF)
option(USE_CUDA "Build using Nvidia CUDA for GPU library" OFF)
option(BUILD_PYTHON "Build Python wrapper" OFF)
option(USE_MKL "Build using MKL" OFF)
option(BUILD_DOC "Build API docuemntation" OFF)
if (USE_ICPC)
  message(STATUS, "Build Cytnx with Intel Compilers")
  message(STATUS, "Finding Intel Compilers....")
  find_program(CMAKE_C_COMPILER NAMES icc)
  find_program(CMAKE_CXX_COMPILER NAMES icpc)
  find_program(CMAKE_AR NAMES xiar)
  find_program(CMAKE_LINKER NAMES xild)

  if (NOT (CMAKE_C_COMPILER AND
           CMAKE_CXX_COMPILER AND
           CMAKE_AR AND
           CMAKE_LINKER ))
    message("Cannot find Intel compilers.")
    message("You may need to run `. /opt/intel/bin/compilervars.sh intel64/ia32'")
    message("Falling back to system compilers.")
    find_program(CMAKE_C_COMPILER NAMES cc)
    find_program(CMAKE_CXX_COMPILER NAMES cpp)
    find_program(CMAKE_AR NAMES ar)
    find_program(CMAKE_LINKER NAMES ld)
  else()
    ADD_DEFINITIONS("-DUNI_ICPC")
  endif ()
endif ()

if (USE_MKL)
  option(MKL_SDL "Link to a single MKL dynamic libary." ON)
  option(MKL_MLT "Use multi-threading libary. [Default]" ON)
  mark_as_advanced(MKL_SDL MKL_MLT)
endif()



######################################################################
### PROJECT
######################################################################
project(CYTNX LANGUAGES C++ C)
######################################################################
### CMAKE INSTALL OPTIONS
######################################################################
if(UNIX)
  if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
	set(CMAKE_INSTALL_PREFIX "/usr/local/cytnx"
		CACHE PATH "Cytnx install prefix"
   		FORCE
	)
  endif()
endif()

######################################################################
### CMAKE BUILD TYPE
######################################################################

if (${CMAKE_BUILD_TYPE} MATCHES Debug)
   set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -coverage -Wall -DUNI_DEBUG  ")
else()
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall")
endif()

######################################################################
### Find BLAS and LAPACK
######################################################################
if( NOT (DEFINED BLAS_LIBRARIES AND DEFINED LAPACK_LIBRARIES))
  if (USE_MKL)
    find_package(MKL REQUIRED)
    ADD_DEFINITIONS("-DUNI_MKL")
    set(LAPACK_LIBRARIES ${MKL_MINIMAL_LIBRARY})
    message("MKL found: ${MKL_MINIMAL_LIBRARY}")
  else()
    find_package( LAPACK REQUIRED)
    message( STATUS "LAPACK found: ${LAPACK_LIBRARIES}" )
  endif()
else()
  set(LAPACK_LIBRARIES  ${BLAS_LIBRARIES}  ${LAPACK_LIBRARIES})
  message( STATUS "LAPACK found: ${LAPACK_LIBRARIES}")
endif()


message(STATUS ${COMPILE_DEFINITIONS})
######################################################################
### Find CUDA
######################################################################
### Reference for Separate Compilation and Linking of CUDA C++ Device Code
### https://devblogs.nvidia.com/parallelforall/separate-compilation-linking-cuda-device-code/

###  NVCC:= $(CUDA_PATH)/bin/nvcc -ccbin $(CC)
### NVFLAGS:= -m64 -g
### SMS ?= 30

if(USE_CUDA)
  find_package(CUDA REQUIRED)
  set(CUDA_SEPARABLE_COMPILATION ON)
  list(APPEND CUDA_NVCC_FLAGS "-O3;-Wno-deprecated-gpu-targets;-m64")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}  "-DUNI_GPU")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}  "-arch=sm_50 \
      -gencode=arch=compute_50,code=sm_50 \
      -gencode=arch=compute_52,code=sm_52 \
      -gencode=arch=compute_60,code=sm_60 \
      -gencode=arch=compute_61,code=sm_61 \
      -gencode=arch=compute_70,code=sm_70 \
      -gencode=arch=compute_75,code=sm_75 \
      -gencode=arch=compute_75,code=compute_75 ")
  add_definitions("-DUNI_GPU")
endif()


# ----------------------------------------
# Find OpenMP
find_package( OpenMP )
if ( OPENMP_FOUND )
   set( CMAKE_C_FLAGS   "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}" )
   set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}" )
endif()




######################################################################
### PATHS
######################################################################

# path to include directory
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

######################################################################
### ADD SUBDIRECTORIES
######################################################################

add_subdirectory(src)


######################################################################
### ADD LIBRARY
######################################################################


add_library(cytnx "")

IF (BUILD_CUDA_SUPPORT)
  CUDA_INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/include)
  CUDA_ADD_LIBRARY(cytnx SHARED

    )
  CUDA_ADD_LIBRARY(cytnx-static STATIC
    ${cytnx-objects}
    src/cytnx_cusolver_gpu/tools_cusolver_gpu/cuda_kernel_funcs/cytnx_exp_kernel_gpu.cu
    src/cytnx_cusolver_gpu/tools_cusolver_gpu/cuda_kernel_funcs/cytnx_kernel_gpu.cu
    src/cytnx_cusolver_gpu/tools_cusolver_gpu/cuda_kernel_funcs/cytnx_reshape_kernel_gpu.cu
    src/cytnx_env_info/cytnx_cusolver_gpu/cytnx_memory_const_cusolver_gpu.cu
    )

  SET_TARGET_PROPERTIES(cytnx-static PROPERTIES OUTPUT_NAME "cytnx")
  if(APPLE)
    set(MACOSX_RPATH 1)
    SET_TARGET_PROPERTIES(cytnx PROPERTIES VERSION ${CYTNX_VERSION} SOVERSION ${CYTNX_VERSION_MAJOR} MACOSX_RPATH 1)
  else()
    SET_TARGET_PROPERTIES(cytnx PROPERTIES VERSION ${CYTNX_VERSION} SOVERSION ${CYTNX_VERSION_MAJOR})
  endif()
  target_link_libraries(cytnx
                ${CUDA_CUDART_LIBRARY}
		${CUDA_CUBLAS_LIBRARIES}
                ${CUDA_cusolver_LIBRARY}
                ${LAPACK_LIBRARIES}
                ${OpenMP_CXX_LIBRARIES})
  target_link_libraries(cytnx-static
                ${CUDA_CUDART_LIBRARY}
		${CUDA_CUBLAS_LIBRARIES}
                ${CUDA_cusolver_LIBRARY}
                ${LAPACK_LIBRARIES}
                ${OpenMP_CXX_LIBRARIES})

   add_library(cytnx SHARED ${cytnx-objects} )
   add_library(cytnx-static STATIC ${cytnx-objects})

  target_link_libraries(cytnx ${LAPACK_LIBRARIES})
  target_link_libraries(cytnx-static ${LAPACK_LIBRARIES})
  SET_TARGET_PROPERTIES(cytnx-static PROPERTIES OUTPUT_NAME "cytnx")
  IF(APPLE)
    set(MACOSX_RPATH 1)
    SET_TARGET_PROPERTIES(cytnx PROPERTIES VERSION ${CYTNX_VERSION} SOVERSION ${CYTNX_VERSION_MAJOR} MACOSX_RPATH 1)
  ELSE()
    SET_TARGET_PROPERTIES(cytnx PROPERTIES VERSION ${CYTNX_VERSION} SOVERSION ${CYTNX_VERSION_MAJOR})
  ENDIF()
  target_link_libraries(cytnx
                ${LAPACK_LIBRARIES}
                ${OpenMP_CXX_LIBRARIES})
  target_link_libraries(cytnx-static
                ${LAPACK_LIBRARIES}
                ${OpenMP_CXX_LIBRARIES})
ENDIF()

IF(BUILD_HDF5_SUPPORT)
  target_link_libraries(cytnx ${HDF5_LIBs})
  target_link_libraries(cytnx-static ${HDF5_LIBs})
ENDIF()
######################################################################
### RPATH SETTINGS
######################################################################
SET(CMAKE_SKIP_BUILD_RPATH  FALSE)
SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
LIST(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
IF("${isSystemDir}" STREQUAL "-1")
   SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
ENDIF("${isSystemDir}" STREQUAL "-1")

######################################################################
### Build Documents
######################################################################
IF(BUILD_DOC)
  FIND_PACKAGE(Doxygen REQUIRED)
  if(DOXYGEN_FOUND)
    configure_file(cytnx_Doxyfile.in cytnxDoxyfile)
    add_custom_target(doc ALL
    ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/cytnxDoxyfile
     WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
     COMMENT "Generating API documentation with Doxygen" VERBATIM
     )
  endif(DOXYGEN_FOUND)
ENDIF()

IF(BUILD_PYTHON_WRAPPER)
   FIND_PACKAGE(pybind11 REQUIRED)
   pybind11_add_module(pycytnx pycytnx/pycytnx.cpp)
   target_link_libraries(pycytnx PRIVATE cytnx-static)
ENDIF()

######################################################################
### INSTALL
######################################################################
install(TARGETS cyntax cyntax-static
        LIBRARY
        DESTINATION lib
        COMPONENT libraries
        ARCHIVE
        DESTINATION lib
        COMPONENT libraries)

######################################################################
### INSTALL HEADER FILES
######################################################################

install(DIRECTORY include/
	DESTINATION include
        COMPONENT headers
	FILES_MATCHING PATTERN 		"*.h*"
)




######################################################################
### INSTALL DOCUMENTATION
######################################################################
if(DOXYGEN_FOUND)
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/doc/
	DESTINATION doc
        COMPONENT documentation
)
endif()
install(FILES INSTALL API_Changes.md	ChangeLog.md	DevGuide.md	README.md GPL LGPL
        DESTINATION .
        COMPONENT common)





set(libname "${CMAKE_SHARED_LIBRARY_PREFIX}cytnx.${CYTNX_VERSION}${CMAKE_SHARED_LIBRARY_SUFFIX}")




######################################################################
# Information Messages
######################################################################
message(STATUS "")
message(STATUS "------------------------------------------------")
message(STATUS "  Cytnx: The Universal Tensor Network Library")
message(STATUS "------------------------------------------------")
message(STATUS "")
message(STATUS " Version: ${CYTNX_VERSION}")
message(STATUS " Generator: ${CMAKE_GENERATOR}")
message(STATUS " Build Target: ${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS " Installation Prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS " CXX Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS " CXX Flags: ${CMAKE_CXX_FLAGS}")
message(STATUS " BLAS and LAPACK Libraries: ${LAPACK_LIBRARIES}")






if(PYTHON)
  message(STATUS " Build Python Wrapper: YES")
  message(STATUS "  - Python Excutable  : ${PYTHON_EXECUTABLE}")
  message(STATUS "  - Python Headers    : ${PYTHON_INCLUDE_DIR}")
  message(STATUS "  - Python Library    : ${PYTHON_LIBRARY}")
else()
  message(STATUS " Build Python Wrapper: NO")
endif()
if(CUDA)
  message( STATUS " Build CUDA Support: YES")
  message( STATUS "  - CUDA Version: ${CUDA_VERSION_STRING}")
  message( STATUS "  - CUDA Toolkit Root: ${CUDA_TOOLKIT_ROOT_DIR}")
  message( STATUS "  - CuSolver library: ${CUDA_cusolver_LIBRARY}")
else()
  message( STATUS " Build CUDA Support: NO")
endif()
if(DOC)
  message( STATUS " Build Documentation: YES")
  message( STATUS "  - Doxygen: ${DOXYGEN_EXECUTABLE}")
else()
  message( STATUS " Build Documentation: NO")
endif()

message(STATUS "")
message(STATUS "")
message(STATUS "")

if (BUILD_PACKAGES)
include("Package.cmake")
include(CPack)
endif()
