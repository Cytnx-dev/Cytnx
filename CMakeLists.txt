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
#    Uni10 is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with Uni10.  If not, see <http://www.gnu.org/licenses/>.
#  @endlicense
#  @brief Main specification file for CMake
#  @author Ying-Jer Kao
#  @date 2014-05-06
#  @since 0.1.0
###


######################################################################
### CMAKE VERSION
######################################################################
cmake_minimum_required(VERSION 3.1.3)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")
set(CMAKE_CXX_STANDARD 14)

######################################################################
### DISABLING IN-SOURCE BUILD
### (DO NOT MODIFY)
######################################################################

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
### COMMAND LINE OPTIONS
######################################################################

option(ICPC "Build with Intel Compilers" OFF)
option(CUDA_SUPPORT "Build using Nvidia CUDA for GPU library" OFF)
option(EXAMPLES "Build Example Codes" ON)
option(PYTHON "Build Python wrapper" OFF)
option(MKL "Build using MKL" OFF)
option(DOC "Build API docuemntation" OFF)
option(HDF5 "Build HDF5" OFF)
option(PACKAGES "Build Source/Binary Packages" ON)
option(API_TESTS "Build API tests" ON)
if (ICPC)
  message(STATUS, "Build with Intel Compilers")
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
  endif ()
endif ()

if (MKL)
  option(MKL_SDL "Link to a single MKL dynamic libary." ON)
  option(MKL_MLT "Use multi-threading libary. [Default]" ON)
  mark_as_advanced(MKL_SDL MKL_MLT)
endif()



######################################################################
### PROJECT
######################################################################

project(Cytnx)

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
if(WIN32)
  if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
        set(CMAKE_INSTALL_PREFIX "C:\\Program Files\\cytnx"
                CACHE PATH "Cytnx install prefix"
                FORCE
        )
  endif()
endif()
######################################################################
### CMAKE BUILD TYPE
######################################################################

if (${CMAKE_BUILD_TYPE} MATCHES Debug)
   set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -coverage -Wall")
else()
   #set(CMAKE_CXX_FLAGS "-O3 -Wall -fvisibility=hidden")
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall")
endif()

######################################################################
### Find BLAS and LAPACK
######################################################################
if( NOT (DEFINED BLAS_LIBRARIES AND DEFINED LAPACK_LIBRARIES))
  if (MKL)
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
  ADD_DEFINITIONS("-DUNI_CPU -DUNI_LAPACK")
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
GENCODE_FLAGS:= -arch=sm_$(SMS)
if(CUDA)
  find_package(CUDA REQUIRED)
  SET(CUDA_SEPARABLE_COMPILATION ON)
  list(APPEND CUDA_NVCC_FLAGS "-O3;-Wno-deprecated-gpu-targets")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}  "-DUNI_GPU")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}  "-arch=sm_35")
  if(WIN32)
    SET(CUDA_PROPAGATE_HOST_FLAGS ON)
  else()
    SET(CUDA_PROPAGATE_HOST_FLAGS OFF)
  endif()
  string(REPLACE "libcusparse" "libcusolver" CUDA_cusolver_LIBRARY ${CUDA_cusparse_LIBRARY})#Workaround
  #set(CUDA_VERBOSE_BUILD true)
  ADD_DEFINITIONS("-DUNI_GPU -DUNI_CUSOLVER")
  set(LNPACKAGE "cusolver")
  set(CALARCH "gpu")
else()
  ADD_DEFINITIONS("-DUNI_CPU -DUNI_LAPACK")
  set(LNPACKAGE "lapack")
  set(CALARCH "cpu")
endif()

### Set architecture specific variable
set(UNI10_ARCH "${LNPACKAGE}_${CALARCH}")

# ----------------------------------------
# Find OpenMP
find_package( OpenMP )
if ( OPENMP_FOUND )
   set( CMAKE_C_FLAGS   "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}" )
   set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}" )
endif()


######################################################################
### Find HDF5 Library and Include dirs
######################################################################
IF(BUILD_HDF5_SUPPORT)
 ADD_DEFINITIONS("-DHDF5")
 if( NOT (DEFINED HDF5_C_LIBRARIES AND DEFINED HDF5_CXX_LIBRARIES AND DEFINED HDF5_INCLUDE_DIRS))
   find_package(HDF5 REQUIRED COMPONENTS C CXX)
   set(HDF5_LIBs ${HDF5_C_LIBRARIES} ${HDF5_CXX_LIBRARIES})
   set(HDF5_INCLUDE_DIRS ${HDF5_INCLUDE_DIRS})
   find_package(ZLIB)
   set( HDF5_LIBs ${HDF5_LIBs} ${ZLIB_LIBRARIES})
  #  find_package(SZ)
  #  set( HDF5_LIBs ${HDF5_LIBs} ${SZLIB_LIBRARIES})
   #TODO: Need win32
 else()
   set(HDF5_LIBs ${HDF5_C_LIBRARIES} ${HDF5_CXX_LIBRARIES})
   set(HDF5_INCLUDE_DIRS ${HDF5_INCLUDE_DIRS})
   message( STATUS "HDF5 found: ${HDF5_LIBRARIES}")
 endif()
 include_directories(${HDF5_INCLUDE_DIRS})
ENDIF()
######################################################################
### FLAGS
######################################################################
if(UNIX )
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
endif()
if(MKL_MLT)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
endif()
if( UNIX OR MINGW OR CYGWIN)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 ")
elseif (WINDOWS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Qstd=c++11")
endif()

if (MKL_MLT)
  set(CMAKE_EXE_LINKER_FLAGS "-pthread")
endif()

######################################################################
### PATHS
######################################################################

#set the default path for built executables to the "bin" directory
if (UNIX OR MINGW OR CYGWIN)
  set(EXECUTABLE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
endif()

#set the default path for built libraries to the "lib" directory
#if (UNIX OR MINGW OR CYGWIN)
#  set(LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
#endif()

# path to include directory
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

######################################################################
### ADD SUBDIRECTORIES
######################################################################

add_subdirectory(src)
if (BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()
if (BUILD_API_TESTS)
if (NOT WIN32)
 add_subdirectory(gtest-1.7.0)
 add_subdirectory(uni10_api_test)
endif()
endif()
######################################################################
### ADD LIBRARY
######################################################################
set(uni10-objects
$<TARGET_OBJECTS:uni10-auxiliary>
$<TARGET_OBJECTS:uni10-env_info>
$<TARGET_OBJECTS:uni10-api>
$<TARGET_OBJECTS:uni10-${UNI10_ARCH}>
)
#IF( BUILD_ARPACK_SUPPORT )
#  set(uni10-objects ${uni10-objects}
#  $<TARGET_OBJECTS:uni10-arpack>
#  )
#ENDIF()
IF(BUILD_HDF5_SUPPORT)
 set(uni10-objects ${uni10-objects}
 $<TARGET_OBJECTS:uni10-hdf5>
 )
ENDIF()
IF (BUILD_CUDA_SUPPORT)
  CUDA_INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/include)
  CUDA_ADD_LIBRARY(uni10 SHARED
    ${uni10-objects}
    src/uni10_cusolver_gpu/tools_cusolver_gpu/cuda_kernel_funcs/uni10_exp_kernel_gpu.cu
    src/uni10_cusolver_gpu/tools_cusolver_gpu/cuda_kernel_funcs/uni10_kernel_gpu.cu
    src/uni10_cusolver_gpu/tools_cusolver_gpu/cuda_kernel_funcs/uni10_reshape_kernel_gpu.cu
    src/uni10_env_info/uni10_cusolver_gpu/uni10_memory_const_cusolver_gpu.cu
    )
  CUDA_ADD_LIBRARY(uni10-static STATIC
    ${uni10-objects}
    src/uni10_cusolver_gpu/tools_cusolver_gpu/cuda_kernel_funcs/uni10_exp_kernel_gpu.cu
    src/uni10_cusolver_gpu/tools_cusolver_gpu/cuda_kernel_funcs/uni10_kernel_gpu.cu
    src/uni10_cusolver_gpu/tools_cusolver_gpu/cuda_kernel_funcs/uni10_reshape_kernel_gpu.cu
    src/uni10_env_info/uni10_cusolver_gpu/uni10_memory_const_cusolver_gpu.cu
    )
#  cuda_add_library( uni10 SHARED ${uni10all})
#  cuda_add_library( uni10-static STATIC ${uni10all})
  SET_TARGET_PROPERTIES(uni10-static PROPERTIES OUTPUT_NAME "uni10")
  IF(APPLE)
    set(MACOSX_RPATH 1)
    SET_TARGET_PROPERTIES(uni10 PROPERTIES VERSION ${UNI10_VERSION} SOVERSION ${UNI10_VERSION_MAJOR} MACOSX_RPATH 1)
  else()
    SET_TARGET_PROPERTIES(uni10 PROPERTIES VERSION ${UNI10_VERSION} SOVERSION ${UNI10_VERSION_MAJOR})
  endif()
  target_link_libraries(uni10
                ${CUDA_CUDART_LIBRARY}
		${CUDA_CUBLAS_LIBRARIES}
                ${CUDA_cusolver_LIBRARY}
                ${LAPACK_LIBRARIES}
                ${OpenMP_CXX_LIBRARIES})
  target_link_libraries(uni10-static
                ${CUDA_CUDART_LIBRARY}
		${CUDA_CUBLAS_LIBRARIES}
                ${CUDA_cusolver_LIBRARY}
                ${LAPACK_LIBRARIES}
                ${OpenMP_CXX_LIBRARIES})
ELSE()
  IF ("${CMAKE_GENERATOR}" STREQUAL "Xcode")
# Add Dummy file to Xcode Project to build the library
    add_custom_command(
    PRE_BUILD
    OUTPUT ${CMAKE_BINARY_DIR}/src/dummy.cpp
    COMMAND touch ${CMAKE_BINARY_DIR}/src/dummy.cpp
   )
   add_library(uni10 SHARED ${uni10-objects} ${CMAKE_BINARY_DIR}/src/dummy.cpp)
   add_library(uni10-static STATIC ${uni10-objects} ${CMAKE_BINARY_DIR}/src/dummy.cpp)
ELSE()
   add_library(uni10 SHARED ${uni10-objects} )
   add_library(uni10-static STATIC ${uni10-objects})
  ENDIF()
  target_link_libraries(uni10 ${LAPACK_LIBRARIES})
  target_link_libraries(uni10-static ${LAPACK_LIBRARIES})
  SET_TARGET_PROPERTIES(uni10-static PROPERTIES OUTPUT_NAME "uni10")
  IF(APPLE)
    set(MACOSX_RPATH 1)
    SET_TARGET_PROPERTIES(uni10 PROPERTIES VERSION ${UNI10_VERSION} SOVERSION ${UNI10_VERSION_MAJOR} MACOSX_RPATH 1)
  ELSE()
    SET_TARGET_PROPERTIES(uni10 PROPERTIES VERSION ${UNI10_VERSION} SOVERSION ${UNI10_VERSION_MAJOR})
  ENDIF()
  target_link_libraries(uni10
                ${LAPACK_LIBRARIES}
                ${OpenMP_CXX_LIBRARIES})
  target_link_libraries(uni10-static
                ${LAPACK_LIBRARIES}
                ${OpenMP_CXX_LIBRARIES})
ENDIF()

IF(BUILD_HDF5_SUPPORT)
  target_link_libraries(uni10 ${HDF5_LIBs})
  target_link_libraries(uni10-static ${HDF5_LIBs})
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
    configure_file(Uni10_Doxyfile.in Uni10Doxyfile)
    add_custom_target(doc ALL
    ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Uni10Doxyfile
     WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
     COMMENT "Generating API documentation with Doxygen" VERBATIM
     )
  endif(DOXYGEN_FOUND)
ENDIF()

IF(BUILD_PYTHON_WRAPPER)
   FIND_PACKAGE(pybind11 REQUIRED)
   pybind11_add_module(pyuni10 pyuni10/pyuni10.cpp)
   target_link_libraries(pyuni10 PRIVATE uni10-static)
ENDIF()

######################################################################
### INSTALL
######################################################################
install(TARGETS uni10 uni10-static
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
### INSTALL LANCZOS/ARNOLDI HEADERS
######################################################################
install(DIRECTORY advanced/
	DESTINATION advanced
        COMPONENT headers
	FILES_MATCHING PATTERN 		"*.*"
)

######################################################################
### INSTALL EXAMPLES
######################################################################
if(BUILD_EXAMPLES)
  install(DIRECTORY examples/
	DESTINATION examples/
	COMPONENT examples
	PATTERN "CMakeLists*" EXCLUDE
  )
endif()
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
######################################################################
### INSTALL TESTS
######################################################################
if(NOT WIN32)
install(DIRECTORY uni10_api_test/
        DESTINATION uni10_api_test/
        COMPONENT test
	PATTERN "CMakeLists*" EXCLUDE
)
endif()



set(libname "${CMAKE_SHARED_LIBRARY_PREFIX}uni10.${UNI10_VERSION}${CMAKE_SHARED_LIBRARY_SUFFIX}")
IF(BUILD_CUDA_SUPPORT)
set(libname "${CMAKE_SHARED_LIBRARY_PREFIX}uni10gpu.${UNI10_VERSION}${CMAKE_SHARED_LIBRARY_SUFFIX}")
ENDIF()



######################################################################
# Information Messages
######################################################################
message(STATUS "")
message(STATUS "------------------------------------------------")
message(STATUS "  Uni10: The Universal Tensor Network Library")
message(STATUS "------------------------------------------------")
message(STATUS "")
message(STATUS " Version: ${UNI10_VERSION}")
message(STATUS " Generator: ${CMAKE_GENERATOR}")
message(STATUS " Build Target: ${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS " Installation Prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS " CXX Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS " CXX Flags: ${CMAKE_CXX_FLAGS}")
message(STATUS " BLAS and LAPACK Libraries: ${LAPACK_LIBRARIES}")

#if(BUILD_ARPACK_SUPPORT)
#  message(STATUS " Build APPACK Support: YES")
#  message(STATUS "  - ARPACK Libraries: ${ARPACK_LIBRARIES}")
#else()
#  message(STATUS " Build APPACK Support: NO")
#endif()

if(BUILD_HDF5_SUPPORT)
 message(STATUS " Build HDF5 Support: YES")
 message(STATUS "  - HDF5 Libraries: ${HDF5_LIBs}")
 message(STATUS "  - HDF5 Includes: ${HDF5_INCLUDE_DIRS}")
else()
 message(STATUS " Build HDF5 Support: NO")
endif()

if(BUILD_EXAMPLES)
  message(STATUS " Build Examples: YES")
else()
  message(STATUS " Build Examples: NO")
endif()

if(BUILD_PYTHON_WRAPPER)
  message(STATUS " Build Python Wrapper: YES")
  message(STATUS "  - Python Excutable  : ${PYTHON_EXECUTABLE}")
  message(STATUS "  - Python Headers    : ${PYTHON_INCLUDE_DIR}")
  message(STATUS "  - Python Library    : ${PYTHON_LIBRARY}")
else()
  message(STATUS " Build Python Wrapper: NO")
endif()
if(BUILD_CUDA_SUPPORT)
  message( STATUS " Build CUDA Support: YES")
  message( STATUS "  - CUDA Version: ${CUDA_VERSION_STRING}")
  message( STATUS "  - CUDA Toolkit Root: ${CUDA_TOOLKIT_ROOT_DIR}")
  message( STATUS "  - CuSolver library: ${CUDA_cusolver_LIBRARY}")
else()
  message( STATUS " Build CUDA Support: NO")
endif()
if(BUILD_DOC)
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
