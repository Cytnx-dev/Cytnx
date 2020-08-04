###
#  @file CMakeLists.txt
#  @license
#    Copyright (c) 2013-
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
#  @author Kai-Hsin Wu, Ying-Jer Kao,
#  @date 2020-01-08
#  @since 0.4.0
#  
###

######################################################################
### CMAKE VERSION
######################################################################
cmake_minimum_required(VERSION 3.11)

include(cmake/target_sources_local.cmake)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")
message(STATUS ${CMAKE_MODULE_PATH})
include(GNUInstallDirs)

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
# Generate pif code
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

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
set(CYTNX_VERSION_MINOR 5)
set(CYTNX_VERSION_PATCH 5)
set(CYTNX_VERSION COUNT 1)
set(CYTNX_VERSION
  ${CYTNX_VERSION_MAJOR}.${CYTNX_VERSION_MINOR}.${CYTNX_VERSION_PATCH}
)



######################################################################
### COMMAND LINE OPTIONS
######################################################################

option(USE_ICPC "Build Cytnx with Intel Compilers" OFF)
option(USE_OMP "Build Cytnx with OpenMP" ON)
option(USE_CUDA "Build using Nvidia CUDA for GPU library" OFF)
option(BUILD_PYTHON "Build Python wrapper" ON)
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
    target_compile_definitions(cytnx UNI_ICPC)
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
project(CYTNX VERSION ${CYTNX_VERSION} LANGUAGES CXX C )

add_library(cytnx STATIC)
set_property(TARGET cytnx PROPERTY C_VISIBILITY_PRESET hidden)
set_property(TARGET cytnx PROPERTY VISIBILITY_INLINES_HIDDEN ON)
target_include_directories(cytnx
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_include_directories(cytnx SYSTEM
    PUBLIC
				${Boost_INCLUDE_DIRS}
)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wformat=0 -Wno-c++11-narrowing -w")
target_compile_features(cytnx PUBLIC cxx_std_11)
#set_property(TARGET cytnx PROPERTY CXX_STANDARD 11)

add_subdirectory(src)



######################################################################
### Find BLAS and LAPACK
######################################################################
if( NOT (DEFINED BLAS_LIBRARIES AND DEFINED LAPACK_LIBRARIES))
  if (USE_MKL)
    set(BLA_VENDOR Intel10_64ilp)
    find_package( BLAS REQUIRED)
    find_package( LAPACK REQUIRED)
    message( STATUS "LAPACK found: ${LAPACK_LIBRARIES}" )
  else()
    #set(BLA_VENDOR OpenBLAS)
    find_package( BLAS REQUIRED)
    find_package( LAPACK REQUIRED)
    message( STATUS "LAPACK found: ${LAPACK_LIBRARIES}" )
  endif()
else()
  set(LAPACK_LIBRARIES  ${BLAS_LIBRARIES}  ${LAPACK_LIBRARIES})
  message( STATUS "LAPACK found: ${LAPACK_LIBRARIES}")
endif()

if (USE_MKL)
    target_compile_definitions(cytnx PRIVATE UNI_MKL)
    target_compile_definitions(cytnx PRIVATE MKL_ILP64)
endif()


find_package(Boost REQUIRED)
message(STATUS ${COMPILE_DEFINITIONS})
# ----------------------------------------
# Find OpenMP
if(USE_OMP)
    find_package( OpenMP )
    if ( OPENMP_FOUND )
      if(NOT TARGET OpenMP::OpenMP_CXX)
        find_package(Threads REQUIRED)
        add_library(OpenMP::OpenMP_CXX IMPORTED INTERFACE)
        set_property(TARGET OpenMP::OpenMP_CXX
                     PROPERTY INTERFACE_COMPILE_OPTIONS "$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CXX>>:${OpenMP_CXX_FLAGS}>$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=${OpenMP_CXX_FLAGS}>")
        # Only works if the same flag is passed to the linker; use CMake 3.9+ otherwise (Intel, AppleClang)
        set_property(TARGET OpenMP::OpenMP_CXX
                     PROPERTY INTERFACE_LINK_LIBRARIES "$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CXX>>:${OpenMP_CXX_FLAGS}>$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=${OpenMP_CXX_FLAGS}>" Threads::Threads)

      else()
        set_property(TARGET OpenMP::OpenMP_CXX
                     PROPERTY INTERFACE_COMPILE_OPTIONS "$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CXX>>:${OpenMP_CXX_FLAGS}>$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=${OpenMP_CXX_FLAGS}>")    
      endif()
      target_link_libraries(cytnx PRIVATE OpenMP::OpenMP_CXX)
      target_compile_definitions(cytnx PRIVATE UNI_OMP)
    endif()
endif()
target_link_libraries(cytnx PUBLIC Boost::boost ${LAPACK_LIBRARIES})


#####
#function(CUDA_CONVERT_FLAGS EXISTING_TARGET)
#    get_property(old_flags TARGET ${EXISTING_TARGET} PROPERTY INTERFACE_COMPILE_OPTIONS)
#    if(NOT "${old_flags}" STREQUAL "")
#        string(REPLACE ";" "," CUDA_flags "${old_flags}")
#        set_property(TARGET ${EXISTING_TARGET} PROPERTY INTERFACE_COMPILE_OPTIONS
#            "$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CXX>>:${old_flags}>$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=${CUDA_flags}>"
#            )
#    endif()
#endfunction()
#CUDA_CONVERT_FLAGS(cytnx)
####

######################################################################
### Find CUDA
######################################################################
### Reference for Separate Compilation and Linking of CUDA C++ Device Code
### https://devblogs.nvidia.com/parallelforall/separate-compilation-linking-cuda-device-code/

###  NVCC:= $(CUDA_PATH)/bin/nvcc -ccbin $(CC)
### NVFLAGS:= -m64 -g
### SMS ?= 30

if(USE_CUDA)
  enable_language(CUDA)
  find_package(CUDA REQUIRED)
  if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  endif()

  set_target_properties(cytnx PROPERTIES 
      CUDA_SEPARABLE_COMPILATION ON
                                   )

  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe=--display_error_number")
  #set(CMAKE_CUDA_FLAGS "-Xcompiler=-Wall -Xcompiler=-Wno-deprecated-gpu-targets -Xcudafe=--display_error_number")
  ##set(CMAKE_CUDA_FLAGS "-Xcompiler=-Wall -Wno-deprecated-gpu-targets -Xcudafe=--display_error_number")
##  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}  "-DUNI_GPU")
#  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}  "-arch=sm_50 \
#      -gencode=arch=compute_50,code=sm_50 \
#      -gencode=arch=compute_52,code=sm_52 \
#      -gencode=arch=compute_60,code=sm_60 \
#      -gencode=arch=compute_61,code=sm_61 \
#      -gencode=arch=compute_70,code=sm_70 \
#      -gencode=arch=compute_75,code=sm_75 \
#      -gencode=arch=compute_75,code=compute_75 ")
  target_compile_definitions(cytnx PRIVATE UNI_GPU)
  #target_compile_options(cytnx PRIVATE ${CMAKE_CUDA_FLAGS})
  #target_compile_options(cytnx PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:>)
  target_include_directories(cytnx PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  target_link_libraries(cytnx PRIVATE ${CUDA_cusolver_LIBRARY})
  target_link_libraries(cytnx PRIVATE -lcublas -lcurand)
endif()




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

IF(BUILD_PYTHON)
    find_package(PythonInterp REQUIRED)
    find_package(pybind11 REQUIRED)
    if( NOT PYBIND11_FOUND)
        include(FetchContent)
        FetchContent_Declare(
          pybind11_sources
          GIT_REPOSITORY https://github.com/pybind/pybind11.git
          GIT_TAG v2.4.3
          )

        FetchContent_GetProperties(pybind11_sources)

        if(NOT pybind11_sources_POPULATED)
          FetchContent_Populate(pybind11_sources)

          add_subdirectory(
            ${pybind11_sources_SOURCE_DIR}
            ${pybind11_sources_BINARY_DIR}
            )
        endif()
    endif()
    pybind11_add_module(pycytnx SHARED pybind/cytnx.cpp)
    target_link_libraries(pycytnx PRIVATE cytnx)
    set_target_properties(pycytnx PROPERTIES OUTPUT_NAME cytnx)
    #add_library(cytnx MODULE pybind/cytnx.cpp)
    #target_link_libraries(cytnx PRIVATE pybind11::module)
    #set_target_properties(cytnx PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
    #                                     SUFFIX "${PYTHON_MODULE_EXTENSION}")
ENDIF()

######################################################################
### INSTALL
######################################################################

set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/cytnx)
INSTALL(TARGETS cytnx EXPORT cytnx_targets
        LIBRARY
          DESTINATION ${CMAKE_INSTALL_LIBDIR}
          COMPONENT libraries
        ARCHIVE
          DESTINATION ${CMAKE_INSTALL_LIBDIR}
          COMPONENT libraries
        PUBLIC_HEADER
          DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
          COMPONENT Development
      )
if(BUILD_PYTHON)
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/cytnx
        DESTINATION ${CMAKE_INSTALL_PREFIX}
        )
    INSTALL(TARGETS pycytnx EXPORT pycytnx_targets
            LIBRARY
              DESTINATION ${CMAKE_INSTALL_PREFIX}/cytnx
              COMPONENT libraries
          )
endif()
install(EXPORT cytnx_targets
        FILE CytnxTargets.cmake
        NAMESPACE Cytnx::
        DESTINATION ${INSTALL_CONFIGDIR}
        )
install(DIRECTORY include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        COMPONENT headers
        FILES_MATCHING PATTERN 		"*.h*")


#Create a ConfigVersion.cmake file
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/CytnxConfigVersion.cmake
        VERSION ${CYTNX_VERSION}
        COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(${CMAKE_CURRENT_LIST_DIR}/cmake/CytnxConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/CytnxConfig.cmake
        INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
        )


install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/CytnxConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/CytnxConfigVersion.cmake
        DESTINATION ${INSTALL_CONFIGDIR}
        )

##############################################
## Exporting from the build tree

export(EXPORT cytnx_targets FILE ${CMAKE_CURRENT_BINARY_DIR}/CytnxTargets.cmake NAMESPACE Cytnx::)

export(PACKAGE Cytnx)


######################################################################
### INSTALL DOCUMENTATION
######################################################################
if(DOXYGEN_FOUND)
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/docs/
	DESTINATION docs
        COMPONENT documentation
)
endif()


set(libname "${CMAKE_SHARED_LIBRARY_PREFIX}cytnx.${CYTNX_VERSION}${CMAKE_SHARED_LIBRARY_SUFFIX}")





######################################################################
# Information Messages
######################################################################
message(STATUS "")
message(STATUS "------------------------------------------------------------------------")
message(STATUS "  Project Cytnx, A Cross-section of Python & C++,Tensor network library ")
message(STATUS "------------------------------------------------------------------------")
message(STATUS "")
message(STATUS " Version: ${CYTNX_VERSION}")
message(STATUS " Generator: ${CMAKE_GENERATOR}")
message(STATUS " Build Target: ${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS " Installation Prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS " CXX Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS " CXX Flags: ${CMAKE_CXX_FLAGS}")
message(STATUS " BLAS and LAPACK Libraries: ${LAPACK_LIBRARIES}")






if(BUILD_PYTHON)
  message(STATUS " Build Python Wrapper: YES")
  message(STATUS "  - Python Excutable  : ${PYTHON_EXECUTABLE}")
  message(STATUS "  - Python Headers    : ${PYTHON_INCLUDE_DIR}")
  message(STATUS "  - Python Library    : ${PYTHON_LIBRARY}")
else()
  message(STATUS " Build Python Wrapper: NO")
endif()
if(USE_CUDA)
  message( STATUS " Build CUDA Support: YES")
  message( STATUS "  - CUDA Version: ${CUDA_VERSION_STRING}")
  message( STATUS "  - CUDA Toolkit Root: ${CUDA_TOOLKIT_ROOT_DIR}")
  message( STATUS "  - CuSolver library: ${CUDA_cusolver_LIBRARY}")
else()
  message( STATUS " Build CUDA Support: NO")
endif()
if(USE_OMP)
    if(OPENMP_FOUND)
      message( STATUS " Build OMP Support: YES")
    else()
      message( STATUS " Build OMP Support: NO  (Not found)")
    endif()
else()
      message( STATUS " Build OMP Support: NO")
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
