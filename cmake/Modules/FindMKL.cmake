# - Find Intel MKL
# Find the MKL libraries
#
# Options:
#
#   MKL_STA       :   use static linking
#   MKL_MLT       :   use multi-threading
#   MKL_SDL       :   Single Dynamic Library interface
#
# This module defines the following variables:
#
#   MKL_FOUND            : True if MKL_INCLUDE_DIR are found
#   MKL_INCLUDE_DIR      : where to find mkl.h, etc.
#   MKL_INCLUDE_DIRS     : set when MKL_INCLUDE_DIR found
#   MKL_LIBRARIES        : the library to link against.
#
# Adapted from https://github.com/s9xie/hed_release-deprecated/blob/master/CMakeScripts/FindMKL.cmake
# by Ying-Jer Kao



include(FindPackageHandleStandardArgs)
if (DEFINED ENV{MKLROOT})
    set(MKL_ROOT $ENV{MKLROOT} CACHE PATH "Folder contains MKL")
    STRING(REGEX REPLACE "(.*intel/).*" "\\1" INTEL_ROOT "$ENV{MKLROOT}" )

    #message("INTEL_ROOT_FOUND" ${INTEL_ROOT})
    set(INTEL_ROOT ${INTEL_ROOT} CACHE PATH "Folder contains intel libs")
else()
    set(INTEL_ROOT "/opt/intel" CACHE PATH "Folder contains intel libs")
    set(MKL_ROOT ${INTEL_ROOT}/mkl CACHE PATH "Folder contains MKL")
  endif()
#message("MKL_ROOT:" ${MKL_ROOT})
#message("INTEL_ROOT:" ${INTEL_ROOT})
#message("MKLROOT:" $ENV{MKLROOT})
# Find include dir
#message("MKL_STA:" ${MKL_STA})
#message("MKL_MLT:" ${MKL_MLT})
#message("MKL_SDL:" ${MKL_SDL})
#message("CMAKE_SYSTEM_PROCESSOR:" ${CMAKE_SYSTEM_PROCESSOR})
find_path(MKL_INCLUDE_DIR mkl.h
    PATHS ${MKL_ROOT}/include)

# Find include directory
if(WIN32)
    find_path(INTEL_INCLUDE_DIR omp.h
        PATHS ${INTEL_ROOT}/include)
    set(MKL_INCLUDE_DIR ${MKL_INCLUDE_DIR} ${INTEL_INCLUDE_DIR})
endif()

# Find libraries
#  message("architecture:x86_64 " ${MKL_LIB_PATH})
if( CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64" )
  find_path(MKL_LIB_PATH  libmkl_core.a PATHS ${MKL_ROOT}/lib/intel64 ${MKL_ROOT}/lib/64 ${MKL_ROOT}/lib)
else()
  find_path(MKL_LIB_PATH  libmkl_core.a PATHS ${MKL_ROOT}/lib/ia32 ${MKL_ROOT}/lib/32 )
endif()
# Handle suffix
set(_MKL_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES})


if(WIN32)
  if(MKL_STA)
    set(CMAKE_FIND_LIBRARY_SUFFIXES .lib)
  else()
    set(CMAKE_FIND_LIBRARY_SUFFIXES _dll.lib)
  endif()
else()
  if(MKL_STA)
    set(CMAKE_FIND_LIBRARY_SUFFIXES .a)
  else()
    if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
      set(CMAKE_FIND_LIBRARY_SUFFIXES .so)
    else()
      set(CMAKE_FIND_LIBRARY_SUFFIXES .dylib)
    endif()
  endif()
endif()
#else()
#  MESSAGE("FINDING LINUX XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
#  if(MKL_STA)
#    set(CMAKE_FIND_LIBRARY_SUFFIXES .a)
#  else()
#    set(CMAKE_FIND_LIBRARY_SUFFIXES .so)
#  endif()
#endif()

# MKL is composed of four layers: Interface, Threading, Computational and RTL

if(MKL_SDL)
    find_library(MKL_LIBRARY mkl_rt PATHS ${MKL_LIB_PATH})
    set(MKL_MINIMAL_LIBRARY ${MKL_LIBRARY})
else()
    ######################### Interface layer #######################
    if(WIN32)
        set(MKL_INTERFACE_LIBNAME mkl_intel_c)
    else()
      if( CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64" )
        set(MKL_INTERFACE_LIBNAME mkl_intel_lp64)
      else()
        set(MKL_INTERFACE_LIBNAME mkl_intel)
      endif()
    endif()

    find_library(MKL_INTERFACE_LIBRARY ${MKL_INTERFACE_LIBNAME}
        PATHS ${MKL_LIB_PATH})

    ######################## Threading layer ########################
    if (MKL_MLT)
        set(MKL_THREADING_LIBNAME mkl_intel_thread)
    else()
        set(MKL_THREADING_LIBNAME mkl_sequential)
    endif()

    find_library(MKL_THREADING_LIBRARY ${MKL_THREADING_LIBNAME}
        PATHS ${MKL_LIB_PATH})

    ####################### Computational layer #####################
    find_library(MKL_CORE_LIBRARY mkl_core
        PATHS ${MKL_LIB_PATH})
    find_library(MKL_FFT_LIBRARY mkl_cdft_core
        PATHS ${MKL_LIB_PATH})
    if( CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64" )
      find_library(MKL_SCALAPACK_LIBRARY mkl_scalapack_lp64
            PATHS ${MKL_LIB_PATH})
    else()
      find_library(MKL_SCALAPACK_LIBRARY mkl_scalapack_core
            PATHS ${MKL_LIB_PATH})
    endif()

    ############################ RTL layer ##########################
    if(WIN32)
        set(MKL_RTL_LIBNAME omp5md)
    else()
        set(MKL_RTL_LIBNAME iomp5)
    endif()
    set(CMAKE_FIND_LIBRARY_SUFFIXES .so) # Link openmp runtime dynamically
    if( CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64" )
      find_library(MKL_RTL_LIBRARY ${MKL_RTL_LIBNAME}
          PATHS ${INTEL_ROOT}/lib/intel64)
    else()
      find_library(MKL_RTL_LIBRARY ${MKL_RTL_LIBNAME}
          PATHS ${INTEL_ROOT}/lib/ia32)
    endif()

    set(MKL_LIBRARY ${MKL_INTERFACE_LIBRARY} ${MKL_THREADING_LIBRARY} ${MKL_CORE_LIBRARY} ${MKL_FFT_LIBRARY} ${MKL_SCALAPACK_LIBRARY} ${MKL_RTL_LIBRARY})
    set(MKL_MINIMAL_LIBRARY ${MKL_INTERFACE_LIBRARY} ${MKL_THREADING_LIBRARY} ${MKL_CORE_LIBRARY} ${MKL_RTL_LIBRARY})
endif()
#message("MKL_LIBRARY: ${MKL_LIBRARY}")
set(CMAKE_FIND_LIBRARY_SUFFIXES ${_MKL_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES})

find_package_handle_standard_args(MKL DEFAULT_MSG MKL_INCLUDE_DIR MKL_LIBRARY MKL_MINIMAL_LIBRARY)

if(MKL_FOUND)
    set(MKL_INCLUDE_DIRS ${MKL_INCLUDE_DIR})
    set(MKL_LIBRARIES ${MKL_LIBRARY})
    set(MKL_MINIMAL_LIBRARIES ${MKL_MINIMAL_LIBRARY})
endif()
