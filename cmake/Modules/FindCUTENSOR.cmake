# - Find the CuTensor library
#
# Usage:
#   find_package(CUTENSOR [REQUIRED] [QUIET] )
#
# It sets the following variables:
#   CUTENSOR_FOUND               ... true if cutensor is found on the system
#   CUTENSOR_LIBRARY_DIRS        ... full path to cutensor library
#   CUTENSOR_INCLUDE_DIRS        ... cutensor include directory
#   CUTENSOR_LIBRARIES           ... cutensor libraries
#
#   MAGMA_ROOT                this is required to set!
#

#If environment variable MAGMA_ROOT is specified, it has same effect as MAGMA_ROOT

if(NOT DEFINED ENV{CUTENSOR_ROOT} AND NOT DEFINED CUTENSOR_ROOT)
  message(FATAL_ERROR "CUTENSOR_ROOT not set!")
else()
  if(DEFINED ENV{CUTENSOR_ROOT})
    set(CUTENSOR_ROOT "$ENV{CUTENSOR_ROOT}")
  endif()
  message("-- Looking for cuTENSOR in ${CUTENSOR_ROOT}")
  if(NOT EXISTS ${CUTENSOR_ROOT})
    message(FATAL_ERROR "Cannot find CUTENSOR_ROOT")
  endif()
endif()

message(STATUS " cudaver: ${CUDAToolkit_VERSION_MAJOR}" )
# Cytnx requires CUDA >= 12 (enforced in CMakeLists.txt) and cuTENSOR >= 2.0.
# Search both library layouts: cuTENSOR 2.x tarballs place the libraries
# directly under lib/, while 1.x tarballs and apt use a per-CUDA subdir
# (lib/<cuda-major>, e.g. lib/12, lib/13). Listing both as find_library
# PATH_SUFFIXES lets the supported 2.x flat layout and the legacy versioned
# layout resolve for both CUDA 12 and 13. The older lib/10.2 and lib/11
# branches were removed as dead code; apt multiarch paths remain (issue #946).
if(CUDAToolkit_VERSION_MAJOR GREATER_EQUAL 12)
  set(CUTNLIB_DIR lib lib/${CUDAToolkit_VERSION_MAJOR})
else()
  message(FATAL_ERROR
    "cuTENSOR support requires CUDA >= 12, but CUDAToolkit_VERSION_MAJOR is "
    "'${CUDAToolkit_VERSION_MAJOR}'.")
endif()

set(CUTENSOR_INCLUDE_DIRS ${CUTENSOR_ROOT}/include)

# Require cuTENSOR >= 2.0. The version macros (CUTENSOR_MAJOR/MINOR/PATCH) live
# in cutensor.h (older releases) or cutensor/types.h (newer ones); read whichever
# defines them and fail early on the 1.x API, which Cytnx no longer supports.
set(_cutensor_version_header "")
foreach(_hdr "${CUTENSOR_INCLUDE_DIRS}/cutensor.h" "${CUTENSOR_INCLUDE_DIRS}/cutensor/types.h")
  if(EXISTS "${_hdr}")
    file(STRINGS "${_hdr}" _cutensor_major_line REGEX "^#define[ \t]+CUTENSOR_MAJOR[ \t]+[0-9]+")
    if(_cutensor_major_line)
      set(_cutensor_version_header "${_hdr}")
      break()
    endif()
  endif()
endforeach()

if(_cutensor_version_header)
  file(STRINGS "${_cutensor_version_header}" _cutensor_minor_line REGEX "^#define[ \t]+CUTENSOR_MINOR[ \t]+[0-9]+")
  string(REGEX REPLACE ".*CUTENSOR_MAJOR[ \t]+([0-9]+).*" "\\1" CUTENSOR_VERSION_MAJOR "${_cutensor_major_line}")
  string(REGEX REPLACE ".*CUTENSOR_MINOR[ \t]+([0-9]+).*" "\\1" CUTENSOR_VERSION_MINOR "${_cutensor_minor_line}")
  set(CUTENSOR_VERSION "${CUTENSOR_VERSION_MAJOR}.${CUTENSOR_VERSION_MINOR}")
  message(STATUS "cuTENSOR version: ${CUTENSOR_VERSION} (from ${_cutensor_version_header})")
  if(CUTENSOR_VERSION_MAJOR LESS 2)
    message(FATAL_ERROR
      "cuTENSOR >= 2.0 is required, but found ${CUTENSOR_VERSION} in "
      "${CUTENSOR_ROOT}. Install cuTENSOR 2.x and point CUTENSOR_ROOT at it.")
  endif()
  # CUDA 13 support was added in cuTENSOR 2.3.0 (NVIDIA cuTENSOR release notes).
  # cuTENSOR 2.0-2.2 are CUDA-12 builds: their major version passes the >= 2.0
  # check above, but they can fail at link/load time against a CUDA 13 toolkit,
  # so require >= 2.3 in that case.
  if(CUDAToolkit_VERSION_MAJOR GREATER_EQUAL 13 AND CUTENSOR_VERSION VERSION_LESS 2.3)
    message(FATAL_ERROR
      "CUDA ${CUDAToolkit_VERSION_MAJOR} requires cuTENSOR >= 2.3 (CUDA 13 support "
      "was added in cuTENSOR 2.3.0), but found ${CUTENSOR_VERSION} in "
      "${CUTENSOR_ROOT}. Install cuTENSOR >= 2.3 and point CUTENSOR_ROOT at it.")
  endif()
else()
  message(WARNING
    "Could not determine the cuTENSOR version from headers under "
    "${CUTENSOR_INCLUDE_DIRS}; Cytnx requires cuTENSOR >= 2.0.")
endif()

# set libs:
find_library(
    CUTENSOR_LIB
    NAMES "cutensor"
    PATHS ${CUTENSOR_ROOT}
    PATH_SUFFIXES ${CUTNLIB_DIR}
    NO_DEFAULT_PATH
)
find_library(
    CUTENSORMg_LIB
    NAMES "cutensorMg"
    PATHS ${CUTENSOR_ROOT}
    PATH_SUFFIXES ${CUTNLIB_DIR}
    NO_DEFAULT_PATH
)
message(STATUS "CUTENSOR_LIB: ${CUTENSOR_LIB}")
message(STATUS "CUTENSORMg_LIB: ${CUTENSORMg_LIB}")
# Report the directory the library was actually found in (flat lib/ or the
# versioned lib/<major>) rather than guessing a subdir, so callers and runtime
# guidance reference the real location.
if(CUTENSOR_LIB)
  get_filename_component(CUTENSOR_LIBRARY_DIRS "${CUTENSOR_LIB}" DIRECTORY)
endif()
set(CUTENSOR_LIBRARIES "")
if(CUTENSOR_LIB)
    list(APPEND CUTENSOR_LIBRARIES "${CUTENSOR_LIB}")
endif()
if(CUTENSORMg_LIB)
    list(APPEND CUTENSOR_LIBRARIES "${CUTENSORMg_LIB}")
endif()

# CUTENSOR_FOUND must reflect whether the core library was actually located:
# the main cutensor lib is mandatory, cutensorMg is optional. Setting it
# unconditionally would let a NOTFOUND silently pass the caller's REQUIRED check.
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CUTENSOR
  REQUIRED_VARS CUTENSOR_LIB CUTENSOR_INCLUDE_DIRS
  VERSION_VAR CUTENSOR_VERSION)
