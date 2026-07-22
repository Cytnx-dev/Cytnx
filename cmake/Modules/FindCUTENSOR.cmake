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
#   CUTENSOR_ROOT              root of the cuTENSOR installation

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

message(STATUS " cudaver: ${CUDAToolkit_VERSION_MAJOR}")
# Search every cuTENSOR library layout: 2.x tarballs place the libraries
# directly under lib/, while older tarballs and apt use a per-CUDA-major subdir
# (lib/<cuda-major>, e.g. lib/11, lib/12, lib/13). The older minor-specific
# lib/10.2 and lib/11.0 special-cases were removed; the generic lib/<major>
# covers them (apt multiarch paths remain, issue #946). The CUDA-version floor
# is a separate policy decision (issue #962), enforced in CMakeLists.txt rather
# than gated here, so this finder stays policy-free.
#
# The per-major directory records which CUDA major the libraries were *built*
# for, which need not equal the host toolkit's major: a cuTENSOR built for CUDA
# 12 ships lib/12 and is usable from a CUDA 13 host. Searching only
# lib/${CUDAToolkit_VERSION_MAJOR} therefore misses valid installs, so glob for
# the per-major directories this install actually has and append them, highest
# first. lib/ and the host major stay at the front so an exact match still wins.
set(CUTENSOR_LIBRARY_SUFFIXES lib lib/${CUDAToolkit_VERSION_MAJOR})
file(GLOB _cutensor_major_dirs RELATIVE "${CUTENSOR_ROOT}" "${CUTENSOR_ROOT}/lib/[0-9]*")
list(SORT _cutensor_major_dirs COMPARE NATURAL ORDER DESCENDING)
list(APPEND CUTENSOR_LIBRARY_SUFFIXES ${_cutensor_major_dirs})
if(WIN32)
  list(PREPEND CUTENSOR_LIBRARY_SUFFIXES "lib/x64")
endif()
list(REMOVE_DUPLICATES CUTENSOR_LIBRARY_SUFFIXES)
message(STATUS "cuTENSOR library search suffixes: ${CUTENSOR_LIBRARY_SUFFIXES}")

find_path(
    CUTENSOR_INCLUDE_DIR
    NAMES "cutensor.h" "cutensor/types.h"
    PATHS ${CUTENSOR_ROOT}
    PATH_SUFFIXES "include"
    NO_DEFAULT_PATH
)
set(CUTENSOR_INCLUDE_DIRS "${CUTENSOR_INCLUDE_DIR}")

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
  # CUTENSOR_MINOR may be absent/unmatched; default it to 0 rather than leaving
  # CUTENSOR_VERSION malformed (e.g. "2."), which would break the VERSION_VAR
  # comparison in find_package_handle_standard_args.
  if(_cutensor_minor_line)
    string(REGEX REPLACE ".*CUTENSOR_MINOR[ \t]+([0-9]+).*" "\\1" CUTENSOR_VERSION_MINOR "${_cutensor_minor_line}")
  else()
    set(CUTENSOR_VERSION_MINOR "0")
  endif()
  set(CUTENSOR_VERSION "${CUTENSOR_VERSION_MAJOR}.${CUTENSOR_VERSION_MINOR}")
  message(STATUS "cuTENSOR version: ${CUTENSOR_VERSION} (from ${_cutensor_version_header})")
  if(CUTENSOR_VERSION_MAJOR LESS 2)
    message(FATAL_ERROR
      "cuTENSOR >= 2.0 is required, but found ${CUTENSOR_VERSION} in "
      "${CUTENSOR_ROOT}. Install cuTENSOR 2.x and point CUTENSOR_ROOT at it.")
  endif()
else()
  message(WARNING
    "Could not determine the cuTENSOR version from headers under "
    "${CUTENSOR_INCLUDE_DIRS}; Cytnx requires cuTENSOR >= 2.0.")
endif()

# set libs:
# Try the CUDA-major-versioned subdirectory first (the layout of NVIDIA's
# standalone cuTENSOR tarball releases, e.g. lib/12/libcutensor.so), then
# fall back to a flat lib/ directory: the cutensor-cuXX PyPI wheels ship
# libcutensor.so.N directly under lib/, with no CUDA-version subdirectory.
find_library(
    CUTENSOR_LIB
    NAMES "cutensor"
    PATHS ${CUTENSOR_ROOT}
    PATH_SUFFIXES ${CUTENSOR_LIBRARY_SUFFIXES}
    NO_DEFAULT_PATH
)
find_library(
    CUTENSORMg_LIB
    NAMES "cutensorMg"
    PATHS ${CUTENSOR_ROOT}
    PATH_SUFFIXES ${CUTENSOR_LIBRARY_SUFFIXES}
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

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CUTENSOR
  REQUIRED_VARS CUTENSOR_INCLUDE_DIR CUTENSOR_LIB
  VERSION_VAR CUTENSOR_VERSION
)

if(CUTENSOR_FOUND)
  set(CUTENSOR_INCLUDE_DIRS "${CUTENSOR_INCLUDE_DIR}")
  get_filename_component(CUTENSOR_LIBRARY_DIRS "${CUTENSOR_LIB}" DIRECTORY)
  if(NOT TARGET CUTENSOR::CUTENSOR)
    add_library(CUTENSOR::CUTENSOR INTERFACE IMPORTED GLOBAL)
    set_target_properties(CUTENSOR::CUTENSOR PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${CUTENSOR_INCLUDE_DIRS}"
      INTERFACE_LINK_LIBRARIES "${CUTENSOR_LIBRARIES}"
    )
  endif()
endif()

mark_as_advanced(CUTENSOR_INCLUDE_DIR CUTENSOR_LIB CUTENSORMg_LIB)
