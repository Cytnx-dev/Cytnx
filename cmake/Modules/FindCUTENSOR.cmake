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

# NVIDIA archives use CUDA-versioned directories on Unix and lib/x64 on
# Windows. PyPI wheels use a flat lib directory on Unix and retain lib/x64 on
# Windows, so search every supported layout explicitly.
set(CUTENSOR_LIBRARY_SUFFIXES
  "lib/${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR}"
  "lib/${CUDAToolkit_VERSION_MAJOR}"
  "lib"
)
if(WIN32)
  list(PREPEND CUTENSOR_LIBRARY_SUFFIXES "lib/x64")
endif()

find_path(
    CUTENSOR_INCLUDE_DIR
    NAMES "cutensor.h"
    PATHS ${CUTENSOR_ROOT}
    PATH_SUFFIXES "include"
    NO_DEFAULT_PATH
)

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
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CUTENSOR
  REQUIRED_VARS CUTENSOR_INCLUDE_DIR CUTENSOR_LIB CUTENSORMg_LIB
)

if(CUTENSOR_FOUND)
  set(CUTENSOR_INCLUDE_DIRS "${CUTENSOR_INCLUDE_DIR}")
  set(CUTENSOR_LIBRARIES "${CUTENSOR_LIB};${CUTENSORMg_LIB}")
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
