# - Find the CuQuantum library
#
# Usage:
#   find_package(CUQUANTUM [REQUIRED] [QUIET] )
#
# It sets the following variables:
#   CUQUANTUM_FOUND               ... true if cuquantum is found on the system
#   CUQUANTUM_LIBRARY_DIRS        ... full path to cuquantum library
#   CUQUANTUM_INCLUDE_DIRS        ... cuquantum include directory
#   CUQUANTUM_LIBRARIES           ... cuquantum libraries
#
#   CUQUANTUM_ROOT                this is required to set!
#

#If environment variable CUQUANTUM_ROOT is specified, it has same effect as CUQUANTUM_ROOT

if(NOT DEFINED ENV{CUQUANTUM_ROOT} AND NOT DEFINED CUQUANTUM_ROOT)
  message(FATAL_ERROR "CUQUANTUM_ROOT not set!")
else()
  if(DEFINED ENV{CUQUANTUM_ROOT})
    set(CUQUANTUM_ROOT "$ENV{CUQUANTUM_ROOT}")
  endif()
  message("-- Looking for cuQuantum in ${CUQUANTUM_ROOT}")
  if(NOT EXISTS ${CUQUANTUM_ROOT})
    message(FATAL_ERROR "Cannot find CUQUANTUM_ROOT")
  endif()
endif()

message(STATUS " cudaver: ${CUDAToolkit_VERSION_MAJOR}" )
set(CUTNLIB_DIR "lib")

find_path(
    CUQUANTUM_INCLUDE_DIR
    NAMES "cutensornet.h" "custatevec.h"
    PATHS ${CUQUANTUM_ROOT}
    PATH_SUFFIXES "include"
    NO_DEFAULT_PATH
)
find_library(
    CUQUANTUM_TENSORNET_LIB
    NAMES "cutensornet"
    PATHS ${CUQUANTUM_ROOT}
    PATH_SUFFIXES ${CUTNLIB_DIR}
    NO_DEFAULT_PATH
)
find_library(
    CUQUANTUM_CUSTATEVEC_LIB
    NAMES "custatevec"
    PATHS ${CUQUANTUM_ROOT}
    PATH_SUFFIXES ${CUTNLIB_DIR}
    NO_DEFAULT_PATH
)
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CUQUANTUM
  REQUIRED_VARS CUQUANTUM_INCLUDE_DIR CUQUANTUM_TENSORNET_LIB CUQUANTUM_CUSTATEVEC_LIB
)

if(CUQUANTUM_FOUND)
  set(CUQUANTUM_INCLUDE_DIRS "${CUQUANTUM_INCLUDE_DIR}")
  set(CUQUANTUM_LIBRARIES "${CUQUANTUM_TENSORNET_LIB};${CUQUANTUM_CUSTATEVEC_LIB}")
  get_filename_component(CUQUANTUM_LIBRARY_DIRS "${CUQUANTUM_TENSORNET_LIB}" DIRECTORY)
  if(NOT TARGET CUQUANTUM::CUQUANTUM)
    add_library(CUQUANTUM::CUQUANTUM INTERFACE IMPORTED GLOBAL)
    set_target_properties(CUQUANTUM::CUQUANTUM PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${CUQUANTUM_INCLUDE_DIRS}"
      INTERFACE_LINK_LIBRARIES "${CUQUANTUM_LIBRARIES}"
    )
  endif()
endif()

mark_as_advanced(CUQUANTUM_INCLUDE_DIR CUQUANTUM_TENSORNET_LIB CUQUANTUM_CUSTATEVEC_LIB)
