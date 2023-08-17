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
if((${CUDAToolkit_VERSION_MAJOR} LESS_EQUAL 10))
  set(CUTNLIB_DIR "lib/10.2")
elseif((${CUDAToolkit_VERSION_MAJOR} GREATER_EQUAL 11) AND (${CUDAToolkit_VERSION_MAJOR} LESS 12) AND (${CUDAToolkit_VERSION_MINOR} LESS_EQUAL 0))
  set(CUTNLIB_DIR "lib/11.0")
elseif((${CUDAToolkit_VERSION_MAJOR} GREATER_EQUAL 11) AND (${CUDAToolkit_VERSION_MAJOR} LESS 12) AND (${CUDAToolkit_VERSION_MINOR} GREATER_EQUAL 1))
  set(CUTNLIB_DIR "lib/11")
elseif((${CUDAToolkit_VERSION_MAJOR} GREATER_EQUAL 12))
  set(CUTNLIB_DIR "lib/12")
endif()

set(CUTENSOR_LIBRARY_DIRS ${CUTENSOR_ROOT}/${CUTNLIB_DIR})
set(CUTENSOR_INCLUDE_DIRS ${CUTENSOR_ROOT}/include)

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
set(CUTENSOR_LIBRARIES "${CUTENSOR_LIB};${CUTENSORMg_LIB}")
message(STATUS "ok")
set(CUTENSOR_FOUND TRUE)
