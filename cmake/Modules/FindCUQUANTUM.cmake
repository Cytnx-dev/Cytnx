# - Find the CuQuantum library
#
# Usage:
#   find_package(CUQUANTUM [REQUIRED] [QUIET] )
#
# It sets the following variables:
#   CUQUANTUM_FOUND               ... true if cutensor is found on the system
#   CUQUANTUM_LIBRARY_DIRS        ... full path to cutensor library
#   CUQUANTUM_INCLUDE_DIRS        ... cutensor include directory
#   CUQUANTUM_LIBRARIES           ... cutensor libraries
#
#   MAGMA_ROOT                this is required to set!
#

#If environment variable MAGMA_ROOT is specified, it has same effect as MAGMA_ROOT

if(NOT DEFINED ENV{CUQUANTUM_ROOT} AND NOT DEFINED CUQUANTUM_ROOT)
  message(FATAL_ERROR "CUQUANTUM_ROOT not set!")
else()
  if(DEFINED ENV{CUQUANTUM_ROOT})
    set(CUQUANTUM_ROOT "$ENV{CUQUANTUM_ROOT}")
  endif()
  message("-- Looking for cuTENSOR in ${CUQUANTUM_ROOT}")
  if(NOT EXISTS ${CUQUANTUM_ROOT})
    message(FATAL_ERROR "Cannot find CUQUANTUM_ROOT")
  endif()
endif()

message(STATUS " cudaver: ${CUDAToolkit_VERSION_MAJOR}" )
set(CUTNLIB_DIR "lib")

set(CUQUANTUM_LIBRARY_DIRS ${CUQUANTUM_ROOT}/${CUTNLIB_DIR})
set(CUQUANTUM_INCLUDE_DIRS ${CUQUANTUM_ROOT}/include)

# set libs:
find_library(
    CUQUANTUM_LIB
    NAMES "cutensornet"
    PATHS ${CUQUANTUM_ROOT}
    PATH_SUFFIXES ${CUTNLIB_DIR}
    NO_DEFAULT_PATH
)
find_library(
    CUQUANTUMMg_LIB
    NAMES "custatevec"
    PATHS ${CUQUANTUM_ROOT}
    PATH_SUFFIXES ${CUTNLIB_DIR}
    NO_DEFAULT_PATH
)
set(CUQUANTUM_LIBRARIES "${CUQUANTUM_LIB};${CUQUANTUMMg_LIB}")
message(STATUS "ok")
set(CUQUANTUM_FOUND TRUE)
