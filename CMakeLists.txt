
message(STATUS "")
message(STATUS "------------------------------------------------------------------------")
message(STATUS "  Project Cytnx, A Cross-section of Python & C++,Tensor network library ")
message(STATUS "------------------------------------------------------------------------")
message(STATUS "")

# #####################################################################
# ## CMAKE and CXX VERSION
# #####################################################################
cmake_minimum_required(VERSION 3.20)

include(cmake/target_sources_local.cmake)
include(ExternalProject)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")
message(STATUS ${CMAKE_MODULE_PATH})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

message(STATUS " Generator: ${CMAKE_GENERATOR}")
message(STATUS " Build Target: ${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS " Installation Prefix: ${CMAKE_INSTALL_PREFIX}")

# #####################################################################
# ## DISABLING IN-SOURCE BUILD
# ## (DO NOT MODIFY)
# #####################################################################
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

# Generate pif code
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(USE_DEBUG)
  set(CMAKE_BUILD_TYPE "DEBUG")
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "RELEASE")
endif()

string(TOLOWER "${CMAKE_BUILD_TYPE}" cmake_build_type_tolower)

if(NOT cmake_build_type_tolower STREQUAL "debug"
  AND NOT cmake_build_type_tolower STREQUAL "release"
  AND NOT cmake_build_type_tolower STREQUAL "relwithdebinfo")
  message(FATAL_ERROR "Unknown build type \"${CMAKE_BUILD_TYPE}\". Allowed values are Debug, Release, RelWithDebInfo (case-insensitive).")
endif()

# #####################################################################
# Version information
# #####################################################################
include(version.cmake)
set(CYTNX_VERSION
  ${CYTNX_VERSION_MAJOR}.${CYTNX_VERSION_MINOR}.${CYTNX_VERSION_PATCH}
)
set(CYTNX_VARIANT_INFO "")

message(STATUS " Version: ${CYTNX_VERSION}")

# create a file that contain all the link flags:
FILE(WRITE "${CMAKE_BINARY_DIR}/linkflags.tmp" "" "")
FILE(WRITE "${CMAKE_BINARY_DIR}/cxxflags.tmp" "" "")

FILE(WRITE "${CMAKE_BINARY_DIR}/version.tmp" "" "")
FILE(APPEND "${CMAKE_BINARY_DIR}/version.tmp" "${CYTNX_VERSION}" "")

FILE(WRITE "${CMAKE_BINARY_DIR}/vinfo.tmp" "" "")
FILE(APPEND "${CMAKE_BINARY_DIR}/vinfo.tmp" "${CYTNX_VARIANT_INFO}" "")

# #####################################################################
# ## COMMAND LINE OPTIONS
# #####################################################################

# -----------------------------------------------------------------------------
# :Function: require_dependent_variable
#
# :Description:
#   Conditionally creates or validates a cache variable based on a set of
#   dependency conditions.
#
#   The dependency conditions are provided in the *depends* parameter as a
#   semicolon-separated list. The cache variable is required only when all
#   these conditions evaluate to true.
#
# :Parameters:
#   - **variable**: The name of the variable to set.
#   - **default**:  The default value for the variable.
#   - **type**:     The cache type (e.g., STRING, PATH) for the variable.
#   - **help**:     A descriptive help message for the variable.
#   - **depends**:  A semicolon-separated list of dependency conditions.
#                  The variable is required only if all conditions are met.
#
# :Behavior:
#   - Evaluates each dependency condition; if any condition evaluates to false,
#     will not update the variable.
#   - If all conditions are met and the variable is not already defined, the
#     variable is set to the provided default value in the cache using the
#     specified type and help message.
#   - If the variable’s value is empty, a fatal error is raised.
#
# :Examples:
#
#   .. code-block:: cmake
#
#      require_dependent_variable(
#        PACKAGE_INCLUDE_DIRS
#        "$ENV{PACKAGE_INCLUDE_DIRS}"
#        PATH
#        "Path to the include directory of the package."
#        "USE_A AND (USE_B OR USE_C);USE_D"
#      )
#
#   In this example, if "USE_A AND (USE_B OR USE_C);USE_D" is true, then
#   PACKAGE_INCLUDE_DIRS must be defined and non-empty. Otherwise, a fatal error
#   is raised.
# -----------------------------------------------------------------------------
function(require_dependent_variable variable default type help depends)
  # Evaluate each dependency condition; if any condition is false, exit early.
  # This is copied from the source code of `cmake_dependent_option` function.
  set(IS_NOT_AVAILABLE 0)
  foreach(d ${depends})
    cmake_language(EVAL CODE "
      if (${d})
      else()
        set(IS_NOT_AVAILABLE 1)
      endif()"
    )
  endforeach()
  if(IS_NOT_AVAILABLE)
    return()
  endif()

  if(NOT DEFINED ${variable})
    set(${variable} "${default}" CACHE ${type} "${help}")
  endif()

  if("${${variable}}" STREQUAL "")
    message(FATAL_ERROR "\
Error: Cache variable '${variable}' is required under dependency conditions \
(${depends}) but its value is empty.
${help}
")
  endif()
endfunction()

include(CMakeDependentOption)

set(CMAKE_INSTALL_PREFIX "/usr/local/cytnx" CACHE PATH "the destination path for installation")
option(BUILD_PYTHON "Build Python API. Require Python and pybind11 installed." ON)
option(BACKEND_TORCH "Use PyTorch as a backend container for tensors." OFF)
cmake_dependent_option(USE_MKL "Use MKL as a BLAS provider otherwise use OpenBLAS." OFF "NOT BACKEND_TORCH" OFF)
cmake_dependent_option(USE_HPTT "Use HPTT library to accelrate tensor transpose." OFF "NOT BACKEND_TORCH" OFF)
cmake_dependent_option(HPTT_ENABLE_FINE_TUNE "Enable fine tune HPTT for the native hardware." OFF "USE_HPTT" OFF)
# TODO: use variable instead of option to handle HPTT variant.
cmake_dependent_option(HPTT_ENABLE_ARM "HPTT variant" OFF "USE_HPTT;NOT HPTT_ENABLE_AVX; NOT HPTT_ENABLE_IBM" OFF)
cmake_dependent_option(HPTT_ENABLE_AVX "HPTT variant" OFF "USE_HPTT;NOT HPTT_ENABLE_ARM; NOT HPTT_ENABLE_IBM" OFF)
cmake_dependent_option(HPTT_ENABLE_IBM "HPTT variant" OFF "USE_HPTT;NOT HPTT_ENABLE_ARM; NOT HPTT_ENABLE_AVX" OFF)
cmake_dependent_option(USE_CUDA "Enable CUDA support." OFF "NOT BACKEND_TORCH" OFF)
cmake_dependent_option(USE_CUTT "Use CUTT library to accelrate tensor transpose." OFF "USE_CUDA" OFF)
cmake_dependent_option(CUTT_ENABLE_FINE_TUNE "Enable CUTT fine tune for the native hardware." OFF "USE_CUTT" OFF)
cmake_dependent_option(USE_CUTENSOR "Enable cuTENSOR." OFF "USE_CUDA" OFF)
require_dependent_variable(CUTENSOR_ROOT "$ENV{CUTENSOR_ROOT}" PATH
"Please provide CUTENSOR_ROOT in the environment variables.
CUTENSOR_ROOT is the path to the folder of cuTENSOR." "USE_CUTENSOR")
cmake_dependent_option(USE_CUQUANTUM "Enable cuQuantum." OFF "USE_CUDA" OFF)
require_dependent_variable(CUQUANTUM_ROOT "$ENV{CUQUANTUM_ROOT}" PATH
"Please provide CUQUANTUM_ROOT in the environment variables.
CUQUANTUM_ROOT is the path to the folder of cuQuantum." "USE_CUQUANTUM")
option(RUN_TESTS "Run Cytnx tests" OFF)
option(RUN_BENCHMARKS "Run Cytnx benchmarks" OFF)
option(USE_DEBUG "Build in debug more and enable address sanitizer" OFF)
option(BUILD_DOC "Build API docuemntation" OFF)
option(DEV_MODE "Build testing dev_test.cpp with cytnx" OFF)

# #####################################################################
# ## PROJECT
# #####################################################################
project(CYTNX VERSION ${CYTNX_VERSION} LANGUAGES CXX C)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(USE_CUDA)
  enable_language(CUDA)
  # Disable generation of "--option-file" flag in compile_commands.json.
  # This workaround helps VSCode's cpptools extension correctly locate CUDA
  # include files.
  # Refer to: https://discourse.cmake.org/t/cmake-target-include-directories-doesnt-export-to-compile-commands-json-when-using-cuda/10072/10
  set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES 0)
  set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES 0)
  set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS 0)
endif()

# C++ uses link-time optimization anyway; this enables additionally -flto=auto,
# for parallel compilation
# It cannot enable on MacOS since it causes building errors when linking the library libcytnx.a.
# Error message: Error running link command: no such file or directory make[2]: *** [CMakeFiles/cytnx.dir/build.make:3109: libcytnx.a]
IF (APPLE)
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION FALSE)
ELSE ()
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
ENDIF ()

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

target_compile_definitions(cytnx PUBLIC _LIBCPP_DISABLE_AVAILABILITY)
target_compile_definitions(cytnx PUBLIC _LIBCPP_ENABLE_CXX17_REMOVED_UNARY_BINARY_FUNCTION)
target_include_directories(cytnx SYSTEM
  PUBLIC
  ${Boost_INCLUDE_DIRS}
)

target_compile_options(cytnx PUBLIC -Wformat=0 -w -fsized-deallocation)
target_compile_features(cytnx PUBLIC cxx_std_17)

add_subdirectory(src)

if(USE_DEBUG)
  target_compile_definitions(cytnx PUBLIC UNI_DEBUG)

  # Remember to add export ASAN_OPTIONS=protect_shadow_gap=0 in Install.sh
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")
endif()

include(GNUInstallDirs)

if(USE_ICPC)
  target_compile_definitions(cytnx PUBLIC UNI_ICPC)
  # This option is to disable the warning of icpc
  target_compile_options(cytnx PUBLIC -diag-disable=10441)
  # This option is to maintain consistency of the floating point operation, also
  # try to improve the performance
  target_compile_options(cytnx PUBLIC -fp-model consistent -fimf-use-svml)
endif()

# #####################################################################

# ## Boost
find_package(Boost REQUIRED)
target_include_directories(cytnx SYSTEM
  PUBLIC
  ${Boost_INCLUDE_DIRS}
)
target_link_libraries(cytnx PUBLIC Boost::boost ${LAPACK_LIBRARIES})
FILE(APPEND "${CMAKE_BINARY_DIR}/cxxflags.tmp" "-I${Boost_INCLUDE_DIRS}\n" "")

# ###
if(BACKEND_TORCH)
  message(STATUS "backend = pytorch")
  target_compile_definitions(cytnx PUBLIC BACKEND_TORCH)

  # let torch python expose where pytorch.cmake is installed
  execute_process(
    COMMAND bash -c "python -c 'import torch;print(torch.utils.cmake_prefix_path)'"
    OUTPUT_VARIABLE TORCH_CMAKE_PATH_C
  )
  string(REGEX REPLACE "\n$" "" TORCH_CMAKE_PATH_C "${TORCH_CMAKE_PATH_C}")
  set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${TORCH_CMAKE_PATH_C})
  message(STATUS ${CMAKE_PREFIX_PATH})

  find_package(Torch REQUIRED)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
  message(STATUS "pytorch: ${TORCH_INSTALL_PREFIX}")
  message(STATUS "pytorch libs: ${TORCH_LIBRARIES}")

  if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    target_link_libraries(cytnx PUBLIC ${TORCH_LIBRARIES} ${TORCH_INSTALL_PREFIX}/lib/libtorch_python.dylib)
  else()
    target_link_libraries(cytnx PUBLIC ${TORCH_LIBRARIES} ${TORCH_INSTALL_PREFIX}/lib/libtorch_python.so)
    target_link_libraries(cytnx PRIVATE ${TORCH_LIBRARIES} ${TORCH_INSTALL_PREFIX}/lib/libtorch_python.so)
    target_link_libraries(cytnx INTERFACE ${TORCH_LIBRARIES} ${TORCH_INSTALL_PREFIX}/lib/libtorch_python.so)
  endif()

else()
  message(STATUS "backend = cytnx")
  include(CytnxBKNDCMakeLists.cmake)
endif() # Backend torch

# #####################################################################
# ## Get Gtest & benchmark
# #####################################################################
if(RUN_TESTS)
  # Include CTest explicitly to suppress the error that DartConfiguration.tcl
  # is not found
  include(CTest)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fprofile-arcs -ftest-coverage")
  target_link_libraries(cytnx PUBLIC "-lgcov --coverage")
  # "-fopenmp" flag is appended to CMAKE_CXX_FLAGS in CytnxBKNDCMakeLists.cmake.
  # To make the executalbe targets defined under tests/ link OpenMP correctly,
  # tests/ must be added after including CytnxBKNDCMakeLists.cmake.
  add_subdirectory(tests)
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
endif()

if(RUN_BENCHMARKS)
  # google benchmark
  find_package(benchmark REQUIRED)
  message(STATUS "benchmark found at: ${benchmark_DIR}")
  add_subdirectory(benchmarks)
endif()

# #######
# ## Python wrapper
# #######
IF(BUILD_PYTHON)
  find_package (Python COMPONENTS Interpreter Development)
  find_package(pybind11 REQUIRED)

  if(NOT pybind11_FOUND)
    include(FetchContent)
    FetchContent_Declare(
      pybind11_sources
      GIT_REPOSITORY https://github.com/pybind/pybind11.git
      GIT_TAG v2.11.1
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

  pybind11_add_module(pycytnx SHARED pybind/cytnx.cpp
    pybind/generator_py.cpp
    pybind/storage_py.cpp
    pybind/tensor_py.cpp
    pybind/symmetry_py.cpp
    pybind/bond_py.cpp
    pybind/network_py.cpp
    pybind/linop_py.cpp
    pybind/unitensor_py.cpp
    pybind/linalg_py.cpp
    pybind/algo_py.cpp
    pybind/physics_related_py.cpp
    pybind/random_py.cpp
    pybind/tnalgo_py.cpp
    pybind/scalar_py.cpp
    pybind/ncon_py.cpp
  )
  target_link_libraries(pycytnx PUBLIC cytnx)
  set_target_properties(pycytnx PROPERTIES OUTPUT_NAME cytnx)
  message(STATUS "pybind11 include dir: ${CUTENSOR_LIBRARY_DIRS}")
  message(STATUS "pybind11 include dir: ${CUQUANTUM_LIBRARY_DIRS}")
  set_target_properties(pycytnx
    PROPERTIES
    INSTALL_RPATH "${CUTENSOR_LIBRARY_DIRS};${CUQUANTUM_LIBRARY_DIRS}"
  )

  message(STATUS " Build Python Wrapper: YES")
  message(STATUS "  - Python Excutable  : ${PYTHON_EXECUTABLE}")
  message(STATUS "  - Python Headers    : ${PYTHON_INCLUDE_DIR}")
  message(STATUS "  - Python Library    : ${PYTHON_LIBRARY}")

ELSE()
  message(STATUS " Build Python Wrapper: NO")
ENDIF()

# #####################################################################
# ## INSTALL
# #####################################################################
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
  FILES_MATCHING PATTERN "*.h*")

# Create a ConfigVersion.cmake file
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

# #############################################
# # Exporting from the build tree
export(EXPORT cytnx_targets FILE ${CMAKE_CURRENT_BINARY_DIR}/CytnxTargets.cmake NAMESPACE Cytnx::)

export(PACKAGE Cytnx)

# build doc:
# #####################################################################
# ## Build & install Documents
# #####################################################################
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

  message(STATUS " Build Documentation: YES")
  message(STATUS "  - Doxygen: ${DOXYGEN_EXECUTABLE}")
else()
  message(STATUS " Build Documentation: NO")
endif()

if(DOXYGEN_FOUND)
  install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/docs/
    DESTINATION docs
    COMPONENT documentation
  )
endif()

# ====================================================================
set(libname "${CMAKE_SHARED_LIBRARY_PREFIX}cytnx.${CYTNX_VERSION}${CMAKE_SHARED_LIBRARY_SUFFIX}")

if(DEV_MODE)
  add_executable(dev_test ${CMAKE_CURRENT_SOURCE_DIR}/dev_test.cpp)
  target_link_libraries(dev_test cytnx)
  add_custom_command(TARGET dev_test POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:dev_test> ${CMAKE_CURRENT_SOURCE_DIR}/dev_test
  )
endif()

# #####################################################################
# Final Information Messages
# #####################################################################
message(STATUS "|= Final FLAGS infomation for install >>>>> ")
message(STATUS "    CXX Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "    CXX Flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "    BLAS and LAPACK Libraries: ${LAPACK_LIBRARIES}")
message(STATUS "    Link libraries: ${CYTNX_LINK_Libs}")

FILE(APPEND "${CMAKE_BINARY_DIR}/linkflags.tmp" "${LAPACK_LIBRARIES}\n" "")
FILE(APPEND "${CMAKE_BINARY_DIR}/cxxflags.tmp" "${CMAKE_CXX_FLAGS}\n" "")

message(STATUS "")
message(STATUS "")
message(STATUS "")

install(FILES ${CMAKE_BINARY_DIR}/linkflags.tmp DESTINATION ${CMAKE_INSTALL_PREFIX}/cytnx)
install(FILES ${CMAKE_BINARY_DIR}/cxxflags.tmp DESTINATION ${CMAKE_INSTALL_PREFIX}/cytnx)
install(FILES ${CMAKE_BINARY_DIR}/version.tmp DESTINATION ${CMAKE_INSTALL_PREFIX}/cytnx)
install(FILES ${CMAKE_BINARY_DIR}/vinfo.tmp DESTINATION ${CMAKE_INSTALL_PREFIX}/cytnx)

if(BUILD_PACKAGES)
  include("Package.cmake")
  include(CPack)
endif()
