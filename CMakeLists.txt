
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
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# set(CMAKE_CXX_EXTENSIONS OFF)
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
option(BACKEND_TORCH "Build Cytnx with pytorch backend (requires pytorch)" OFF)

# # the following are only referenced when BACKEND_TORCH=OFF
option(USE_CUDA "Build using Nvidia CUDA for GPU library" OFF)
option(BUILD_PYTHON "Build Python wrapper" ON)
option(USE_MKL "Build using MKL" ON)
option(USE_OMP "Build Cytnx with OpenMP" OFF)
option(BUILD_DOC "Build API docuemntation" OFF)
option(USE_HPTT "Build Cytnx with HPTT" OFF)
option(RUN_TESTS "Run Cytnx tests" OFF)
option(USE_CUTT "Build Cytnx with CUTT" OFF)
option(USE_MAGMA "Build Cytnx with MAGMA (requires CUDA)" ON)
option(USE_CUTENSOR "Build Cytnx with CuTensor (requires CUDA)" ON)
option(USE_CUQUANTUM "Build Cytnx with CUQuantum (requires CUDA)" ON)

option(DEV_MODE "Build testing dev_test.cpp with cytnx" OFF)

# #####################################################################
# ## PROJECT
# #####################################################################
project(CYTNX VERSION ${CYTNX_VERSION} LANGUAGES CXX C)

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

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wformat=0 -Wno-c++11-narrowing -w")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wformat=0 -w -fsized-deallocation -fsanitize=address")

target_compile_options(cytnx PUBLIC -Wformat=0 -w -fsized-deallocation)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wformat=0 -w -fsized-deallocation -diag-disable=10441")

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_LIBCPP_DISABLE_AVAILABILITY")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_LIBCPP_ENABLE_CXX17_REMOVED_UNARY_BINARY_FUNCTION")
target_compile_features(cytnx PUBLIC cxx_std_17)

# target_compile_definitions(cytnx PUBLIC _LIBCPP_DISABLE_AVAILABILITY)
# target_compile_definitions(cytnx PUBLIC _LIBCPP_ENABLE_CXX17_REMOVED_UNARY_BINARY_FUNCTION)
# set_property(TARGET cytnx PROPERTY CXX_STANDARD 11)
add_subdirectory(src)

if(USE_DEBUG)
  target_compile_definitions(cytnx PUBLIC UNI_DEBUG)

  # Remember to add export ASAN_OPTIONS=protect_shadow_gap=0 in Install.sh
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")
endif()

if(RUN_TESTS)
  enable_testing()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fprofile-arcs -ftest-coverage")
  target_link_libraries(cytnx PUBLIC "-lgcov --coverage")
  add_subdirectory(tests)
  #add_subdirectory(bm_tests)
endif()

include(GNUInstallDirs)

if(USE_ICPC)
  target_compile_definitions(cytnx PUBLIC UNI_ICPC)
  # This option is to disable the warning of icpc
  target_compile_options(cytnx PUBLIC -diag-disable=10441)
  # This option is to maintain consistency of the floating point operation, also try to improve the performance
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
  find_package(GTest REQUIRED)
  #find_package(benchmark REQUIRED)


  #include(FetchContent)
  #FetchContent_Declare(
  #  googletest
  #  URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
  #)

  # For Windows: Prevent overriding the parent project's compiler/linker settings
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

  #FetchContent_Declare(
  #  googlebenchmark
  #  GIT_REPOSITORY https://github.com/google/benchmark.git
  #  GIT_TAG v1.7.1) # need master for benchmark::benchmark
  #set(BENCHMARK_ENABLE_TESTING off) # to suppress benchmark internal tests

  #FetchContent_MakeAvailable(
  #  googletest
  #  googlebenchmark)
endif()

# #######
# ## Python wrapper
# #######
IF(BUILD_PYTHON)
  # PythonInterp is removed in newer version of cmake
  # find_package(PythonInterp REQUIRED)
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

# # this need to append into the linkflags.tmp
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
