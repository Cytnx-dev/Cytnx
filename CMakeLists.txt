
######################################################################
### CMAKE VERSION
######################################################################
cmake_minimum_required(VERSION 3.13)

include(cmake/target_sources_local.cmake)
include(ExternalProject)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")
message(STATUS ${CMAKE_MODULE_PATH})

#set(CMAKE_CXX_STANDARD 11)
#set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(CMAKE_CXX_EXTENSIONS OFF)


######################################################################
### DISABLING IN-SOURCE BUILD
### (DO NOT MODIFY)
######################################################################
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

# Generate pif code
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "DEBUG")
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
#include(version.cmake)
set(CYTNX_VERSION_MAJOR 0)
set(CYTNX_VERSION_MINOR 7)
set(CYTNX_VERSION_PATCH 7)
set(CYTNX_VERSION COUNT 0)
set(CYTNX_VERSION
  ${CYTNX_VERSION_MAJOR}.${CYTNX_VERSION_MINOR}.${CYTNX_VERSION_PATCH}
)

set(CYTNX_VARIANT_INFO "")


######################################################################
### COMMAND LINE OPTIONS
######################################################################


option(USE_CUDA "Build using Nvidia CUDA for GPU library" OFF)
option(BUILD_PYTHON "Build Python wrapper" ON)
option(USE_MKL "Build using MKL" OFF)
option(USE_OMP "Build Cytnx with OpenMP" ON)
option(BUILD_DOC "Build API docuemntation" OFF)
option(USE_HPTT "Build Cytnx with HPTT" OFF)
option(RUN_TESTS "Run Cytnx tests" OFF)

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
    set(USE_ICPC OFF)
  endif ()
endif ()

if (USE_MKL)
  option(MKL_SDL "Link to a single MKL dynamic libary." ON)
  option(MKL_MLT "Use multi-threading libary. [Default]" ON)
  mark_as_advanced(MKL_SDL MKL_MLT)
  set(CYTNX_VARIANT_INFO "${CYTNX_VARIANT_INFO} UNI_MKL")
endif()

if (USE_HPTT)
option(HPTT_ENABLE_ARM "HPTT option ARM" OFF)
option(HPTT_ENABLE_AVX "HPTT option AVX" OFF)
option(HPTT_ENABLE_IBM "HPTT option IBM" OFF)
option(HPTT_ENABLE_FINE_TUNE "HPTT option FINE_TUNE" OFF)
set(CYTNX_VARIANT_INFO "${CYTNX_VARIANT_INFO} UNI_HPTT")
ExternalProject_Add(hptt
  PREFIX hptt_src
  GIT_REPOSITORY https://github.com/kaihsin/hptt.git
  GIT_TAG fc9c8cb9b71f4f6d16aad435bdce20025b342a73
  BINARY_DIR hptt_src/build
  INSTALL_DIR hptt
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR> -DENABLE_ARM=${HPTT_ENABLE_ARM} -DENABLE_AVX=${HPTT_ENABLE_AVX} -DENABLE_IBM=${HPTT_ENABLE_IBM} -DFINE_TUNE=${HPTT_ENABLE_FINE_TUNE}
)
endif()

if (USE_CUDA)
    if (USE_CUTT)
    option(CUTT_ENABLE_FINE_TUNE "CUTT option FINE_TUNE" OFF)
    option(CUTT_ENABLE_NVTOOLS "CUTT option NVTOOLS" OFF)
    option(CUTT_NO_ALIGN_ALLOC "CUTT option NO_ALIGN_ALLIC" OFF)
    set(CYTNX_VARIANT_INFO "${CYTNX_VARIANT_INFO} UNI_CUTT")
    ExternalProject_Add(cutt
      PREFIX cutt_src
      GIT_REPOSITORY https://github.com/kaihsin/cutt.git
      GIT_TAG 27ed59a42f2610923084c4687327d00f4c2d1d2d
      BINARY_DIR cutt_src/build
      INSTALL_DIR cutt
      CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR> -DNO_ALIGN_ALLOC=${CUTT_NO_ALIGN_ALLOC} -DENABLE_NVTOOLS=${CUTT_ENABLE_NVTOOLS} -DFINE_TUNE=${CUTT_ENABLE_FINE_TUNE}
    )
    endif()
endif()

######################################################################
### PROJECT
######################################################################
project(CYTNX VERSION ${CYTNX_VERSION} LANGUAGES CXX C )
set(CMAKE_CXX_STANDARD 11)

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

if(RUN_TESTS)
	enable_testing()
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fprofile-arcs -ftest-coverage")
	target_link_libraries(cytnx PUBLIC "-lgcov --coverage")
	add_subdirectory(tests)
endif()

include(GNUInstallDirs)


if(USE_ICPC)
    target_compile_definitions(cytnx PRIVATE UNI_ICPC)
endif()


#####################################################################
### Dependency of CUTT
#####################################################################
if(USE_CUDA)
    if(USE_CUTT)
        ExternalProject_Get_Property(cutt install_dir)
        include_directories(${install_dir}/include)
        message(STATUS "cutt install dir: ${install_dir}")
        add_dependencies(cytnx cutt)
        set_property(TARGET cytnx PROPERTY CUDA_ARCHITECTURES 52 53 60 61 62 70 72 75 80 86)
        target_compile_definitions(cytnx PRIVATE UNI_CUTT)
        target_link_libraries(cytnx PRIVATE ${install_dir}/lib/libcutt.a)
    endif()
endif()

#####################################################################
### Dependency of HPTT
#####################################################################
if(USE_HPTT)
    ExternalProject_Get_Property(hptt install_dir)
    include_directories(${install_dir}/include)
    message(STATUS "hptt install dir: ${install_dir}")
    add_dependencies(cytnx hptt)
    target_compile_definitions(cytnx PRIVATE UNI_HPTT)
    target_link_libraries(cytnx PRIVATE ${install_dir}/lib/libhptt.a)
endif()



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
    set(BLA_VENDOR OpenBLAS)
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
    # set(USE_OMP ON FORCE)
endif()


######################################################################
### Get cmake-boost
######################################################################
#include(FetchContent)
#message("Starting to clone boost-cmake into boost-cmake")
#FetchContent_Declare(
#  boost-cmake
#  URL https://github.com/Orphis/boost-cmake/archive/d3951bc7f0b9d09005f92aedcf6acfc595f050ea.zip
#  CONFIGURE_COMMAND ""
#  BUILD_COMMAND ""
#  INSTALL_COMMAND ""
#)
#FetchContent_MakeAvailable(boost-cmake)
find_package(Boost REQUIRED)
#add_subdirectory(_deps/boost-cmake-src)
#message(STATUS "Boost INCCC: ${Boost_INCLUDE_DIRS}")
#message(STATUS ${COMPILE_DEFINITIONS})

# ----------------------------------------
# Find OpenMP
if(USE_OMP)
    set(CYTNX_VARIANT_INFO "${CYTNX_VARIANT_INFO} UNI_OMP")
    if(USE_MKL)
        target_compile_definitions(cytnx PRIVATE UNI_OMP)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
    else()
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
else()
    if(USE_MKL)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
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
  set(CYTNX_VARIANT_INFO "${CYTNX_VARIANT_INFO} UNI_CUDA")
  enable_language(CUDA)
  find_package(CUDA REQUIRED)
  if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  endif()

  set_target_properties(cytnx PROPERTIES 
      CUDA_SEPARABLE_COMPILATION ON
                                   )
  set_target_properties(cytnx PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe=--display_error_number -lineinfo")
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
### Get Gtest
######################################################################
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)


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


# create a file that contain all the link flags:
FILE(WRITE "${CMAKE_BINARY_DIR}/linkflags.tmp" "" "")
FILE(WRITE "${CMAKE_BINARY_DIR}/cxxflags.tmp" "" "")

FILE(WRITE "${CMAKE_BINARY_DIR}/version.tmp" "" "")
FILE(APPEND "${CMAKE_BINARY_DIR}/version.tmp" "${CYTNX_VERSION}" "")

FILE(WRITE "${CMAKE_BINARY_DIR}/vinfo.tmp" "" "")
FILE(APPEND "${CMAKE_BINARY_DIR}/vinfo.tmp" "${CYTNX_VARIANT_INFO}" "")

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
message(STATUS " Link libraries: ${CYTNX_LINK_Libs}")
#message(STATUS " Boost libs: ${Boost_LIBRARIES}")
FILE(APPEND "${CMAKE_BINARY_DIR}/linkflags.tmp" "${LAPACK_LIBRARIES}\n" "")

FILE(APPEND "${CMAKE_BINARY_DIR}/cxxflags.tmp" "-I${Boost_INCLUDE_DIRS}\n" "")
FILE(APPEND "${CMAKE_BINARY_DIR}/cxxflags.tmp" "${CMAKE_CXX_FLAGS}\n" "")
#FILE(APPEND "${CMAKE_BINARY_DIR}/cxxflags.tmp" "-std=c++11\n" "")
#foreach(vars ${ALL_CXXDEF})
#    FILE(APPEND "${CMAKE_BINARY_DIR}/cxxflags.tmp" "-D${vars}\n" "")
#endforeach(vars)



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
  FILE(APPEND "${CMAKE_BINARY_DIR}/cxxflags.tmp" "-DUNI_GPU\n" "")
  message( STATUS "  - Internal macro switch: GPU/CUDA")
  FILE(APPEND "${CMAKE_BINARY_DIR}/linkflags.tmp" "${CUDA_cusolver_LIBRARY}\n" "")
  message( STATUS "  - Curand library: ${CUDA_curand_LIBRARY}")
  FILE(APPEND "${CMAKE_BINARY_DIR}/linkflags.tmp" "${CUDA_curand_LIBRARY}\n" "")
  message( STATUS "  - CuBlas library: ${CUDA_cublas_LIBRARY}")
  FILE(APPEND "${CMAKE_BINARY_DIR}/linkflags.tmp" "${CUDA_cublas_LIBRARY}\n" "")
  message( STATUS "  - Cuda rt library: ${CUDA_cudart_static_LIBRARY} -ldl")
  FILE(APPEND "${CMAKE_BINARY_DIR}/linkflags.tmp" "${CUDA_cudart_static_LIBRARY} -ldl\n" "") # use > to indicate special rt processing 
  message( STATUS "  - Cuda devrt library: ${CUDA_cudadevrt_LIBRARY} -lrt")
  FILE(APPEND "${CMAKE_BINARY_DIR}/linkflags.tmp" "${CUDA_cudadevrt_LIBRARY} -lrt\n" "") # use > to indicate special rt processing 
  if(USE_CUTT)    
        message( STATUS " Build CUTT Support: YES")
        message( STATUS " --CUTT option FINE_TUNE: ${CUTT_ENABLE_FINE_TUNE}")
        message( STATUS " --CUTT option NVTOOLS: ${CUTT_ENABLE_NVTOOLS}")
        message( STATUS " --CUTT option NO_ALIGN_ALLOC: ${HPTT_NO_ALIGN_ALLOC}")
        FILE(APPEND "${CMAKE_BINARY_DIR}/cxxflags.tmp" "-DUNI_CUTT\n" "")
  endif()
else()
  message( STATUS " Build CUDA Support: NO")
endif()


if(USE_OMP)
    FILE(APPEND "${CMAKE_BINARY_DIR}/cxxflags.tmp" "-DUNI_OMP\n" "")
    if(USE_MKL)
          # message( STATUS " Build OMP Support: YES (force with MKL=on)")
          message( STATUS " Build OMP Support: YES")
          FILE(APPEND "${CMAKE_BINARY_DIR}/cxxflags.tmp" "-DUNI_MKL\n" "")
    else()
        if(OPENMP_FOUND)
          message( STATUS " Build OMP Support: YES")
        else()
          message( STATUS " Build OMP Support: NO  (Not found)")
        endif()
    endif()
else()
      message( STATUS " Build OMP Support: NO")
endif()

if(USE_HPTT)    
    message( STATUS " Build HPTT Support: YES")
    message( STATUS " --HPTT option FINE_TUNE: ${HPTT_ENABLE_FINE_TUNE}")
    message( STATUS " --HPTT option ARM: ${HPTT_ENABLE_ARM}")
    message( STATUS " --HPTT option AVX: ${HPTT_ENABLE_AVX}")
    message( STATUS " --HPTT option IBM: ${HPTT_ENABLE_IBM}")
    FILE(APPEND "${CMAKE_BINARY_DIR}/cxxflags.tmp" "-DUNI_HPTT\n" "")
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


install(FILES ${CMAKE_BINARY_DIR}/linkflags.tmp DESTINATION ${CMAKE_INSTALL_PREFIX}/cytnx)
install(FILES ${CMAKE_BINARY_DIR}/cxxflags.tmp DESTINATION ${CMAKE_INSTALL_PREFIX}/cytnx)
install(FILES ${CMAKE_BINARY_DIR}/version.tmp DESTINATION ${CMAKE_INSTALL_PREFIX}/cytnx)
install(FILES ${CMAKE_BINARY_DIR}/vinfo.tmp DESTINATION ${CMAKE_INSTALL_PREFIX}/cytnx)

if(USE_HPTT) # relocate HPTT library 
install(DIRECTORY ${CMAKE_BINARY_DIR}/hptt DESTINATION ${CMAKE_INSTALL_PREFIX})
endif()

if(USE_CUDA)
    if(USE_CUTT)
        install(DIRECTORY ${CMAKE_BINARY_DIR}/cutt DESTINATION ${CMAKE_INSTALL_PREFIX})
    endif()
endif()


if (BUILD_PACKAGES)
include("Package.cmake")
include(CPack)
endif()
