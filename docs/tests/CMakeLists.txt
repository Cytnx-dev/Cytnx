# CMAKE VERSION
cmake_minimum_required(VERSION 3.20)
project(CYTNX_DOC_TEST)
set(BLA_VENDOR Intel10_64_dyn)
find_package( BLAS REQUIRED)
find_package( LAPACK REQUIRED)
set(LAPACK_LIBRARIES  ${BLAS_LIBRARIES}  ${LAPACK_LIBRARIES})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
message( STATUS "LAPACK found: ${LAPACK_LIBRARIES}")

include(ExternalProject)

option(MAGMA_ROOT "HPTT option AVX" OFF)
ExternalProject_Add(cytnx
  PREFIX cytnx_src
  GIT_REPOSITORY https://github.com/Cytnx-dev/Cytnx.git
  BINARY_DIR cytnx_src/build
  INSTALL_DIR cytnx_install
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR> -DUSE_MKL=ON -DBUILD_PYTHON=ON 
  -DUSE_HPTT=ON -DHPTT_ENABLE_FINE_TUNE=ON -DHPTT_ENABLE_AVX=ON
  -DUSE_CUDA=OFF
#-DUSE_MAGMA=OFF -DMAGMA_ROOT=/home/hunghaoti/MAGMA 
#-DUSE_CUTENSOR=OFF -DCUTENSOR_ROOT=/home/hunghaoti/CUTENSOR 
#-DUSE_CUQUANTUM=OFF -DCUQUANTUM_ROOT=/home/hunghaoti/CUQUANTUM 
  -DCMAKE_EXPORT_COMPILE_COMMANDS=1 -DUSE_OMP=OFF -DRUN_TESTS=OFF
)

include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.13.0
)

FetchContent_MakeAvailable(googletest)

enable_testing()
ExternalProject_Get_Property(cytnx install_dir)
add_executable(
  test_doc_cplusplus
  test_doc.cpp
)
target_link_libraries(
  test_doc_cplusplus
  gtest_main
  gtest
  ${install_dir}/lib64/libcytnx.a
  ${install_dir}/hptt/lib/libhptt.a
  ${LAPACK_LIBRARIES}
)

target_include_directories(
  test_doc_cplusplus
  PRIVATE
  ${install_dir}/include
  ../code/cplusplus/doc_codes
) 

include(GoogleTest)

#file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/ctm.net"
#  DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
