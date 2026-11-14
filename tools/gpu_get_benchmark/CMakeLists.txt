cmake_minimum_required(VERSION 3.25)
project(cytnx_gpu_get_benchmark LANGUAGES CXX CUDA)

# Standalone backend comparison: compile the exact S0 production source and
# the experimental S1-S4 source under identical CUDA compiler settings.
# No Cytnx library, install tree, or RUN_BENCHMARKS build is required.
if(NOT DEFINED CYTNX_SOURCE_DIR)
  get_filename_component(CYTNX_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
endif()
foreach(source IN ITEMS
  src/backend/utils_internal_gpu/cuGetElems_contiguous_gpu.cu
  src/backend/utils_internal_gpu/cuGetElems_contiguous_gpu_bench.cu
  benchmarks/gpu/Get_contiguous_double_bm.cpp)
  if(NOT EXISTS "${CYTNX_SOURCE_DIR}/${source}")
    message(FATAL_ERROR "Missing benchmark source: ${CYTNX_SOURCE_DIR}/${source}")
  endif()
endforeach()

find_package(CUDAToolkit REQUIRED)
find_package(benchmark CONFIG QUIET)
if(NOT benchmark_FOUND)
  include(FetchContent)
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(googlebenchmark
    URL https://github.com/google/benchmark/archive/refs/tags/v1.9.4.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  FetchContent_MakeAvailable(googlebenchmark)
endif()

add_executable(gpu_get_benchmark
  "${CMAKE_CURRENT_LIST_DIR}/standalone_globals.cpp"
  "${CYTNX_SOURCE_DIR}/benchmarks/benchmarks_main.cpp"
  "${CYTNX_SOURCE_DIR}/benchmarks/gpu/Get_contiguous_double_bm.cpp"
  "${CYTNX_SOURCE_DIR}/src/backend/utils_internal_gpu/cuGetElems_contiguous_gpu.cu"
  "${CYTNX_SOURCE_DIR}/src/backend/utils_internal_gpu/cuGetElems_contiguous_gpu_bench.cu"
)
target_compile_features(gpu_get_benchmark PRIVATE cxx_std_20 cuda_std_17)
target_compile_definitions(gpu_get_benchmark PRIVATE UNI_GPU)
target_include_directories(gpu_get_benchmark PRIVATE
  "${CYTNX_SOURCE_DIR}/src"
  "${CYTNX_SOURCE_DIR}/include")
target_link_libraries(gpu_get_benchmark PRIVATE benchmark::benchmark CUDA::cudart)
