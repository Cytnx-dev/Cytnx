cmake_minimum_required(VERSION 3.25)
project(cytnx_gpu_get_benchmark LANGUAGES CXX)

# Consume the existing Cytnx install tree. Never configure or rebuild Cytnx
# with RUN_BENCHMARKS=ON: that would invalidate the release-build cache.
find_package(Cytnx CONFIG REQUIRED)
find_package(benchmark CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(gpu_get_benchmark
  ${CYTNX_SOURCE_DIR}/benchmarks/benchmarks_main.cpp
  ${CYTNX_SOURCE_DIR}/benchmarks/gpu/Get_contiguous_double_bm.cpp
)
if(NOT DEFINED CYTNX_SOURCE_DIR OR
   NOT EXISTS "${CYTNX_SOURCE_DIR}/src/backend/utils_internal_gpu/cuGetElems_contiguous_gpu_bench.hpp")
  message(FATAL_ERROR "Set CYTNX_SOURCE_DIR to the matching Cytnx source checkout; internal benchmark headers are not installed.")
endif()
target_compile_features(gpu_get_benchmark PRIVATE cxx_std_20)
target_compile_definitions(gpu_get_benchmark PRIVATE UNI_GPU)
target_include_directories(gpu_get_benchmark PRIVATE "${CYTNX_SOURCE_DIR}/src")
target_link_libraries(gpu_get_benchmark PRIVATE
  Cytnx::cytnx benchmark::benchmark CUDA::cudart)
