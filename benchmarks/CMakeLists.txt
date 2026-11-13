add_executable(
  benchmarks_main
  benchmarks_main.cpp
  Tensor_bm.cpp
  gpu/Get_contiguous_double_bm.cpp
  UniTensor_bm.cpp
  cublas_handle_bm.cpp
  cutensor_handle_bm.cpp
  search_tree_bm.cpp
  algo/Vsplit_bm.cpp
  algo/Vstack_bm.cpp
  algo/Hsplit_bm.cpp
  algo/Hstack_bm.cpp
  linalg/Arnoldi_bm.cpp
  linalg/Abs_bm.cpp
  linalg/iArithmetic_bm.cpp
  linalg/Directsum_bm.cpp
  linalg/Svd_bm.cpp
  linalg/Svd_truncate_bm.cpp
  linalg/Trace_bm.cpp
  linalg/Mul_noncontig_bm.cpp
  linalg/Vectordot_bm.cpp
  linalg/Lanczos_bm.cpp
  linalg/linalg_basic_bm.cpp
  linalg/QR_bm.cpp

)
if(USE_CUDA)
  target_compile_definitions(benchmarks_main PRIVATE UNI_GPU)
  # The L2 get benchmark intentionally calls experimental internal CUDA APIs.
  # cytnx exports include/ publicly, but keeps src/ private to the library;
  # explicitly expose it to this benchmark-only target.
  target_include_directories(benchmarks_main PRIVATE "${PROJECT_SOURCE_DIR}/src")
endif()

target_link_libraries(benchmarks_main cytnx)
target_link_libraries(benchmarks_main benchmark::benchmark)
