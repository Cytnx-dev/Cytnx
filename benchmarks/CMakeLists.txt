if(BACKEND_TORCH)
else()

add_executable(
  benchmarks_main
  benchmarks_main.cpp
  Tensor_bm.cpp
  UniTensor_bm.cpp
  algo/Vsplit_bm.cpp
  algo/Vstack_bm.cpp
  algo/Hsplit_bm.cpp
  algo/Hstack_bm.cpp
  linalg/Arnoldi_bm.cpp
  linalg/Directsum_bm.cpp
  linalg/Svd_bm.cpp
  linalg/Svd_truncate_bm.cpp
  linalg/Lanczos_bm.cpp
  linalg/linalg_basic_bm.cpp
  linalg/QR_bm.cpp

)
if(USE_CUDA)
  target_compile_definitions(benchmarks_main PRIVATE UNI_GPU)
endif()

target_link_libraries(benchmarks_main cytnx)
target_link_libraries(benchmarks_main benchmark::benchmark)

endif()
