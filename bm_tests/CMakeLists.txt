if(BACKEND_TORCH)
else()

add_executable(
  bm_test_main
  bm_test_main.cpp
  Tensor_bm.cpp
  UniTensor_bm.cpp
  algo/Vsplit_bm.cpp
  algo/Vstack_bm.cpp
  algo/Hsplit_bm.cpp
  algo/Hstack_bm.cpp
  linalg/Directsum_bm.cpp
  linalg/Svd_bm.cpp
  linalg/Svd_truncate_bm.cpp
  linalg/Lanczos_bm.cpp

)
if(USE_CUDA)
  target_compile_definitions(bm_test_main PRIVATE UNI_GPU)
endif()

target_link_libraries(bm_test_main cytnx)
target_link_libraries(bm_test_main benchmark::benchmark)

endif()
