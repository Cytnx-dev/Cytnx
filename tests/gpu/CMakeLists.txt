add_executable(
  gpu_test_main
  gpu_test_tools.cpp
  common_data_generator.cpp
  Bond_test.cpp
  Network_test.cpp
  UniTensor_base_test.cpp
  ncon_test.cpp
  Contract_test.cpp
  BlockUniTensor_test.cpp
  DenseUniTensor_test.cpp
  Accessor_test.cpp
  Tensor_test.cpp
  utils_test/vec_concatenate.cpp
  utils_test/vec_unique.cpp
  utils/getNconParameter.cpp
  linalg_test/Abs_test.cpp
  linalg_test/Add_test.cpp
  linalg_test/Cpr_test.cpp
  linalg_test/Div_test.cpp
  linalg_test/Mod_test.cpp
  linalg_test/Mul_test.cpp
  linalg_test/Sub_test.cpp
  linalg_test/Det_test.cpp
  linalg_test/Directsum_test.cpp
  linalg_test/ExpH_test.cpp
  linalg_test/ExpM_test.cpp
  linalg_test/Lanczos_test.cpp
  linalg_test/Lanczos_Gnd_test.cpp
  linalg_test/Arnoldi_test.cpp
  linalg_test/Arnoldi_Ut_test.cpp
  linalg_test/Svd_test.cpp
  linalg_test/GeSvd_test.cpp
  linalg_test/Rsvd_test.cpp
  linalg_test/truncate_test.cpp
  linalg_test/Trace_test.cpp
  linalg_test/linalg_test.cpp
  linalg_test/sum_test.cpp
  algo_test/Hsplit_test.cpp
  algo_test/Hstack_test.cpp
  algo_test/Sort_test.cpp
  algo_test/Vsplit_test.cpp
  algo_test/Vstack_test.cpp
  random_test/uniform_test.cpp
)

target_include_directories(
  gpu_test_main
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(
  gpu_test_main
  GTest::gtest_main
  GTest::gmock
  GTest::gtest
)
target_link_libraries(gpu_test_main cytnx)
#target_link_libraries(test_main PUBLIC "-lgcov --coverage")

# gpu_test_main's .cpp TUs are host-compiled (g++) but include cytnx public
# headers (Type.hpp) that pull <cuda/std/complex>. cytnx adds the CUDA toolkit
# and CCCL include dirs PRIVATEly, so they do not propagate here through
# linking; add them explicitly, otherwise the host compile fails with
# "fatal error: cuda/std/complex: No such file or directory".
target_include_directories(gpu_test_main PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
if(CYTNX_CCCL_INCLUDE_DIR)
  target_include_directories(gpu_test_main PRIVATE ${CYTNX_CCCL_INCLUDE_DIR})
endif()

include(GoogleTest)
cmake_path(APPEND CMAKE_CURRENT_SOURCE_DIR ".." OUTPUT_VARIABLE test_dir)
gtest_discover_tests(gpu_test_main
                     WORKING_DIRECTORY ${test_dir}
                     # Tag every discovered case with the `gpu` ctest label so
                     # CI can run just this suite via `ctest -L gpu`.
                     PROPERTIES LABELS gpu
                     # PRE_TEST defers running the binary to enumerate tests
                     # until ctest time instead of at build time, so a
                     # CUDA-enabled build still links on a machine without a
                     # GPU driver (only running the tests needs real hardware).
                     DISCOVERY_MODE PRE_TEST)

file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/testNet.net"
      DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/testNet.net"
      DESTINATION ${CMAKE_BINARY_DIR})

file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/utils/output.txt"
      DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/utils/output.txt"
      DESTINATION ${CMAKE_BINARY_DIR})

file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/utils/answer.txt"
      DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/utils/answer.txt"
      DESTINATION ${CMAKE_BINARY_DIR})
