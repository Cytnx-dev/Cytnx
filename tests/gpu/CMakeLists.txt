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
  linalg_test/Det_test.cpp
  linalg_test/Directsum_test.cpp
  linalg_test/ExpH_test.cpp
  linalg_test/ExpM_test.cpp
  linalg_test/Lanczos_Gnd_test.cpp
  linalg_test/Arnoldi_test.cpp
  linalg_test/Svd_test.cpp
  linalg_test/GeSvd_test.cpp
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
  gtest_main
  gmock
  gtest
)
target_link_libraries(gpu_test_main cytnx)
#target_link_libraries(test_main PUBLIC "-lgcov --coverage")
include(GoogleTest)
cmake_path(APPEND CMAKE_CURRENT_SOURCE_DIR ".." OUTPUT_VARIABLE test_dir)
gtest_discover_tests(gpu_test_main
                     WORKING_DIRECTORY ${test_dir})

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
