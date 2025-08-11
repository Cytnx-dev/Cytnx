
if(BACKEND_TORCH)
else()

# Define the path to the test data directory
set(CYTNX_TEST_DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/test_data_base")

# Add the preprocessor definition for CYTNX_TEST_DATA_DIR
add_compile_definitions(CYTNX_TEST_DATA_DIR="${CYTNX_TEST_DATA_DIR}")

add_executable(
  test_main
  test_tools.cpp
  common_data_generator.cpp
  Bond_test.cpp
  Network_test.cpp
  UniTensor_base_test.cpp
  ncon_test.cpp
  Contract_test.cpp
  BlockUniTensor_test.cpp
  BlockFermionicUniTensor_test.cpp
  DenseUniTensor_test.cpp
  Accessor_test.cpp
  Tensor_test.cpp
  Storage_test.cpp
  search_tree_test.cpp
  utils_test/vec_concatenate.cpp
  utils_test/vec_unique.cpp
  utils/getNconParameter.cpp
  linalg_test/Det_test.cpp
  linalg_test/Directsum_test.cpp
  linalg_test/ExpH_test.cpp
  linalg_test/ExpM_test.cpp
  linalg_test/Lanczos_Exp_test.cpp
  linalg_test/Lanczos_Gnd_test.cpp
  linalg_test/Arnoldi_test.cpp
  linalg_test/Arnoldi_Ut_test.cpp
  linalg_test/Svd_test.cpp
  linalg_test/GeSvd_test.cpp
  linalg_test/linalg_test.cpp
  linalg_test/sum_test.cpp
  algo_test/Hsplit_test.cpp
  algo_test/Hstack_test.cpp
  algo_test/Vsplit_test.cpp
  algo_test/Vstack_test.cpp
  random_test/uniform_test.cpp
)

if(USE_CUDA)
      add_subdirectory(gpu)
endif()

target_include_directories(
  test_main
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(
  test_main
  gtest_main
  gmock
  gtest
)
target_link_libraries(test_main cytnx)

add_compile_options(-fsanitize=address)
add_link_options(-fsanitize=address)

#target_link_libraries(test_main PUBLIC "-lgcov --coverage")
include(GoogleTest)
gtest_discover_tests(test_main
                     WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

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

endif()
