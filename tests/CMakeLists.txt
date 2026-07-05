
if(BACKEND_TORCH)
else()

# GoogleTest (gtest + gmock): prefer a system/toolchain install, otherwise
# fetch a pinned release so the test suite builds on machines without one
# (e.g. macOS). The GTest::gmock target only exists with a config-mode
# install or CMake's FindGTest module from 3.23+, so key the fallback on
# the target rather than GTest_FOUND.
find_package(GTest QUIET)
if(NOT TARGET GTest::gmock)
  include(FetchContent)
  # Keep the MSVC runtime selection consistent with the parent project.
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz
    URL_HASH SHA256=8ad598c73ad796e0d8280b082cebd82a630d73e73cd3c70057938a6501bba5d7
  )
  FetchContent_MakeAvailable(googletest)
endif()

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
  Type_test.cpp
  Tensor_strides_test.cpp
  Storage_test.cpp
  search_tree_test.cpp
  utils_test/vec_concatenate.cpp
  utils_test/vec_unique.cpp
  utils/getNconParameter.cpp
  linalg_test/Det_test.cpp
  linalg_test/Directsum_test.cpp
  linalg_test/ExpH_test.cpp
  linalg_test/ExpM_test.cpp
  linalg_test/Lanczos_test.cpp
  linalg_test/Lanczos_Exp_test.cpp
  linalg_test/Lanczos_Gnd_test.cpp
  linalg_test/Arnoldi_test.cpp
  linalg_test/Arnoldi_Ut_test.cpp
  linalg_test/Svd_test.cpp
  linalg_test/GeSvd_test.cpp
  linalg_test/Svd_truncate_test.cpp
  linalg_test/Gesvd_truncate_test.cpp
  linalg_test/Rsvd_test.cpp
  linalg_test/Gemm_Batch_test.cpp
  linalg_test/Trace_test.cpp
  linalg_test/linalg_test.cpp
  linalg_test/stride_view_test.cpp
  linalg_test/sum_test.cpp
  linalg_test/Vectordot_test.cpp
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
  # Internal backend headers (e.g. linalg_internal_cpu/stride_view.hpp) live
  # under src/ and are PRIVATE to the cytnx target; expose them to the test
  # binary so internal utilities can be unit-tested directly.
  ${CMAKE_SOURCE_DIR}/src
)
target_link_libraries(
  test_main
  GTest::gtest_main
  GTest::gmock
  GTest::gtest
)
target_link_libraries(test_main cytnx)

add_compile_options(-fsanitize=address)
add_link_options(-fsanitize=address)

#target_link_libraries(test_main PUBLIC "-lgcov --coverage")
include(GoogleTest)
gtest_discover_tests(test_main
                     WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                     # The test binary takes longer to enumerate than the
                     # default 5s under MKL + ASAN on slower runners.
                     DISCOVERY_TIMEOUT 120)

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
