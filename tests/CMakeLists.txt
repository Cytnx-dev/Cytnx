
if(BACKEND_TORCH)
else()

# GoogleTest (gtest + gmock) is a required prerequisite for the test suite and
# must be provided by the toolchain / a config-mode install. It is deliberately
# NOT fetched here so the CMake configuration stays free of network access
# (see #944, which moved the other optional deps to submodules/finders).
# find_package(GTest) can succeed with only gtest -- the GTest::gmock target
# needs a config-mode install or CMake's FindGTest from 3.23+ -- so require the
# gmock target explicitly and fail with actionable guidance when it is missing.
find_package(GTest REQUIRED)
if(NOT TARGET GTest::gmock)
  message(FATAL_ERROR
    "GoogleTest was found but the GTest::gmock target is missing, which the test "
    "suite links against. Install a GoogleTest that also provides GMock (it is not "
    "fetched automatically):\n"
    "  macOS:         brew install googletest\n"
    "  Ubuntu/Debian: apt install libgtest-dev libgmock-dev\n"
    "  conda:         conda install -c conda-forge gtest gmock\n"
    "or point CMake at such an install with -DCMAKE_PREFIX_PATH=<gtest-prefix>.")
endif()

# Define the path to the test data directory
set(CYTNX_TEST_DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/test_data_base")

# Add the preprocessor definition for CYTNX_TEST_DATA_DIR
add_compile_definitions(CYTNX_TEST_DATA_DIR="${CYTNX_TEST_DATA_DIR}")

add_executable(
  test_main
  test_tools.cpp
  common_data_generator.cpp
  Symmetry_test.cpp
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
  cytnx_error_test.cpp
  include_order_canary_test.cpp
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
  linalg_test/DtypePromotion_test.cpp
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
