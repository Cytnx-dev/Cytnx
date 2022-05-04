add_executable(
  test_main
  test_main.cpp
  utils_test/vec_test.cpp
)
target_link_libraries(
  test_main
  gtest_main
)
target_link_libraries(test_main cytnx)
include(GoogleTest)
gtest_discover_tests(test_main)
