add_executable(
  test_main
  test_main.cpp
)
target_link_libraries(
  test_main
  gtest_main
)

include(GoogleTest)
gtest_discover_tests(test_main)
