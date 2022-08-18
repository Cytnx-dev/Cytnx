add_executable(
  test_main
  test_main.cpp
  Bond_test.cpp
  utils_test/vec_concatenate.cpp
  utils_test/vec_unique.cpp
)
target_link_libraries(
  test_main
  gtest_main
  gmock
  gtest
)
target_link_libraries(test_main cytnx)
#target_link_libraries(test_main PUBLIC "-lgcov --coverage")
include(GoogleTest)
gtest_discover_tests(test_main)

file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/testNet.net"
      DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/testNet.net"
      DESTINATION ${CMAKE_BINARY_DIR})
