#ifndef CYTNX_TESTS_GPU_ACCESSOR_TEST_H_
#define CYTNX_TESTS_GPU_ACCESSOR_TEST_H_

#include "gtest/gtest.h"

#include "Accessor.hpp"
// Test fixture for Accessor class

namespace cytnx {
  namespace gpu_test {
    class AccessorTest : public ::testing::Test {
     protected:
      // Declare member variables that are used in the tests
      Accessor single, all, range, tilend, step, list;

      // Set up the test fixture
      void SetUp() override {
        single = Accessor(5);
        all = Accessor::all();
        range = Accessor::range(1, 4, 2);
        tilend = Accessor::tilend(2, 1);
        step = Accessor::step(3);
        list = Accessor({0, 2, 3});
      }
    };

  }  // namespace gpu_test
}  // namespace cytnx
#endif  // CYTNX_TESTS_GPU_ACCESSOR_TEST_H_
