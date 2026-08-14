#ifndef CYTNX_TESTS_UNITENSOR_BASE_TEST_H_
#define CYTNX_TESTS_UNITENSOR_BASE_TEST_H_

#include <gtest/gtest.h>

#include "cytnx.hpp"

namespace cytnx {
  namespace test {
    class UniTensor_baseTest : public ::testing::Test {
     public:
      UniTensor utzero345;

     protected:
      void SetUp() override { utzero345 = UniTensor(zeros({3, 4, 5})); }
      void TearDown() override {}
    };

  }  // namespace test
}  // namespace cytnx
#endif  // CYTNX_TESTS_UNITENSOR_BASE_TEST_H_
