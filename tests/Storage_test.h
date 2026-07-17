#ifndef CYTNX_TESTS_STORAGE_TEST_H_
#define CYTNX_TESTS_STORAGE_TEST_H_

#include <gtest/gtest.h>

#include "cytnx.hpp"
namespace cytnx {
  namespace test {

    class StorageTest : public ::testing::Test {
     public:
     protected:
      void SetUp() override {}
      void TearDown() override {}
    };

  }  // namespace test
}  // namespace cytnx
#endif  // CYTNX_TESTS_STORAGE_TEST_H_
