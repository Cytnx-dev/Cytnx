#ifndef _H_UNITENSOR_BASE_TEST
#define _H_UNITENSOR_BASE_TEST

#include "cytnx.hpp"
#include <gtest/gtest.h>

class UniTensor_baseTest : public ::testing::Test {
 public:
  cytnx::UniTensor utzero345;

 protected:
  void SetUp() override {
    utzero345 = cytnx::UniTensor(cytnx::zeros(3 * 4 * 5)).reshape({3, 4, 5});
    utzero345.to_(cytnx::Device.cuda);
  }
  void TearDown() override {}
};

#endif
