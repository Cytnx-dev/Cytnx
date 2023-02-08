#ifndef _H_DENSEUNITENSOR_BASE_TEST
#define _H_DENSEUNITENSOR_BASE_TEST

#include "cytnx.hpp"
#include <gtest/gtest.h>
using namespace std;
using namespace cytnx;
class DenseUniTensorTest : public ::testing::Test {
 public:
  cytnx::UniTensor utzero345;
  cytnx::UniTensor utone345;
 protected:
  void SetUp() override {
    utzero345 = cytnx::UniTensor(cytnx::zeros(3 * 4 * 5)).reshape({3, 4, 5});
  }
  void TearDown() override {}
};

#endif