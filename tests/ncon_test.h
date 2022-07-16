#ifndef _H_ncon_test
#define _H_ncon_test

#include "cytnx.hpp"
#include "utils/getNconParameter.h"
#include <gtest/gtest.h>

class NconTest : public ::testing::Test {
 public:
  std::pair<std::vector<cytnx::UniTensor>, std::vector<std::vector<cytnx::cytnx_int64>>> input;

 protected:
  void SetUp() override { input = getNconParameter("output.txt"); }
  void TearDown() override {}
};

#endif
