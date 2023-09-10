#ifndef _H_ncon_test
#define _H_ncon_test

#include "cytnx.hpp"
#include <gtest/gtest.h>
#include "test_tools.h"

using namespace cytnx;
using namespace TestTools;

class NconTest : public ::testing::Test {
 public:
  // std::pair<std::vector<cytnx::UniTensor>, std::vector<std::vector<cytnx::cytnx_int64>>> input;
  UniTensor utdnA =
    UniTensor(arange(0, 8, 1, Type.ComplexDouble)).reshape({2, 2, 2}).to(cytnx::Device.cuda);
  ;
  UniTensor utdnB = UniTensor(ones({2, 2}, Type.ComplexDouble)).to(cytnx::Device.cuda);
  ;
  UniTensor utdnC = UniTensor(eye(2, Type.ComplexDouble)).to(cytnx::Device.cuda);
  ;
  UniTensor utdnAns = UniTensor(zeros({2, 2, 2}, Type.ComplexDouble)).to(cytnx::Device.cuda);
  ;

 protected:
  void SetUp() override {
    // input = getNconParameter("utils/output.txt");
    utdnAns.at({0, 0, 0}) = 1;
    utdnAns.at({0, 0, 1}) = 1;
    utdnAns.at({0, 1, 0}) = 5;
    utdnAns.at({0, 1, 1}) = 5;
    utdnAns.at({1, 0, 0}) = 9;
    utdnAns.at({1, 0, 1}) = 9;
    utdnAns.at({1, 1, 0}) = 13;
    utdnAns.at({1, 1, 1}) = 13;
  }
  void TearDown() override {}
};

#endif
