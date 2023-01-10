#ifndef _H_BlockUniTensor_test
#define _H_BlockUniTensor_test

#include "cytnx.hpp"
#include <gtest/gtest.h>

using namespace cytnx;

class BlockUniTensorTest : public ::testing::Test {
 public:
  Bond B1 = Bond(BD_IN, {Qs(0), Qs(1)}, {1, 2});
  Bond B2 = Bond(BD_IN, {Qs(0), Qs(1)}, {3, 4});
  Bond B3 = Bond(BD_OUT, {Qs(0), Qs(1)}, {2, 3});
  Bond B4 = Bond(BD_OUT, {Qs(0), Qs(1)}, {1, 2});
  UniTensor BUT1 = UniTensor({B1, B2, B3, B4});

  Bond bd_sym_a = Bond(BD_KET, {{0, 2}, {3, 5}, {1, 6}, {4, 1}}, {4, 7, 2, 3});
  Bond bd_sym_b = Bond(BD_BRA, {{0, 2}, {3, 5}, {1, 6}, {4, 1}}, {4, 7, 2, 3});
  UniTensor BUT2 = UniTensor({bd_sym_a, bd_sym_b});

  Bond bd_sym_c =
    Bond(BD_KET, {{0, 2}, {1, 5}, {1, 6}, {0, 1}}, {4, 7, 2, 3}, {Symmetry::Zn(2), Symmetry::U1()});
  Bond bd_sym_d =
    Bond(BD_BRA, {{0, 2}, {1, 5}, {1, 6}, {0, 1}}, {4, 7, 2, 3}, {Symmetry::Zn(2), Symmetry::U1()});
  UniTensor BUT3 = UniTensor({bd_sym_c, bd_sym_d});

 protected:
  void SetUp() override {}
  void TearDown() override {}
};

#endif