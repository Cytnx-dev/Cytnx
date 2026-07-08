#include <vector>

#include "gtest/gtest.h"

#include "Symmetry.hpp"

namespace {

  using cytnx::Symmetry;

  TEST(Symmetry, FermionParityCheckQnum) {
    Symmetry sym = Symmetry::FermionParity();
    EXPECT_TRUE(sym.check_qnum(0));
    EXPECT_TRUE(sym.check_qnum(1));
    EXPECT_FALSE(sym.check_qnum(-1));
    EXPECT_FALSE(sym.check_qnum(2));
  }

  TEST(Symmetry, FermionParityCheckQnums) {
    Symmetry sym = Symmetry::FermionParity();
    EXPECT_TRUE(sym.check_qnums({0, 1}));
    EXPECT_FALSE(sym.check_qnums({2}));
    EXPECT_FALSE(sym.check_qnums({-1}));
    EXPECT_FALSE(sym.check_qnums({0, 1, 2}));
  }

}  // namespace
