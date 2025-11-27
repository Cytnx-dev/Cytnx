#ifndef _H_BlockFermionicUniTensor_test
#define _H_BlockFermionicUniTensor_test

#include "cytnx.hpp"
#include <gtest/gtest.h>
#include "test_tools.h"

using namespace cytnx;
using namespace TestTools;
class BlockFermionicUniTensorTest : public ::testing::Test {
 public:
  std::string data_dir = CYTNX_TEST_DATA_DIR "/common/BlockFermionicUniTensor/";

  Bond B1 = Bond(BD_IN, {Qs(0) >> 1, Qs(1) >> 1}, {Symmetry::FermionParity()});
  Bond B2 = Bond(BD_IN, {Qs(0), Qs(1)}, {1, 1}, {Symmetry::FermionParity()});
  Bond B12 = B1.combineBond(B2).redirect_();
  Bond B3 = Bond(BD_OUT, {Qs(0) >> 1, Qs(1) >> 1}, {Symmetry::FermionParity()});
  Bond B4 = Bond(BD_IN, {Qs(0) >> 1, Qs(1) >> 1}, {Symmetry::FermionParity()});
  UniTensor BFUT1 = UniTensor({B1, B2, B12}, {"a", "b", "c"});
  UniTensor BFUT2;
  UniTensor BFUT3 = UniTensor({B1, B2, B12, B3, B4}, {"a", "b", "c", "d", "e"});
  UniTensor BFUT3PERM =
    UniTensor({B3, B2, B4, B12, B1}, {"d", "b", "e", "c", "a"});  // permutation {3, 1, 4, 2, 0}

  // UniTensor BFUTfparfnum; // fpar x fnum symmetries

 protected:
  void SetUp() override {
    BFUT1.at({0, 0, 0}) = 1.;
    BFUT1.at({0, 0, 1}) = 2.;
    BFUT1.at({0, 1, 2}) = 3.;
    BFUT1.at({0, 1, 3}) = 4.;
    BFUT1.at({1, 0, 2}) = 5.;
    BFUT1.at({1, 0, 3}) = 6.;
    BFUT1.at({1, 1, 0}) = 7.;
    BFUT1.at({1, 1, 1}) = 8.;

    BFUT2 = BFUT1.clone();
    // BFUT2.permute_nosignflip_({2, 1, 0});
    BFUT2.Transpose_();
    BFUT2.set_rowrank_(1);

    BFUT3.at({0, 0, 0, 0, 0}) = 1.;
    BFUT3.at({0, 0, 1, 0, 0}) = 2.;
    BFUT3.at({0, 1, 2, 0, 0}) = 3.;
    BFUT3.at({0, 1, 3, 0, 0}) = 4.;
    BFUT3.at({1, 0, 2, 0, 0}) = 5.;
    BFUT3.at({1, 0, 3, 0, 0}) = 6.;
    BFUT3.at({1, 1, 0, 0, 0}) = 7.;
    BFUT3.at({1, 1, 1, 0, 0}) = 8.;

    BFUT3PERM.at({0, 0, 0, 0, 0}) = 1.;
    BFUT3PERM.at({0, 0, 0, 1, 0}) = 2.;
    BFUT3PERM.at({0, 1, 0, 2, 0}) = 3.;
    BFUT3PERM.at({0, 1, 0, 3, 0}) = 4.;
    BFUT3PERM.at({0, 0, 0, 2, 1}) = -5.;
    BFUT3PERM.at({0, 0, 0, 3, 1}) = -6.;
    BFUT3PERM.at({0, 1, 0, 0, 1}) = -7.;
    BFUT3PERM.at({0, 1, 0, 1, 1}) = -8.;

    // BFUTfparfnum = UniTensor::Load(data_dir + "ferm_fpar_fnum_F64.cytnx");
  }
  void TearDown() override {}
};

#endif
