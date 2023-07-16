#ifndef _H_Network_test
#define _H_Network_test

#include "cytnx.hpp"
#include <gtest/gtest.h>

using namespace cytnx;

class NetworkTest : public ::testing::Test {
 public:
  // cytnx::Network NEmpty;
  // cytnx::Network NetFromFile = cytnx::Network("testNet.net");
  Bond B1p = Bond(BD_IN, {Qs(-1), Qs(0), Qs(1)}, {2, 1, 2});
  Bond B2p = Bond(BD_OUT, {Qs(-1), Qs(0), Qs(1)}, {4, 3, 4});
  Bond B3p = Bond(BD_IN, {Qs(-1), Qs(0), Qs(2)}, {1, 1, 1});
  Bond B4p = Bond(BD_OUT, {Qs(-1), Qs(0), Qs(1)}, {2, 1, 2});
  UniTensor bkut1 = UniTensor({B1p, B4p});
  UniTensor bkut2 = UniTensor({B1p.redirect(), B2p, B3p, B4p});
  UniTensor bkut3 = UniTensor({B1p, B2p, B3p, B4p.redirect()});
  UniTensor ut1 = UniTensor(ones({5, 5}));
  UniTensor ut2 = UniTensor(ones({5, 11, 3, 5}));
  UniTensor ut3 = UniTensor(ones({5, 11, 3, 5}));

 protected:
  void SetUp() override {}
  void TearDown() override {
    // NEmpty = cytnx::Network();
    // NetFromFile = cytnx::Network("testNet.net");
  }
};

#endif
