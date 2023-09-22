#ifndef _H_Network_test
#define _H_Network_test

#include "cytnx.hpp"
#include <gtest/gtest.h>
#include "test_tools.h"

using namespace cytnx;
using namespace TestTools;

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

  UniTensor utdnA = UniTensor(arange(0, 8, 1, Type.ComplexDouble)).reshape({2, 2, 2});
  UniTensor utdnB = UniTensor(ones({2, 2}, Type.ComplexDouble));
  UniTensor utdnC = UniTensor(eye(2, Type.ComplexDouble));
  UniTensor utdnAns = UniTensor(zeros({2, 2, 2}, Type.ComplexDouble));

 protected:
  void SetUp() override {
    utdnAns.at({0, 0, 0}) = 1;
    utdnAns.at({0, 0, 1}) = 1;
    utdnAns.at({0, 1, 0}) = 5;
    utdnAns.at({0, 1, 1}) = 5;
    utdnAns.at({1, 0, 0}) = 9;
    utdnAns.at({1, 0, 1}) = 9;
    utdnAns.at({1, 1, 0}) = 13;
    utdnAns.at({1, 1, 1}) = 13;
  }
  void TearDown() override {
    // NEmpty = cytnx::Network();
    // NetFromFile = cytnx::Network("testNet.net");
  }
};

#endif
