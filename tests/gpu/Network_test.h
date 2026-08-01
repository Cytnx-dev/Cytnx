#ifndef CYTNX_TESTS_GPU_NETWORK_TEST_H_
#define CYTNX_TESTS_GPU_NETWORK_TEST_H_

#include "gtest/gtest.h"

#include "cytnx.hpp"
#include "gpu_test_tools.h"

namespace cytnx {
  namespace gpu_test {

    class NetworkTest : public ::testing::Test {
     public:
      // Network NEmpty;
      // Network NetFromFile = Network("testNet.net");
      Bond B1p = Bond(BD_IN, {Qs(-1), Qs(0), Qs(1)}, {2, 1, 2});
      Bond B2p = Bond(BD_OUT, {Qs(-1), Qs(0), Qs(1)}, {4, 3, 4});
      Bond B3p = Bond(BD_IN, {Qs(-1), Qs(0), Qs(2)}, {1, 1, 1});
      Bond B4p = Bond(BD_OUT, {Qs(-1), Qs(0), Qs(1)}, {2, 1, 2});
      UniTensor bkut1 = UniTensor({B1p, B4p}).to(Device.cuda);
      UniTensor bkut2 = UniTensor({B1p.redirect(), B2p, B3p, B4p}).to(Device.cuda);
      UniTensor bkut3 = UniTensor({B1p, B2p, B3p, B4p.redirect()}).to(Device.cuda);
      UniTensor ut1 = UniTensor(ones({5, 5}, Type.Double, Device.cuda));
      UniTensor ut2 = UniTensor(ones({5, 11, 3, 5}, Type.Double, Device.cuda));
      UniTensor ut3 = UniTensor(ones({5, 11, 3, 5}, Type.Double, Device.cuda));

      UniTensor utdnA =
        UniTensor(arange(0, 8, 1, Type.ComplexDouble, Device.cuda)).reshape({2, 2, 2});
      UniTensor utdnB = UniTensor(ones({2, 2}, Type.ComplexDouble, Device.cuda));
      UniTensor utdnC = UniTensor(eye(2, Type.ComplexDouble, Device.cuda));
      UniTensor utdnAns = UniTensor(zeros({2, 2, 2}, Type.ComplexDouble, Device.cuda));

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
        // NEmpty = Network();
        // NetFromFile = Network("testNet.net");
      }
    };

  }  // namespace gpu_test
}  // namespace cytnx
#endif  // CYTNX_TESTS_GPU_NETWORK_TEST_H_
