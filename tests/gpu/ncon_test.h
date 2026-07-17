#ifndef CYTNX_TESTS_GPU_NCON_TEST_H_
#define CYTNX_TESTS_GPU_NCON_TEST_H_

#include "gtest/gtest.h"

#include "cytnx.hpp"
#include "gpu_test_tools.h"
namespace cytnx {
  namespace test {

    class NconTest : public ::testing::Test {
     public:
      // std::pair<std::vector<UniTensor>, std::vector<std::vector<cytnx_int64>>>
      // input;
      UniTensor utdnA =
        UniTensor(arange(0, 8, 1, Type.ComplexDouble, Device.cuda)).reshape({2, 2, 2});
      ;
      UniTensor utdnB = UniTensor(ones({2, 2}, Type.ComplexDouble, Device.cuda));
      ;
      UniTensor utdnC = UniTensor(eye(2, Type.ComplexDouble, Device.cuda));
      ;
      UniTensor utdnAns = UniTensor(zeros({2, 2, 2}, Type.ComplexDouble, Device.cuda));
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

  }  // namespace test
}  // namespace cytnx
#endif  // CYTNX_TESTS_GPU_NCON_TEST_H_
