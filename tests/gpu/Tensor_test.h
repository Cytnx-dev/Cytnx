#ifndef CYTNX_TESTS_GPU_TENSOR_TEST_H_
#define CYTNX_TESTS_GPU_TENSOR_TEST_H_

#include <gtest/gtest.h>

#include "cytnx.hpp"

namespace cytnx {
  namespace gpu_test {

    class TensorTest : public ::testing::Test {
     public:
      std::string data_dir = CYTNX_TEST_DATA_DIR "/common/Tensor/";

      Tensor tzero345;
      Tensor tone345;
      Tensor tar345;
      Tensor tzero3456;
      Tensor tone3456;
      Tensor tar3456;
      Tensor tarcomplex345;
      Tensor tarcomplex3456;

      Tensor tslice1;

     protected:
      void SetUp() override {
        using std::complex_literals::operator""i;

        tzero345 = zeros({3, 4, 5}, Type.ComplexDouble, Device.cuda);
        tone345 = ones({3, 4, 5}, Type.ComplexDouble, Device.cuda);
        tar345 = arange(0, 3 * 4 * 5, 1, Type.ComplexDouble, Device.cuda).reshape({3, 4, 5});
        tzero3456 = zeros({3, 4, 5, 6}, Type.ComplexDouble, Device.cuda);
        tone3456 = ones({3, 4, 5, 6}, Type.ComplexDouble, Device.cuda);
        tar3456 =
          arange(0, 3 * 4 * 5 * 6, 1, Type.ComplexDouble, Device.cuda).reshape({3, 4, 5, 6});
        tarcomplex345 = ((1.0 + 1.0i) * arange(0, 3 * 4 * 5, 1, Type.ComplexDouble, Device.cuda))
                          .reshape({3, 4, 5});
        tarcomplex3456 =
          ((1.0 + 1.0i) * arange(0, 3 * 4 * 5 * 6, 1, Type.ComplexDouble, Device.cuda))
            .reshape({3, 4, 5, 6});

        tslice1 = Tensor::Load(data_dir + "tensorslice1.cytn").to(Device.cuda);
      }
      void TearDown() override {}
    };

  }  // namespace gpu_test
}  // namespace cytnx
#endif  // CYTNX_TESTS_GPU_TENSOR_TEST_H_
