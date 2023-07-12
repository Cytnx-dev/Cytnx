#ifndef _H_TENSOR_TEST
#define _H_TENSOR_TEST

#include "cytnx.hpp"
#include <gtest/gtest.h>

using namespace cytnx;
using namespace std;
class TensorTest : public ::testing::Test {
 public:
  std::string data_dir = "../../tests/test_data_base/common/Tensor/";

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
    tzero345 = zeros(3 * 4 * 5).reshape({3, 4, 5}).astype(Type.ComplexDouble);
    tone345 = ones(3 * 4 * 5).reshape({3, 4, 5}).astype(Type.ComplexDouble);
    tar345 = arange(3 * 4 * 5).reshape({3, 4, 5}).astype(Type.ComplexDouble);
    tzero3456 = zeros(3 * 4 * 5 * 6).reshape({3, 4, 5, 6}).astype(Type.ComplexDouble);
    tone3456 = ones(3 * 4 * 5 * 6).reshape({3, 4, 5, 6}).astype(Type.ComplexDouble);
    tar3456 = arange(3 * 4 * 5 * 6).reshape({3, 4, 5, 6}).astype(Type.ComplexDouble);
    tarcomplex345 = arange(3 * 4 * 5).astype(Type.ComplexDouble);
    for (size_t i = 0; i < 3 * 4 * 5; i++) tarcomplex345.at({i}) = cytnx_complex128(i, i);
    tarcomplex345 = tarcomplex345.reshape({3, 4, 5}).astype(Type.ComplexDouble);
    tarcomplex3456 = arange(3 * 4 * 5 * 6).astype(Type.ComplexDouble);
    for (size_t i = 0; i < 3 * 4 * 5 * 6; i++) tarcomplex3456.at({i}) = cytnx_complex128(i, i);
    tarcomplex3456 = tarcomplex3456.reshape({3, 4, 5, 6}).astype(Type.ComplexDouble);

    tslice1 = Tensor::Load(data_dir + "tensorslice1.cytn");
  }
  void TearDown() override {}
};

#endif
