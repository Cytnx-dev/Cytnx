#include "Tensor_test.h"
#include "test_tools.h"

TEST_F(TensorTest, gpu_Constructor) {
  Tensor D({3, 4, 5}, Type.Double, Device.cuda);
  EXPECT_EQ(D.dtype(), Type.Double);
  EXPECT_EQ(D.device(), Device.cuda);
  EXPECT_EQ(D.shape().size(), 3);
  EXPECT_EQ(D.shape()[0], 3);
  EXPECT_EQ(D.shape()[1], 4);
  EXPECT_EQ(D.shape()[2], 5);
  EXPECT_EQ(D.is_contiguous(), true);

  Tensor E({3, 4, 5}, Type.Double, Device.cuda, true);
  EXPECT_EQ(E.dtype(), Type.Double);
  EXPECT_EQ(E.device(), Device.cuda);
  EXPECT_EQ(E.shape().size(), 3);
  EXPECT_EQ(E.shape()[0], 3);
  EXPECT_EQ(E.shape()[1], 4);
  EXPECT_EQ(E.shape()[2], 5);
  EXPECT_EQ(E.is_contiguous(), true);

  Tensor F({3, 4, 5}, Type.Double, Device.cuda, false);
  EXPECT_EQ(F.dtype(), Type.Double);
  EXPECT_EQ(F.device(), Device.cuda);
  EXPECT_EQ(F.shape().size(), 3);
  EXPECT_EQ(F.shape()[0], 3);
  EXPECT_EQ(F.shape()[1], 4);
  EXPECT_EQ(F.shape()[2], 5);
  EXPECT_EQ(F.is_contiguous(), true);
}

TEST_F(TensorTest, gpu_CopyConstructor) {
  Tensor A({3, 4, 5}, Type.Double, Device.cuda, false);
  Tensor B(A);
  EXPECT_EQ(B.dtype(), Type.Double);
  EXPECT_EQ(B.device(), Device.cuda);
  EXPECT_EQ(B.shape().size(), 3);
  EXPECT_EQ(B.shape()[0], 3);
  EXPECT_EQ(B.shape()[1], 4);
  EXPECT_EQ(B.shape()[2], 5);
  EXPECT_EQ(B.is_contiguous(), true);

  Tensor C = A;
  EXPECT_EQ(C.dtype(), Type.Double);
  EXPECT_EQ(C.device(), Device.cuda);
  EXPECT_EQ(C.shape().size(), 3);
  EXPECT_EQ(C.shape()[0], 3);
  EXPECT_EQ(C.shape()[1], 4);
  EXPECT_EQ(C.shape()[2], 5);
  EXPECT_EQ(C.is_contiguous(), true);
}

TEST_F(TensorTest, gpu_shape) {
  Tensor A({3, 4, 5}, Type.Double, Device.cuda, false);
  EXPECT_EQ(A.shape().size(), 3);
  EXPECT_EQ(A.shape()[0], 3);
  EXPECT_EQ(A.shape()[1], 4);
  EXPECT_EQ(A.shape()[2], 5);
  EXPECT_EQ(A.is_contiguous(), true);

  A.reshape_({4, 5, 3});
  EXPECT_EQ(A.shape().size(), 3);
  EXPECT_EQ(A.shape()[0], 4);
  EXPECT_EQ(A.shape()[1], 5);
  EXPECT_EQ(A.shape()[2], 3);

  A.reshape_(3, 4, 5);
  EXPECT_EQ(A.shape().size(), 3);
  EXPECT_EQ(A.shape()[0], 3);
  EXPECT_EQ(A.shape()[1], 4);
  EXPECT_EQ(A.shape()[2], 5);

  auto tmp = A.reshape({4, 3, 5});
  EXPECT_EQ(tmp.shape().size(), 3);
  EXPECT_EQ(tmp.shape()[0], 4);
  EXPECT_EQ(tmp.shape()[1], 3);
  EXPECT_EQ(tmp.shape()[2], 5);

  tmp = A.reshape(4, 5, 3);
  EXPECT_EQ(tmp.shape().size(), 3);
  EXPECT_EQ(tmp.shape()[0], 4);
  EXPECT_EQ(tmp.shape()[1], 5);
  EXPECT_EQ(tmp.shape()[2], 3);

  Tensor B({1}, Type.Double, Device.cuda, true);
  EXPECT_EQ(B.shape().size(), 1);
  EXPECT_EQ(B.shape()[0], 1);

  B.reshape_({1, 1});
  EXPECT_EQ(B.shape().size(), 2);
  EXPECT_EQ(B.shape()[0], 1);
  EXPECT_EQ(B.shape()[1], 1);

  B.reshape_({1});
  EXPECT_EQ(B.shape().size(), 1);
  EXPECT_EQ(B.shape()[0], 1);

  B.reshape_(1, 1, 1);
  EXPECT_EQ(B.shape().size(), 3);
  EXPECT_EQ(B.shape()[0], 1);
  EXPECT_EQ(B.shape()[1], 1);
  EXPECT_EQ(B.shape()[2], 1);

  EXPECT_THROW(Tensor({0}, Type.Double, Device.cuda, true), std::logic_error);
}

TEST_F(TensorTest, gpu_permute) {
  Tensor A({3, 4, 5}, Type.Double, Device.cuda);
  EXPECT_EQ(A.shape().size(), 3);
  EXPECT_EQ(A.shape()[0], 3);
  EXPECT_EQ(A.shape()[1], 4);
  EXPECT_EQ(A.shape()[2], 5);
  EXPECT_EQ(A.is_contiguous(), true);

  A.permute_({2, 1, 0});
  EXPECT_EQ(A.shape().size(), 3);
  EXPECT_EQ(A.shape()[0], 5);
  EXPECT_EQ(A.shape()[1], 4);
  EXPECT_EQ(A.shape()[2], 3);
  EXPECT_EQ(A.is_contiguous(), false);

  A.permute_(2, 1, 0);
  EXPECT_EQ(A.shape().size(), 3);
  EXPECT_EQ(A.shape()[0], 3);
  EXPECT_EQ(A.shape()[1], 4);
  EXPECT_EQ(A.shape()[2], 5);
  EXPECT_EQ(A.is_contiguous(), true);

  auto tmp = A.permute({2, 0, 1});
  EXPECT_EQ(tmp.shape().size(), 3);
  EXPECT_EQ(tmp.shape()[0], 5);
  EXPECT_EQ(tmp.shape()[1], 3);
  EXPECT_EQ(tmp.shape()[2], 4);
  EXPECT_EQ(tmp.is_contiguous(), false);

  tmp = A.permute(2, 0, 1);
  EXPECT_EQ(tmp.shape().size(), 3);
  EXPECT_EQ(tmp.shape()[0], 5);
  EXPECT_EQ(tmp.shape()[1], 3);
  EXPECT_EQ(tmp.shape()[2], 4);
  EXPECT_EQ(tmp.is_contiguous(), false);

  Tensor B({1}, Type.Double, Device.cuda, true);
  EXPECT_EQ(B.shape().size(), 1);
  EXPECT_EQ(B.shape()[0], 1);

  B.permute_({0});
  EXPECT_EQ(B.shape().size(), 1);
  EXPECT_EQ(B.shape()[0], 1);

  B.permute_(0);
  EXPECT_EQ(B.shape().size(), 1);
  EXPECT_EQ(B.shape()[0], 1);

  EXPECT_THROW(Tensor({0}, Type.Double, Device.cuda, true), std::logic_error);
}

TEST_F(TensorTest, gpu_get) {
  Tensor tmp = tzero3456(":", ":", ":", ":");
  EXPECT_EQ(tmp.shape().size(), 4);
  EXPECT_EQ(tmp.shape()[0], 3);
  EXPECT_EQ(tmp.shape()[1], 4);
  EXPECT_EQ(tmp.shape()[2], 5);
  EXPECT_EQ(tmp.shape()[3], 6);
  EXPECT_EQ(tmp.is_contiguous(), true);

  tmp = tzero3456(0, ":", ":", ":");
  EXPECT_EQ(tmp.shape().size(), 3);
  EXPECT_EQ(tmp.shape()[0], 4);
  EXPECT_EQ(tmp.shape()[1], 5);
  EXPECT_EQ(tmp.shape()[2], 6);
  EXPECT_EQ(tmp.is_contiguous(), true);

  tmp = tzero3456(0, 0, ":", ":");
  EXPECT_EQ(tmp.shape().size(), 2);
  EXPECT_EQ(tmp.shape()[0], 5);
  EXPECT_EQ(tmp.shape()[1], 6);
  EXPECT_EQ(tmp.is_contiguous(), true);

  tmp = tzero3456(1, 0, 0, ":");
  EXPECT_EQ(tmp.shape().size(), 1);
  EXPECT_EQ(tmp.shape()[0], 6);
  EXPECT_EQ(tmp.is_contiguous(), true);

  tmp = tzero3456(0, 1, 4, 4);
  EXPECT_EQ(tmp.shape().size(), 1);
  EXPECT_EQ(tmp.shape()[0], 1);
  EXPECT_EQ(tmp.is_contiguous(), true);

  tmp = tarcomplex3456.get({Accessor::all(), Accessor::all(), Accessor::all(), Accessor::all()});
  EXPECT_EQ(tmp.shape().size(), 4);
  EXPECT_EQ(tmp.shape()[0], 3);
  EXPECT_EQ(tmp.shape()[1], 4);
  EXPECT_EQ(tmp.shape()[2], 5);
  EXPECT_EQ(tmp.shape()[3], 6);
  EXPECT_EQ(tmp.is_contiguous(), true);

  tmp = tarcomplex3456(":1", ":", ":", ":");
  EXPECT_EQ(tmp.shape().size(), 4);
  EXPECT_EQ(tmp.shape()[0], 1);
  EXPECT_EQ(tmp.shape()[1], 4);
  EXPECT_EQ(tmp.shape()[2], 5);
  EXPECT_EQ(tmp.shape()[3], 6);
  EXPECT_EQ(tmp.is_contiguous(), true);

  tmp = tarcomplex3456("0:1", ":", ":", ":");
  EXPECT_EQ(tmp.shape().size(), 4);
  EXPECT_EQ(tmp.shape()[0], 1);
  EXPECT_EQ(tmp.shape()[1], 4);
  EXPECT_EQ(tmp.shape()[2], 5);
  EXPECT_EQ(tmp.shape()[3], 6);
  EXPECT_EQ(tmp.is_contiguous(), true);

  tmp = tarcomplex3456("0:2", ":", ":", ":");
  EXPECT_EQ(tmp.shape().size(), 4);
  EXPECT_EQ(tmp.shape()[0], 2);
  EXPECT_EQ(tmp.shape()[1], 4);
  EXPECT_EQ(tmp.shape()[2], 5);
  EXPECT_EQ(tmp.shape()[3], 6);

  tmp = tarcomplex3456("1:2", ":", ":", ":");
  EXPECT_EQ(tmp.shape().size(), 4);
  EXPECT_EQ(tmp.shape()[0], 1);
  EXPECT_EQ(tmp.shape()[1], 4);
  EXPECT_EQ(tmp.shape()[2], 5);
  EXPECT_EQ(tmp.shape()[3], 6);
  TestTools::AreNearlyEqTensor(tmp, tslice1);

  tmp = tar3456.permute({2, 0, 1, 3});
  EXPECT_EQ(tmp.shape().size(), 4);
  EXPECT_EQ(tmp.shape()[0], 5);
  EXPECT_EQ(tmp.shape()[1], 3);
  EXPECT_EQ(tmp.shape()[2], 4);
  EXPECT_EQ(tmp.shape()[3], 6);
  EXPECT_EQ(tmp.is_contiguous(), false);
  cout << tmp(0, ":", ":", 0) << endl;
  TestTools::AreNearlyEqTensor(tmp(0, ":", ":", 0), arange(12).reshape({3, 4}), 1e-5);
  TestTools::AreNearlyEqTensor(tmp(":", 0, 0, ":"), arange(30).reshape({5, 6}), 1e-5);
  TestTools::AreNearlyEqTensor(tmp(":", 0, 0, "4:6"), arange(20, 30).reshape({5, 2}), 1e-5);
  TestTools::AreNearlyEqTensor(tmp(":", "2:4", ":", ":"),
                               arange(3 * 2 * 5 * 6, 3 * 4 * 5 * 6).reshape({3, 2, 5, 6}), 1e-5);
}

TEST_F(TensorTest, gpu_set) {
  auto tmp = tar3456.clone();
  tar3456(1, 2, 3, 4) = -1999;
  for (size_t i = 0; i < 3; i++)
    for (size_t j = 0; j < 4; j++)
      for (size_t k = 0; k < 5; k++)
        for (size_t l = 0; l < 6; l++)
          if (i == 1 && j == 2 && k == 3 && l == 4) {
            EXPECT_EQ(tar3456(i, j, k, l).item().real(), -1999);
          } else {
            EXPECT_EQ(tar3456(i, j, k, l).item().real(), tmp(i, j, k, l).item().real());
          }
}

// TEST_F(TensorTest, gpu_approx_eq) {
//   cytnx::User_debug = true;
//   EXPECT_TRUE(tar3456.approx_eq(tar3456));
//   EXPECT_FALSE(tar345.approx_eq(tar345.permute(1, 0, 2)));
//   EXPECT_FALSE(tar3456.approx_eq(tarcomplex3456));
//   EXPECT_TRUE(tone3456.approx_eq(tone3456.astype(Type.ComplexFloat), 1e-5));
// }
