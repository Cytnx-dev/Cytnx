#include "Tensor_test.h"
#include "test_tools.h"

TEST_F(TensorTest, Constructor) {
  Tensor A;
  EXPECT_EQ(A.dtype(), Type.Void);
  EXPECT_EQ(A.device(), Device.cpu);
  EXPECT_EQ(A.shape().size(), 0);
  EXPECT_EQ(A.is_contiguous(), true);

  Tensor B({3, 4, 5});
  EXPECT_EQ(B.dtype(), Type.Double);
  EXPECT_EQ(B.device(), Device.cpu);
  EXPECT_EQ(B.shape().size(), 3);
  EXPECT_EQ(B.shape()[0], 3);
  EXPECT_EQ(B.shape()[1], 4);
  EXPECT_EQ(B.shape()[2], 5);
  EXPECT_EQ(B.is_contiguous(), true);

  Tensor C({3, 4, 5}, Type.Double);
  EXPECT_EQ(C.dtype(), Type.Double);
  EXPECT_EQ(C.device(), Device.cpu);
  EXPECT_EQ(C.shape().size(), 3);
  EXPECT_EQ(C.shape()[0], 3);
  EXPECT_EQ(C.shape()[1], 4);
  EXPECT_EQ(C.shape()[2], 5);
  EXPECT_EQ(C.is_contiguous(), true);
#ifdef UNI_GPU
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
#endif
}

TEST_F(TensorTest, CopyConstructor) {
#ifdef UNI_GPU
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
#endif

  Tensor D;
  D = tarcomplex3456;
  EXPECT_EQ(D.dtype(), Type.ComplexDouble);
  EXPECT_EQ(D.device(), Device.cpu);
  EXPECT_EQ(D.shape().size(), 4);
  EXPECT_EQ(D.shape()[0], 3);
  EXPECT_EQ(D.shape()[1], 4);
  EXPECT_EQ(D.shape()[2], 5);
  EXPECT_EQ(D.shape()[3], 6);
  EXPECT_EQ(D.is_contiguous(), true);

  Tensor E;
  E = tarcomplex3456.permute({1, 2, 3, 0});
  EXPECT_EQ(E.dtype(), Type.ComplexDouble);
  EXPECT_EQ(E.device(), Device.cpu);
  EXPECT_EQ(E.shape().size(), 4);
  EXPECT_EQ(E.shape()[0], 4);
  EXPECT_EQ(E.shape()[1], 5);
  EXPECT_EQ(E.shape()[2], 6);
  EXPECT_EQ(E.shape()[3], 3);
  EXPECT_EQ(E.is_contiguous(), false);
}

TEST_F(TensorTest, shape) {
  Tensor A({3, 4, 5}, Type.Double, Device.cpu, false);
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

  Tensor B({1}, Type.Double, Device.cpu, true);
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

  EXPECT_THROW(Tensor({0}, Type.Double, Device.cpu, true), std::logic_error);
}

TEST_F(TensorTest, permute) {
  Tensor A({3, 4, 5}, Type.Double, Device.cpu);
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

  Tensor B({1}, Type.Double, Device.cpu, true);
  EXPECT_EQ(B.shape().size(), 1);
  EXPECT_EQ(B.shape()[0], 1);

  B.permute_({0});
  EXPECT_EQ(B.shape().size(), 1);
  EXPECT_EQ(B.shape()[0], 1);

  B.permute_(0);
  EXPECT_EQ(B.shape().size(), 1);
  EXPECT_EQ(B.shape()[0], 1);

  EXPECT_THROW(Tensor({0}, Type.Double, Device.cpu, true), std::logic_error);
}

TEST_F(TensorTest, get) {
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
}

TEST_F(TensorTest, set) {
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

TEST_F(TensorTest, identity) {
  Tensor tn = cytnx::identity(2, Type.Double, Device.cpu);
  EXPECT_EQ(tn.shape().size(), 2);
  EXPECT_EQ(tn.shape()[0], 2);
  EXPECT_EQ(tn.shape()[1], 2);
  EXPECT_EQ(tn.is_contiguous(), true);
  EXPECT_EQ(tn.dtype(), Type.Double);
  EXPECT_EQ(tn.device(), Device.cpu);
  EXPECT_DOUBLE_EQ((double)tn(0, 0).item().real(), 1);
  EXPECT_DOUBLE_EQ((double)tn(1, 1).item().real(), 1);
  EXPECT_DOUBLE_EQ((double)tn(0, 1).item().real(), 0);
  EXPECT_DOUBLE_EQ((double)tn(1, 0).item().real(), 0);

  tn = cytnx::identity(3, Type.Double, Device.cpu);
  EXPECT_EQ(tn.shape().size(), 2);
  EXPECT_EQ(tn.shape()[0], 3);
  EXPECT_EQ(tn.shape()[1], 3);
  EXPECT_EQ(tn.is_contiguous(), true);
  EXPECT_EQ(tn.dtype(), Type.Double);
  EXPECT_EQ(tn.device(), Device.cpu);
  EXPECT_DOUBLE_EQ(tn.at<double>({0, 0}), 1);
  EXPECT_DOUBLE_EQ(tn.at<double>({1, 1}), 1);
  EXPECT_DOUBLE_EQ(tn.at<double>({0, 1}), 0);
  EXPECT_DOUBLE_EQ(tn.at<double>({1, 0}), 0);
  EXPECT_DOUBLE_EQ(tn.at<double>({2, 0}), 0);
  EXPECT_DOUBLE_EQ(tn.at<double>({2, 1}), 0);
  EXPECT_DOUBLE_EQ(tn.at<double>({0, 2}), 0);
  EXPECT_DOUBLE_EQ(tn.at<double>({1, 2}), 0);
  EXPECT_DOUBLE_EQ(tn.at<double>({2, 2}), 1);
}

TEST_F(TensorTest, eye) {
  Tensor tn = cytnx::eye(2, Type.Double, Device.cpu);
  EXPECT_EQ(tn.shape().size(), 2);
  EXPECT_EQ(tn.shape()[0], 2);
  EXPECT_EQ(tn.shape()[1], 2);
  EXPECT_EQ(tn.is_contiguous(), true);
  EXPECT_EQ(tn.dtype(), Type.Double);
  EXPECT_EQ(tn.device(), Device.cpu);
  EXPECT_DOUBLE_EQ((double)tn(0, 0).item().real(), 1);
  EXPECT_DOUBLE_EQ((double)tn(1, 1).item().real(), 1);
  EXPECT_DOUBLE_EQ((double)tn(0, 1).item().real(), 0);
  EXPECT_DOUBLE_EQ((double)tn(1, 0).item().real(), 0);

  tn = cytnx::eye(3, Type.Double, Device.cpu);
  EXPECT_EQ(tn.shape().size(), 2);
  EXPECT_EQ(tn.shape()[0], 3);
  EXPECT_EQ(tn.shape()[1], 3);
  EXPECT_EQ(tn.is_contiguous(), true);
  EXPECT_EQ(tn.dtype(), Type.Double);
  EXPECT_EQ(tn.device(), Device.cpu);
  EXPECT_DOUBLE_EQ(tn.at<double>({0, 0}), 1);
  EXPECT_DOUBLE_EQ(tn.at<double>({1, 1}), 1);
  EXPECT_DOUBLE_EQ(tn.at<double>({0, 1}), 0);
  EXPECT_DOUBLE_EQ(tn.at<double>({1, 0}), 0);
  EXPECT_DOUBLE_EQ(tn.at<double>({2, 0}), 0);
  EXPECT_DOUBLE_EQ(tn.at<double>({2, 1}), 0);
  EXPECT_DOUBLE_EQ(tn.at<double>({0, 2}), 0);
  EXPECT_DOUBLE_EQ(tn.at<double>({1, 2}), 0);
  EXPECT_DOUBLE_EQ(tn.at<double>({2, 2}), 1);
}
// TEST_F(TensorTest, approx_eq) {
//   cytnx::User_debug = true;
//   EXPECT_TRUE(tar3456.approx_eq(tar3456));
//   EXPECT_FALSE(tar345.approx_eq(tar345.permute(1, 0, 2)));
//   EXPECT_FALSE(tar3456.approx_eq(tarcomplex3456));
//   EXPECT_TRUE(tone3456.approx_eq(tone3456.astype(Type.ComplexFloat), 1e-5));
// }

TEST(Tensor, ItemDtypeMismatchThrows) {
  Tensor t = zeros({1}, Type.Float);
  EXPECT_THROW(t.item<double>(), std::logic_error);
  EXPECT_NO_THROW(t.item<float>());
}

TEST(Tensor, ReshapeRejectsMultipleUnknownDims) {
  Tensor t = zeros({12}, Type.Double);
  EXPECT_THROW(t.reshape({-1, -1}), std::logic_error);
  EXPECT_THROW(t.reshape_({-1, -1}), std::logic_error);
  EXPECT_THROW(t.reshape({-2, 6}), std::logic_error);
  // a single -1 must keep working
  Tensor r = t.reshape({3, -1});
  EXPECT_EQ(r.shape(), (std::vector<cytnx_uint64>{3, 4}));
}

TEST(Tensor, ReshapeRejectsZeroDimWithUnknownDim) {
  Tensor t = zeros({12}, Type.Double);
  // new_N == 0 previously fed a modulo/division by zero (UB / SIGFPE on x86)
  EXPECT_THROW(t.reshape({0, -1}), std::logic_error);
  EXPECT_THROW(t.reshape_({0, -1}), std::logic_error);
  EXPECT_THROW(t.reshape_({-1, 0}), std::logic_error);
}

TEST(Tensor, FailedReshapeLeavesShapeUnchanged) {
  Tensor t = zeros({12}, Type.Double);
  EXPECT_THROW(t.reshape_({7, 2}), std::logic_error);
  EXPECT_EQ(t.shape(), (std::vector<cytnx_uint64>{12}));
  EXPECT_THROW(t.reshape_({5, -1}), std::logic_error);
  EXPECT_EQ(t.shape(), (std::vector<cytnx_uint64>{12}));
  EXPECT_THROW(t.reshape_({0, -1}), std::logic_error);
  EXPECT_EQ(t.shape(), (std::vector<cytnx_uint64>{12}));
}

// NOTE: `Tensor b = a;` makes `b` an alias of the *same* Tensor_impl as `a`
// (Tensor's copy ctor just copies the intrusive_ptr), so `is(a.storage(),
// b.storage())` would trivially read the same Storage field twice and could
// never observe the #906 detach bug. To exercise a genuine "two independent
// Tensor handles sharing one Storage" scenario (e.g. what a view/slice would
// produce), we build `b` via Tensor::from_storage(a.storage()), which creates
// a brand-new Tensor_impl whose _storage is a copy of the Storage handle
// (same underlying Storage_base, distinct Tensor_impl).
TEST(Tensor, ScalarInplaceAddKeepsStorageSharing) {
  Tensor a = zeros({4}, Type.Double);
  Tensor b = Tensor::from_storage(a.storage());  // distinct Tensor_impl, shared Storage
  ASSERT_TRUE(is(a.storage(), b.storage()));
  a += 1.0;
  EXPECT_TRUE(is(a.storage(), b.storage()));
  EXPECT_DOUBLE_EQ(b.storage().at<double>(0), 1.0);
}

TEST(Tensor, ScalarInplaceOpsPreserveDtype) {
  Tensor a = ones({2}, Type.Float);
  a += 1.0;  // double scalar must not promote the tensor
  a -= 0.5;
  a *= 2.0;
  a /= 3.0;
  EXPECT_EQ(a.dtype(), Type.Float);
  EXPECT_FLOAT_EQ(a.storage().at<float>(0), 1.0f);
}

TEST(Tensor, ScalarInplaceRealPlusComplexThrows) {
  Tensor a = zeros({2}, Type.Double);
  EXPECT_THROW(a += cytnx_complex128(0, 1), std::logic_error);
  EXPECT_THROW(a -= cytnx_complex128(0, 1), std::logic_error);
  EXPECT_THROW(a *= cytnx_complex128(0, 1), std::logic_error);
  EXPECT_THROW(a /= cytnx_complex128(0, 1), std::logic_error);
}

TEST(Tensor, ScalarInplaceIntTensorTruncatesFractionalScalar) {
  Tensor a = ones({2}, Type.Int64);
  a += 2.7;  // was: promoted to Double (3.7); now: stays Int64, truncates
  EXPECT_EQ(a.dtype(), Type.Int64);
  EXPECT_EQ(a.storage().at<cytnx_int64>(0), 3);
}

// Mirrors the actual #906 report through public API: permute() produces a
// distinct Tensor_impl sharing the same Storage (Tensor_impl::permute does
// `out->_storage = this->_storage`) flagged non-contiguous, so this also
// exercises the non-contiguous scalar broadcast path of iAdd.
TEST(Tensor, ScalarInplaceOnPermutedViewMutatesSharedStorage) {
  Tensor a = zeros({2, 3}, Type.Double);
  Tensor v = a.permute({1, 0});  // distinct impl, shared storage, non-contiguous
  ASSERT_FALSE(v.is_contiguous());
  ASSERT_TRUE(is(a.storage(), v.storage()));
  v += 1.0;
  EXPECT_TRUE(is(a.storage(), v.storage()));
  EXPECT_DOUBLE_EQ(a.storage().at<double>(0), 1.0);
}

TEST(Tensor, ScalarInplaceSubMulDivKeepStorageSharing) {
  Tensor a = ones({3}, Type.Double);
  Tensor b = Tensor::from_storage(a.storage());
  a -= 0.5;
  a *= 4.0;
  a /= 2.0;
  EXPECT_TRUE(is(a.storage(), b.storage()));
  EXPECT_DOUBLE_EQ(b.storage().at<double>(2), 1.0);
}

TEST(Tensor, CytnxScalarInplaceAddKeepsStorageSharing) {
  Tensor a = zeros({2}, Type.Double);
  Tensor b = Tensor::from_storage(a.storage());
  a += Scalar(2.5);
  EXPECT_TRUE(is(a.storage(), b.storage()));
  EXPECT_DOUBLE_EQ(b.storage().at<double>(1), 2.5);
}

TEST(Tensor, ScalarInplaceRealTimesComplexErrorNamesOperator) {
  Tensor a = zeros({2}, Type.Double);
  try {
    a *= cytnx_complex128(0, 1);
    FAIL() << "expected real *= complex to throw";
  } catch (const std::logic_error &e) {
    EXPECT_NE(std::string(e.what()).find("*="), std::string::npos) << e.what();
  }
  try {
    a /= cytnx_complex128(0, 1);
    FAIL() << "expected real /= complex to throw";
  } catch (const std::logic_error &e) {
    EXPECT_NE(std::string(e.what()).find("/="), std::string::npos) << e.what();
  }
}
