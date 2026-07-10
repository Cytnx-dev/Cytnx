#include "Tensor_test.h"
#include "gpu_test_tools.h"

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
  EXPECT_EQ(tmp.dtype(), Type.ComplexDouble);
  EXPECT_EQ(tmp.device(), Device.cuda);
  EXPECT_TRUE(tmp.is_scalar());
  EXPECT_EQ(tmp.storage().size(), 1);
  EXPECT_EQ(tmp.to(Device.cpu).item<cytnx_complex128>(), cytnx_complex128(0, 0));
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
  std::cout << tmp(0, ":", ":", 0) << std::endl;
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

TEST_F(TensorTest, gpu_RankZeroScalarAccessAndBroadcast) {
  Tensor scalar(std::vector<cytnx_uint64>{}, Type.Double, Device.cuda);
  scalar.set(std::vector<Accessor>{}, 3.25);

  Tensor selected = scalar.get(std::vector<Accessor>{});
  EXPECT_EQ(selected.device(), Device.cuda);
  EXPECT_TRUE(selected.is_scalar());
  EXPECT_DOUBLE_EQ(selected.to(Device.cpu).item<double>(), 3.25);

  Tensor replacement(std::vector<cytnx_uint64>{}, Type.Double, Device.cuda);
  replacement.set(std::vector<Accessor>{}, -2.0);
  scalar.set(std::vector<Accessor>{}, replacement);
  EXPECT_DOUBLE_EQ(scalar.to(Device.cpu).item<double>(), -2.0);

  scalar.set(std::vector<Accessor>{}, 2.0);
  Tensor vec = arange(3).astype(Type.Double).to(Device.cuda);
  Tensor out = scalar + vec;
  Tensor host = out.to(Device.cpu);
  EXPECT_EQ(host.shape(), (std::vector<cytnx_uint64>{3}));
  EXPECT_DOUBLE_EQ(host.at<double>({0}), 2.0);
  EXPECT_DOUBLE_EQ(host.at<double>({1}), 3.0);
  EXPECT_DOUBLE_EQ(host.at<double>({2}), 4.0);

  out = vec + scalar;
  host = out.to(Device.cpu);
  EXPECT_EQ(host.shape(), (std::vector<cytnx_uint64>{3}));
  EXPECT_DOUBLE_EQ(host.at<double>({0}), 2.0);
  EXPECT_DOUBLE_EQ(host.at<double>({1}), 3.0);
  EXPECT_DOUBLE_EQ(host.at<double>({2}), 4.0);

  out = scalar - vec;
  host = out.to(Device.cpu);
  EXPECT_EQ(host.shape(), (std::vector<cytnx_uint64>{3}));
  EXPECT_DOUBLE_EQ(host.at<double>({0}), 2.0);
  EXPECT_DOUBLE_EQ(host.at<double>({1}), 1.0);
  EXPECT_DOUBLE_EQ(host.at<double>({2}), 0.0);

  out = vec - scalar;
  host = out.to(Device.cpu);
  EXPECT_EQ(host.shape(), (std::vector<cytnx_uint64>{3}));
  EXPECT_DOUBLE_EQ(host.at<double>({0}), -2.0);
  EXPECT_DOUBLE_EQ(host.at<double>({1}), -1.0);
  EXPECT_DOUBLE_EQ(host.at<double>({2}), 0.0);

  Tensor scalar2(std::vector<cytnx_uint64>{}, Type.Double, Device.cuda);
  scalar2.set(std::vector<Accessor>{}, 5.0);
  out = scalar * scalar2;
  host = out.to(Device.cpu);
  EXPECT_TRUE(host.is_scalar());
  EXPECT_DOUBLE_EQ(host.item<double>(), 10.0);

  out = scalar * vec;
  host = out.to(Device.cpu);
  EXPECT_EQ(host.shape(), (std::vector<cytnx_uint64>{3}));
  EXPECT_DOUBLE_EQ(host.at<double>({0}), 0.0);
  EXPECT_DOUBLE_EQ(host.at<double>({1}), 2.0);
  EXPECT_DOUBLE_EQ(host.at<double>({2}), 4.0);

  out = vec * scalar;
  host = out.to(Device.cpu);
  EXPECT_EQ(host.shape(), (std::vector<cytnx_uint64>{3}));
  EXPECT_DOUBLE_EQ(host.at<double>({0}), 0.0);
  EXPECT_DOUBLE_EQ(host.at<double>({1}), 2.0);
  EXPECT_DOUBLE_EQ(host.at<double>({2}), 4.0);

  Tensor denom = arange(1, 4, 1, Type.Double).to(Device.cuda);
  out = scalar / denom;
  host = out.to(Device.cpu);
  EXPECT_EQ(host.shape(), (std::vector<cytnx_uint64>{3}));
  EXPECT_DOUBLE_EQ(host.at<double>({0}), 2.0);
  EXPECT_DOUBLE_EQ(host.at<double>({1}), 1.0);
  EXPECT_DOUBLE_EQ(host.at<double>({2}), 2.0 / 3.0);

  out = denom / scalar;
  host = out.to(Device.cpu);
  EXPECT_EQ(host.shape(), (std::vector<cytnx_uint64>{3}));
  EXPECT_DOUBLE_EQ(host.at<double>({0}), 0.5);
  EXPECT_DOUBLE_EQ(host.at<double>({1}), 1.0);
  EXPECT_DOUBLE_EQ(host.at<double>({2}), 1.5);

  Tensor mod_scalar(std::vector<cytnx_uint64>{}, Type.Int64, Device.cuda);
  mod_scalar.set(std::vector<Accessor>{}, cytnx_int64(5));
  Tensor mod_vec = arange(2, 5, 1, Type.Int64).to(Device.cuda);

  out = linalg::Mod(mod_scalar, mod_vec);
  host = out.to(Device.cpu);
  EXPECT_EQ(host.shape(), (std::vector<cytnx_uint64>{3}));
  EXPECT_EQ(host.at<cytnx_int64>({0}), 1);
  EXPECT_EQ(host.at<cytnx_int64>({1}), 2);
  EXPECT_EQ(host.at<cytnx_int64>({2}), 1);

  out = linalg::Mod(mod_vec, mod_scalar);
  host = out.to(Device.cpu);
  EXPECT_EQ(host.shape(), (std::vector<cytnx_uint64>{3}));
  EXPECT_EQ(host.at<cytnx_int64>({0}), 2);
  EXPECT_EQ(host.at<cytnx_int64>({1}), 3);
  EXPECT_EQ(host.at<cytnx_int64>({2}), 4);

  Tensor mod_rhs(std::vector<cytnx_uint64>{}, Type.Int64, Device.cuda);
  mod_rhs.set(std::vector<Accessor>{}, cytnx_int64(3));
  out = linalg::Mod(mod_scalar, mod_rhs);
  host = out.to(Device.cpu);
  EXPECT_TRUE(host.is_scalar());
  EXPECT_EQ(host.item<cytnx_int64>(), 2);

  Tensor cmp = linalg::Cpr(scalar, vec);
  Tensor cmp_host = cmp.to(Device.cpu);
  EXPECT_EQ(cmp_host.shape(), (std::vector<cytnx_uint64>{3}));
  EXPECT_FALSE(cmp_host.at<cytnx_bool>({0}));
  EXPECT_FALSE(cmp_host.at<cytnx_bool>({1}));
  EXPECT_TRUE(cmp_host.at<cytnx_bool>({2}));

  cmp = linalg::Cpr(scalar, scalar);
  cmp_host = cmp.to(Device.cpu);
  EXPECT_TRUE(cmp_host.is_scalar());
  EXPECT_TRUE(cmp_host.item<cytnx_bool>());

  cmp = linalg::Cpr(scalar, scalar2);
  cmp_host = cmp.to(Device.cpu);
  EXPECT_TRUE(cmp_host.is_scalar());
  EXPECT_FALSE(cmp_host.item<cytnx_bool>());

  Tensor dot = linalg::Vectordot(vec, vec, false);
  Tensor dot_host = dot.to(Device.cpu);
  EXPECT_TRUE(dot_host.is_scalar());
  EXPECT_EQ(dot_host.shape().size(), 0);
  EXPECT_DOUBLE_EQ(dot_host.item<double>(), 5.0);

  host = (dot * vec).to(Device.cpu);
  EXPECT_EQ(host.shape(), (std::vector<cytnx_uint64>{3}));
  EXPECT_DOUBLE_EQ(host.at<double>({0}), 0.0);
  EXPECT_DOUBLE_EQ(host.at<double>({1}), 5.0);
  EXPECT_DOUBLE_EQ(host.at<double>({2}), 10.0);

  Tensor shape_one = zeros({1}, Type.Double, Device.cuda);
  shape_one.set(std::vector<Accessor>{Accessor(0)}, 4.0);
  out = scalar + shape_one;
  host = out.to(Device.cpu);
  EXPECT_EQ(host.shape(), (std::vector<cytnx_uint64>{1}));
  EXPECT_FALSE(host.is_scalar());
  EXPECT_DOUBLE_EQ(host.at<double>({0}), 6.0);

  out = shape_one - scalar;
  host = out.to(Device.cpu);
  EXPECT_EQ(host.shape(), (std::vector<cytnx_uint64>{1}));
  EXPECT_FALSE(host.is_scalar());
  EXPECT_DOUBLE_EQ(host.at<double>({0}), 2.0);

  Tensor complex_scalar(std::vector<cytnx_uint64>{}, Type.ComplexDouble, Device.cuda);
  complex_scalar.set(std::vector<Accessor>{}, cytnx_complex128(4.0, 0.0));
  cmp = linalg::Cpr(complex_scalar, shape_one);
  cmp_host = cmp.to(Device.cpu);
  EXPECT_EQ(cmp_host.shape(), (std::vector<cytnx_uint64>{1}));
  EXPECT_TRUE(cmp_host.at<cytnx_bool>({0}));

  Tensor float_mod_scalar(std::vector<cytnx_uint64>{}, Type.Float, Device.cuda);
  float_mod_scalar.set(std::vector<Accessor>{}, cytnx_float(5.5));
  Tensor float_mod_vec = arange(2, 5, 1, Type.Float).to(Device.cuda);
  out = linalg::Mod(float_mod_scalar, float_mod_vec);
  host = out.to(Device.cpu);
  EXPECT_EQ(host.shape(), (std::vector<cytnx_uint64>{3}));
  EXPECT_FLOAT_EQ(host.at<cytnx_float>({0}), 1.5f);
  EXPECT_FLOAT_EQ(host.at<cytnx_float>({1}), 2.5f);
  EXPECT_FLOAT_EQ(host.at<cytnx_float>({2}), 1.5f);

  EXPECT_THROW((void)(shape_one + vec), std::logic_error);
  EXPECT_THROW((void)(vec + shape_one), std::logic_error);
  EXPECT_THROW(vec += shape_one, std::logic_error);
}

// TEST_F(TensorTest, gpu_approx_eq) {
//   cytnx::User_debug = true;
//   EXPECT_TRUE(tar3456.approx_eq(tar3456));
//   EXPECT_FALSE(tar345.approx_eq(tar345.permute(1, 0, 2)));
//   EXPECT_FALSE(tar3456.approx_eq(tarcomplex3456));
//   EXPECT_TRUE(tone3456.approx_eq(tone3456.astype(Type.ComplexFloat), 1e-5));
// }

// ---------------------------------------------------------------------------
// GPU scalar in-place arithmetic (issue #988). These mirror the CPU-side
// Tensor.ScalarInplace* tests on CUDA tensors and pin the non-contiguous
// *= / /= behaviour on the GPU path. Test names avoid underscores (#857).
// ---------------------------------------------------------------------------

// Regression (#988): scalar *= on a *non-contiguous* GPU tensor used to throw
// "[iMul][on GPU/CUDA] ... must be contiguous". It must now scale in place,
// keeping the shared storage, with the layout mappers irrelevant to a
// broadcast scalar.
TEST(Tensor, GpuScalarInplaceNoncontigMul) {
  Tensor a = arange(6).reshape({2, 3}).to(Device.cuda);  // Double, contiguous
  Tensor v = a.permute({1, 0});  // distinct impl, shared storage, non-contiguous
  ASSERT_FALSE(v.is_contiguous());
  ASSERT_TRUE(is(a.storage(), v.storage()));
  v *= 2.0;  // must not throw
  EXPECT_TRUE(is(a.storage(), v.storage()));  // still in place / shared
  Tensor a_cpu = a.to(Device.cpu);
  for (cytnx_uint64 i = 0; i < 6; i++)
    EXPECT_DOUBLE_EQ(a_cpu.storage().at<cytnx_double>(i), 2.0 * i);
}

// Regression (#988): scalar /= on a non-contiguous GPU tensor.
TEST(Tensor, GpuScalarInplaceNoncontigDiv) {
  Tensor a = arange(6).reshape({2, 3}).to(Device.cuda);
  Tensor v = a.permute({1, 0});
  ASSERT_FALSE(v.is_contiguous());
  ASSERT_TRUE(is(a.storage(), v.storage()));
  v /= 2.0;  // must not throw
  EXPECT_TRUE(is(a.storage(), v.storage()));
  Tensor a_cpu = a.to(Device.cpu);
  for (cytnx_uint64 i = 0; i < 6; i++)
    EXPECT_DOUBLE_EQ(a_cpu.storage().at<cytnx_double>(i), i / 2.0);
}

// Contiguous scalar ops on a GPU tensor still produce the right values. This
// also exercises the #988 efficiency path: the scalar wrapper stays on the
// host and is read by the GPU kernel with a host-side dereference.
TEST(Tensor, GpuScalarInplaceContiguousValues) {
  Tensor a = arange(6).to(Device.cuda);
  a *= 3.0;
  a += 1.0;
  EXPECT_EQ(a.device(), Device.cuda);
  Tensor a_cpu = a.to(Device.cpu);
  for (cytnx_uint64 i = 0; i < 6; i++)
    EXPECT_DOUBLE_EQ(a_cpu.storage().at<cytnx_double>(i), 3.0 * i + 1.0);
}

TEST(Tensor, GpuRankZeroTensorRhsInplacePreserveDtype) {
  Tensor rhs(std::vector<cytnx_uint64>{}, Type.Double, Device.cuda);
  rhs.set(std::vector<Accessor>{}, 2.0);

  Tensor a = ones({2}, Type.Float, Device.cuda);
  a += rhs;
  a -= rhs;
  a *= rhs;
  a /= rhs;

  EXPECT_EQ(a.dtype(), Type.Float);
  EXPECT_EQ(a.device(), Device.cuda);
  Tensor a_cpu = a.to(Device.cpu);
  EXPECT_FLOAT_EQ(a_cpu.storage().at<cytnx_float>(0), 1.0f);
  EXPECT_FLOAT_EQ(a_cpu.storage().at<cytnx_float>(1), 1.0f);
}

// Scalar in-place ops mutate the LHS storage in place (never detach), so a
// second handle onto the same GPU storage observes the change. Mirrors
// Tensor.ScalarInplaceSubMulDivKeepStorageSharing on CUDA.
TEST(Tensor, GpuScalarInplaceKeepsStorageSharing) {
  Tensor a = zeros({4}, Type.Double, Device.cuda);
  Tensor b = Tensor::from_storage(a.storage());  // distinct impl, shared storage
  ASSERT_TRUE(is(a.storage(), b.storage()));
  a += 1.0;
  a -= 0.5;
  a *= 4.0;
  a /= 2.0;  // ((0 + 1 - 0.5) * 4) / 2 == 1
  EXPECT_TRUE(is(a.storage(), b.storage()));
  Tensor b_cpu = b.to(Device.cpu);
  EXPECT_DOUBLE_EQ(b_cpu.storage().at<cytnx_double>(0), 1.0);
}

// A double scalar must not promote a Float GPU tensor to Double. Mirrors
// Tensor.ScalarInplaceOpsPreserveDtype on CUDA.
TEST(Tensor, GpuScalarInplacePreserveDtype) {
  Tensor a = ones({2}, Type.Float, Device.cuda);
  a += 1.0;  // double scalar
  a -= 0.5;
  a *= 2.0;
  a /= 3.0;  // ((1 + 1 - 0.5) * 2) / 3 == 1
  EXPECT_EQ(a.dtype(), Type.Float);
  EXPECT_EQ(a.device(), Device.cuda);
  Tensor a_cpu = a.to(Device.cpu);
  EXPECT_FLOAT_EQ(a_cpu.storage().at<cytnx_float>(0), 1.0f);
}

// A real GPU tensor op= a complex scalar cannot store a complex result, so it
// must throw (as on CPU) rather than silently reinterpreting the real buffer
// as complex. Mirrors Tensor.ScalarInplaceRealPlusComplexThrows on CUDA.
TEST(Tensor, GpuScalarInplaceRealOpComplexThrows) {
  Tensor a = zeros({2}, Type.Double, Device.cuda);
  EXPECT_THROW(a += cytnx_complex128(0, 1), std::logic_error);
  EXPECT_THROW(a -= cytnx_complex128(0, 1), std::logic_error);
  EXPECT_THROW(a *= cytnx_complex128(0, 1), std::logic_error);
  EXPECT_THROW(a /= cytnx_complex128(0, 1), std::logic_error);
}

// The cuMul/cuDiv GPU kernels (unlike cuAdd/cuSub) do not consume layout
// mappers, so a genuine non-contiguous tensor*=tensor must fail loudly instead
// of silently pairing mismatched elements. (The scalar broadcast case above is
// still supported because it ignores the mappers.)
TEST(Tensor, GpuNoncontigTensorTensorMulDivThrows) {
  Tensor a = arange(6).reshape({2, 3}).to(Device.cuda);
  Tensor b = arange(6).reshape({3, 2}).to(Device.cuda).permute({1, 0});  // {2,3}, non-contiguous
  ASSERT_EQ(a.shape(), b.shape());
  ASSERT_FALSE(b.is_contiguous());
  EXPECT_THROW(a *= b, std::logic_error);
  EXPECT_THROW(a /= b, std::logic_error);
}
