#include <vector>

#include "gtest/gtest.h"

#include "Device.hpp"
#include "Generator.hpp"
#include "Type.hpp"
#include "linalg.hpp"
#include "algo.hpp"
#include "UniTensor.hpp"

// Mixed-dtype promotion coverage for the linalg entry points that used to
// hand-roll "pick the lower enum index" dtype selection instead of
// Type.type_promote. The discriminating pairs:
//   * ComplexFloat x Double -> ComplexDouble (old rule kept ComplexFloat,
//     discarding double precision),
//   * Uint64 x Int32 -> Int64 (old rule kept Uint64, disagreeing with the
//     kernels' Type_class::type_promote_t),
//   * integer inputs to the BLAS-only ops (Ger/Gemm) must floor to
//     Double, since their kernel tables only cover the four float types.

namespace cytnx {
  namespace {

    // A: 2x2 ComplexFloat [[1+2i, 3], [-1i, 2+0.5i]] (exactly float-representable)
    Tensor MakeComplexFloatA() {
      Tensor A = zeros({2, 2}, Type.ComplexFloat, Device.cpu);
      A.at<cytnx_complex64>({0, 0}) = cytnx_complex64(1, 2);
      A.at<cytnx_complex64>({0, 1}) = cytnx_complex64(3, 0);
      A.at<cytnx_complex64>({1, 0}) = cytnx_complex64(0, -1);
      A.at<cytnx_complex64>({1, 1}) = cytnx_complex64(2, 0.5);
      return A;
    }

    // B: 2x2 Double [[0.5, 1], [2, -1.5]]
    Tensor MakeDoubleB() {
      Tensor B = zeros({2, 2}, Type.Double, Device.cpu);
      B.at<cytnx_double>({0, 0}) = 0.5;
      B.at<cytnx_double>({0, 1}) = 1;
      B.at<cytnx_double>({1, 0}) = 2;
      B.at<cytnx_double>({1, 1}) = -1.5;
      return B;
    }

    Tensor MakeInt64(const std::vector<cytnx_uint64>& shape, const std::vector<cytnx_int64>& data) {
      Tensor t = zeros(shape, Type.Int64, Device.cpu);
      for (cytnx_uint64 i = 0; i < data.size(); i++) t.storage().at<cytnx_int64>(i) = data[i];
      return t;
    }

    void ExpectComplexNear(const Tensor& t, const std::vector<cytnx_uint64>& idx, double re,
                           double im, double tol = 1e-6) {
      auto v = const_cast<Tensor&>(t).at<cytnx_complex128>(idx);
      EXPECT_NEAR(v.real(), re, tol);
      EXPECT_NEAR(v.imag(), im, tol);
    }

    // A*B for the fixtures above:
    // [[6.5+1i, -3.5+2i], [4+0.5i, -3-1.75i]]
    void ExpectAB(const Tensor& out, double scale = 1.0) {
      ExpectComplexNear(out, {0, 0}, 6.5 * scale, 1 * scale);
      ExpectComplexNear(out, {0, 1}, -3.5 * scale, 2 * scale);
      ExpectComplexNear(out, {1, 0}, 4 * scale, 0.5 * scale);
      ExpectComplexNear(out, {1, 1}, -3 * scale, -1.75 * scale);
    }

  }  // namespace

  TEST(DtypePromotion, Matmul_complexfloat_double) {
    Tensor out = linalg::Matmul(MakeComplexFloatA(), MakeDoubleB());
    ASSERT_EQ(out.dtype(), Type.ComplexDouble);
    ExpectAB(out);
  }

  TEST(DtypePromotion, Matmul_double_complexfloat) {
    // Same pair, operands swapped: B*A = [[0.5, 3.5+0.5i], [2+5.5i, 3-0.75i]]
    Tensor out = linalg::Matmul(MakeDoubleB(), MakeComplexFloatA());
    ASSERT_EQ(out.dtype(), Type.ComplexDouble);
    ExpectComplexNear(out, {0, 0}, 0.5, 0);
    ExpectComplexNear(out, {0, 1}, 3.5, 0.5);
    ExpectComplexNear(out, {1, 0}, 2, 5.5);
    ExpectComplexNear(out, {1, 1}, 3, -0.75);
  }

  TEST(DtypePromotion, Dot_matvec_complexfloat_double) {
    // Rank-2 x rank-1 goes through Dot's own Matvec path (not Matmul/Vectordot).
    Tensor v = zeros({2}, Type.Double, Device.cpu);
    v.at<cytnx_double>({0}) = 0.5;
    v.at<cytnx_double>({1}) = 2;
    Tensor out = linalg::Dot(MakeComplexFloatA(), v);
    ASSERT_EQ(out.dtype(), Type.ComplexDouble);
    // A.v = {(1+2i)*0.5 + 3*2, -1i*0.5 + (2+0.5i)*2} = {6.5+1i, 4+0.5i}
    ExpectComplexNear(out, {0}, 6.5, 1);
    ExpectComplexNear(out, {1}, 4, 0.5);
  }

  TEST(DtypePromotion, Matmul_dg_complexfloat_double) {
    // diag(d) * B with d = {1+2i, -1i}: rows of B scaled by d.
    Tensor d = zeros({2}, Type.ComplexFloat, Device.cpu);
    d.at<cytnx_complex64>({0}) = cytnx_complex64(1, 2);
    d.at<cytnx_complex64>({1}) = cytnx_complex64(0, -1);
    Tensor out = linalg::Matmul_dg(d, MakeDoubleB());
    ASSERT_EQ(out.dtype(), Type.ComplexDouble);
    ExpectComplexNear(out, {0, 0}, 0.5, 1);
    ExpectComplexNear(out, {0, 1}, 1, 2);
    ExpectComplexNear(out, {1, 0}, 0, -2);
    ExpectComplexNear(out, {1, 1}, 0, 1.5);
  }

  TEST(DtypePromotion, Tensordot_complexfloat_double) {
    Tensor out = linalg::Tensordot(MakeComplexFloatA(), MakeDoubleB(), {1}, {0});
    ASSERT_EQ(out.dtype(), Type.ComplexDouble);
    ExpectAB(out);
  }

  TEST(DtypePromotion, Gemm_complexfloat_double) {
    Tensor out = linalg::Gemm(Scalar(2.0), MakeComplexFloatA(), MakeDoubleB());
    ASSERT_EQ(out.dtype(), Type.ComplexDouble);
    ExpectAB(out, 2.0);
  }

  TEST(DtypePromotion, Gemm_integer_floors_to_double) {
    Tensor x = MakeInt64({2, 2}, {1, 2, 3, 4});
    Tensor y = MakeInt64({2, 2}, {5, 6, 7, 8});
    Tensor out = linalg::Gemm(Scalar((cytnx_int64)1), x, y);
    ASSERT_EQ(out.dtype(), Type.Double);
    EXPECT_DOUBLE_EQ(out.at<cytnx_double>({0, 0}), 19);
    EXPECT_DOUBLE_EQ(out.at<cytnx_double>({0, 1}), 22);
    EXPECT_DOUBLE_EQ(out.at<cytnx_double>({1, 0}), 43);
    EXPECT_DOUBLE_EQ(out.at<cytnx_double>({1, 1}), 50);
  }

  TEST(DtypePromotion, GemmInplace_promotes_c) {
    // c = 1*A*B + 2*c with c starting as the Double identity -> ComplexDouble
    Tensor c = zeros({2, 2}, Type.Double, Device.cpu);
    c.at<cytnx_double>({0, 0}) = 1;
    c.at<cytnx_double>({1, 1}) = 1;
    linalg::Gemm_(Scalar(1.0), MakeComplexFloatA(), MakeDoubleB(), Scalar(2.0), c);
    ASSERT_EQ(c.dtype(), Type.ComplexDouble);
    ExpectComplexNear(c, {0, 0}, 8.5, 1);
    ExpectComplexNear(c, {0, 1}, -3.5, 2);
    ExpectComplexNear(c, {1, 0}, 4, 0.5);
    ExpectComplexNear(c, {1, 1}, -1, -1.75);
  }

  TEST(DtypePromotion, GemmBatch_complexfloat_double) {
    std::vector<Tensor> as = {MakeComplexFloatA()};
    std::vector<Tensor> bs = {MakeDoubleB()};
    std::vector<Tensor> cs = {zeros({2, 2}, Type.Double, Device.cpu)};
    std::vector<Scalar> alphas = {Scalar(1.0)}, betas = {Scalar(0.0)};
#ifdef UNI_MKL
    linalg::Gemm_Batch(alphas, as, bs, betas, cs, {1});
    ASSERT_EQ(cs[0].dtype(), Type.ComplexDouble);
    ExpectAB(cs[0]);
#else
    // Without MKL the CPU kernel throws, but only after the inputs (including
    // the in/out c tensors) have been cast to the promoted dtype - which is
    // exactly the decision under test.
    EXPECT_THROW(linalg::Gemm_Batch(alphas, as, bs, betas, cs, {1}), std::logic_error);
    ASSERT_EQ(cs[0].dtype(), Type.ComplexDouble);
#endif
  }

  TEST(DtypePromotion, Ger_complexfloat_double) {
    // outer product x ⊗ y (geru: no conjugation) with x = {1+2i, -1i}, y = {0.5, 2}
    Tensor x = zeros({2}, Type.ComplexFloat, Device.cpu);
    x.at<cytnx_complex64>({0}) = cytnx_complex64(1, 2);
    x.at<cytnx_complex64>({1}) = cytnx_complex64(0, -1);
    Tensor y = zeros({2}, Type.Double, Device.cpu);
    y.at<cytnx_double>({0}) = 0.5;
    y.at<cytnx_double>({1}) = 2;
    Tensor out = linalg::Ger(x, y);
    ASSERT_EQ(out.dtype(), Type.ComplexDouble);
    ExpectComplexNear(out, {0, 0}, 0.5, 1);
    ExpectComplexNear(out, {0, 1}, 2, 4);
    ExpectComplexNear(out, {1, 0}, 0, -0.5);
    ExpectComplexNear(out, {1, 1}, 0, -2);
  }

  TEST(DtypePromotion, Ger_integer_floors_to_double) {
    Tensor x = MakeInt64({2}, {1, 2});
    Tensor y = MakeInt64({2}, {3, 4});
    Tensor out = linalg::Ger(x, y);
    ASSERT_EQ(out.dtype(), Type.Double);
    EXPECT_DOUBLE_EQ(out.at<cytnx_double>({0, 0}), 3);
    EXPECT_DOUBLE_EQ(out.at<cytnx_double>({0, 1}), 4);
    EXPECT_DOUBLE_EQ(out.at<cytnx_double>({1, 0}), 6);
    EXPECT_DOUBLE_EQ(out.at<cytnx_double>({1, 1}), 8);
  }

  TEST(DtypePromotion, Mod_uint64_int32_promotes_to_int64) {
    // The Mod kernel computes in Type_class::type_promote_t<TL,TR> (= Int64
    // for Uint64 x Int32); the output buffer dtype must agree with it.
    Tensor l = zeros({2}, Type.Uint64, Device.cpu);
    l.at<cytnx_uint64>({0}) = 5;
    l.at<cytnx_uint64>({1}) = 7;
    Tensor r = zeros({2}, Type.Int32, Device.cpu);
    r.at<cytnx_int32>({0}) = 3;
    r.at<cytnx_int32>({1}) = 4;
    Tensor out = linalg::Mod(l, r);
    ASSERT_EQ(out.dtype(), Type.Int64);
    EXPECT_EQ(out.at<cytnx_int64>({0}), 2);
    EXPECT_EQ(out.at<cytnx_int64>({1}), 3);
  }

  TEST(DtypePromotion, Mod_double_int64) {
    Tensor l = zeros({2}, Type.Double, Device.cpu);
    l.at<cytnx_double>({0}) = 5.5;
    l.at<cytnx_double>({1}) = 7.25;
    Tensor r = MakeInt64({2}, {3, 4});
    Tensor out = linalg::Mod(l, r);
    ASSERT_EQ(out.dtype(), Type.Double);
    EXPECT_DOUBLE_EQ(out.at<cytnx_double>({0}), 2.5);
    EXPECT_DOUBLE_EQ(out.at<cytnx_double>({1}), 3.25);
  }

  TEST(DtypePromotion, Mod_tensor_scalar_uint64_int32) {
    Tensor l = zeros({2}, Type.Uint64, Device.cpu);
    l.at<cytnx_uint64>({0}) = 5;
    l.at<cytnx_uint64>({1}) = 7;
    Tensor out = linalg::Mod(l, Scalar(3, Type.Int32));
    ASSERT_EQ(out.dtype(), Type.Int64);
    EXPECT_EQ(out.at<cytnx_int64>({0}), 2);
    EXPECT_EQ(out.at<cytnx_int64>({1}), 1);
  }

  TEST(DtypePromotion, Mod_scalar_tensor_uint64_int32) {
    Tensor r = zeros({2}, Type.Uint64, Device.cpu);
    r.at<cytnx_uint64>({0}) = 3;
    r.at<cytnx_uint64>({1}) = 4;
    Tensor out = linalg::Mod(Scalar(7, Type.Int32), r);
    ASSERT_EQ(out.dtype(), Type.Int64);
    EXPECT_EQ(out.at<cytnx_int64>({0}), 1);
    EXPECT_EQ(out.at<cytnx_int64>({1}), 3);
  }

  TEST(DtypePromotion, Mod_tensor_typed_scalar_uint64_int32) {
    // The typed-scalar specializations hard-code the same lower-enum-index
    // rule; Uint64 % cytnx_int32 must also come out Int64.
    Tensor l = zeros({2}, Type.Uint64, Device.cpu);
    l.at<cytnx_uint64>({0}) = 5;
    l.at<cytnx_uint64>({1}) = 7;
    Tensor out = linalg::Mod(l, (cytnx_int32)3);
    ASSERT_EQ(out.dtype(), Type.Int64);
    EXPECT_EQ(out.at<cytnx_int64>({0}), 2);
    EXPECT_EQ(out.at<cytnx_int64>({1}), 1);
  }

  // ---------------------------------------------------------------------------
  // Follow-up to the review of #984: entry points that still selected the
  // lower-enum-index dtype instead of Type.type_promote. The discriminating
  // pair ComplexFloat x Double must promote to ComplexDouble (the old rule kept
  // ComplexFloat, dropping double precision and, for Outer/Exp, writing the
  // narrower complex into the wider output buffer).
  // ---------------------------------------------------------------------------

  TEST(DtypePromotion, Outer_complexfloat_double) {
    Tensor a = zeros({2}, Type.ComplexFloat, Device.cpu);
    a.at<cytnx_complex64>({0}) = cytnx_complex64(1, 1);
    a.at<cytnx_complex64>({1}) = cytnx_complex64(2, 0);
    Tensor b = zeros({2}, Type.Double, Device.cpu);
    b.at<cytnx_double>({0}) = 3;
    b.at<cytnx_double>({1}) = 0.5;
    Tensor out = linalg::Outer(a, b);
    ASSERT_EQ(out.dtype(), Type.ComplexDouble);
    ExpectComplexNear(out, {0, 0}, 3, 3);  // (1+1i) * 3
    ExpectComplexNear(out, {1, 1}, 1, 0);  // 2 * 0.5
  }

  TEST(DtypePromotion, Exp_complexfloat_promotes_to_complexdouble) {
    Tensor a = zeros({1}, Type.ComplexFloat, Device.cpu);  // exp(0) = 1
    Tensor out = linalg::Exp(a);
    ASSERT_EQ(out.dtype(), Type.ComplexDouble);
    ExpectComplexNear(out, {0}, 1, 0);
  }

  TEST(DtypePromotion, Exp_float_reads_correct_buffer_width) {
    // Old dispatch fed the float storage to the Double kernel, reinterpreting
    // float bits as double. Nonzero input makes the corruption visible.
    Tensor a = zeros({1}, Type.Float, Device.cpu);
    a.storage().at<cytnx_float>(0) = 1.0f;
    Tensor out = linalg::Exp(a);
    ASSERT_EQ(out.dtype(), Type.Double);
    EXPECT_NEAR(out.at<cytnx_double>({0}), 2.718281828459045, 1e-6);
  }

  TEST(DtypePromotion, Concatenate_complexfloat_double) {
    Tensor a = zeros({2}, Type.ComplexFloat, Device.cpu);
    a.at<cytnx_complex64>({0}) = cytnx_complex64(1, 2);
    Tensor b = zeros({2}, Type.Double, Device.cpu);
    b.at<cytnx_double>({0}) = 3;
    Tensor out = algo::Concatenate(a, b);
    ASSERT_EQ(out.dtype(), Type.ComplexDouble);
    ExpectComplexNear(out, {0}, 1, 2);
    ExpectComplexNear(out, {2}, 3, 0);
  }

  TEST(DtypePromotion, Hstack_complexfloat_double) {
    Tensor a = zeros({1, 1}, Type.ComplexFloat, Device.cpu);
    a.at<cytnx_complex64>({0, 0}) = cytnx_complex64(1, 2);
    Tensor b = zeros({1, 1}, Type.Double, Device.cpu);
    b.at<cytnx_double>({0, 0}) = 3;
    Tensor out = algo::Hstack({a, b});
    ASSERT_EQ(out.dtype(), Type.ComplexDouble);
    ExpectComplexNear(out, {0, 0}, 1, 2);
    ExpectComplexNear(out, {0, 1}, 3, 0);
  }

  TEST(DtypePromotion, Vstack_complexfloat_double) {
    Tensor a = zeros({1, 1}, Type.ComplexFloat, Device.cpu);
    a.at<cytnx_complex64>({0, 0}) = cytnx_complex64(1, 2);
    Tensor b = zeros({1, 1}, Type.Double, Device.cpu);
    b.at<cytnx_double>({0, 0}) = 3;
    Tensor out = algo::Vstack({a, b});
    ASSERT_EQ(out.dtype(), Type.ComplexDouble);
    ExpectComplexNear(out, {0, 0}, 1, 2);
    ExpectComplexNear(out, {1, 0}, 3, 0);
  }

  TEST(DtypePromotion, Directsum_complexfloat_double) {
    Tensor a = zeros({1, 1}, Type.ComplexFloat, Device.cpu);
    a.at<cytnx_complex64>({0, 0}) = cytnx_complex64(1, 2);
    Tensor b = zeros({1, 1}, Type.Double, Device.cpu);
    b.at<cytnx_double>({0, 0}) = 3;
    Tensor out = linalg::Directsum(a, b, {});
    ASSERT_EQ(out.dtype(), Type.ComplexDouble);
    ExpectComplexNear(out, {0, 0}, 1, 2);
    ExpectComplexNear(out, {1, 1}, 3, 0);
  }

  TEST(DtypePromotion, Lstsq_complexfloat_double) {
    Tensor A = MakeComplexFloatA();
    Tensor b = zeros({2, 1}, Type.Double, Device.cpu);
    b.at<cytnx_double>({0, 0}) = 1;
    b.at<cytnx_double>({1, 0}) = 2;
    std::vector<Tensor> res = linalg::Lstsq(A, b);
    ASSERT_EQ(res[0].dtype(), Type.ComplexDouble);
  }

  TEST(DtypePromotion, Scalar_complexfloat_double_promotes) {
    Scalar a = Scalar(cytnx_complex64(1, 2));
    Scalar b = Scalar(cytnx_double(3));
    EXPECT_EQ((a + b).dtype(), Type.ComplexDouble);
    EXPECT_EQ((a - b).dtype(), Type.ComplexDouble);
    EXPECT_EQ((a * b).dtype(), Type.ComplexDouble);
    EXPECT_EQ((a / b).dtype(), Type.ComplexDouble);
  }

  TEST(DtypePromotion, UniTensorArithmetic_complexfloat_double_promotes) {
    UniTensor A(ones({2, 2}, Type.ComplexFloat, Device.cpu));
    UniTensor B(ones({2, 2}, Type.Double, Device.cpu));
    EXPECT_EQ(linalg::Add(A, B).dtype(), Type.ComplexDouble);
    EXPECT_EQ(linalg::Sub(A, B).dtype(), Type.ComplexDouble);
    EXPECT_EQ(linalg::Mul(A, B).dtype(), Type.ComplexDouble);
    EXPECT_EQ(linalg::Div(A, B).dtype(), Type.ComplexDouble);
  }

  namespace {
    // Every typed-scalar Mod specialization allocates its own output buffer;
    // each must agree with the kernel's Type_class::type_promote_t. Exercises
    // both the scalar-left and scalar-right specializations for one scalar
    // type against a different-kind tensor dtype.
    template <typename S>
    void ExpectModTypedScalarPromotes(unsigned int tensor_dtype, S sc) {
      const unsigned int sc_dtype = Type.cy_typeid(sc);
      Tensor t = zeros({2}, Type.Double, Device.cpu);
      t.at<cytnx_double>({0}) = 5;
      t.at<cytnx_double>({1}) = 7;
      t = t.astype(tensor_dtype);

      Tensor r = linalg::Mod(t, sc);
      EXPECT_EQ(r.dtype(), Type.type_promote(tensor_dtype, sc_dtype))
        << "tensor % scalar, tensor dtype " << tensor_dtype << ", scalar dtype " << sc_dtype;

      Tensor l = linalg::Mod(sc, t);
      EXPECT_EQ(l.dtype(), Type.type_promote(sc_dtype, tensor_dtype))
        << "scalar % tensor, tensor dtype " << tensor_dtype << ", scalar dtype " << sc_dtype;
    }
  }  // namespace

  TEST(DtypePromotion, Mod_typed_scalar_specializations_promote) {
    ExpectModTypedScalarPromotes(Type.Float, cytnx_double(3));
    ExpectModTypedScalarPromotes(Type.Int64, cytnx_float(3));
    ExpectModTypedScalarPromotes(Type.Uint64, cytnx_int64(3));
    ExpectModTypedScalarPromotes(Type.Int32, cytnx_uint64(3));
    ExpectModTypedScalarPromotes(Type.Uint32, cytnx_int32(3));
    ExpectModTypedScalarPromotes(Type.Int16, cytnx_uint32(3));
    ExpectModTypedScalarPromotes(Type.Uint16, cytnx_int16(3));
    ExpectModTypedScalarPromotes(Type.Int64, cytnx_uint16(3));
    ExpectModTypedScalarPromotes(Type.Uint16, cytnx_bool(true));
  }

  TEST(DtypePromotion, Mod_complexfloat_scalar_throws_after_promotion) {
    // Mod on complex is rejected by the kernel, but the ComplexFloat-scalar
    // specializations still allocate the promoted output buffer first.
    Tensor t = zeros({2}, Type.Double, Device.cpu);
    t.at<cytnx_double>({0}) = 5;
    t.at<cytnx_double>({1}) = 7;
    EXPECT_THROW(linalg::Mod(t, cytnx_complex64(1, 0)), std::logic_error);
    EXPECT_THROW(linalg::Mod(cytnx_complex64(1, 0), t), std::logic_error);
  }

}  // namespace cytnx
