// Coverage + behavior tests for the #941 typed-storage CPU arithmetic
// conversion. The out-of-place families (Add/Sub/Mul/Div/Cpr/Mod) each expose a
// per-dtype scalar-left `Op<T>(scalar, Tensor)` and scalar-right
// `Op<T>(Tensor, scalar)` overload; exercising every one instantiates the typed
// std::visit dispatch for the full scalar-dtype range. We also drive the
// non-contiguous (permuted) tensor-tensor path and the in-place kernels, and
// pin a handful of independent expected values (dtype + value).

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "cytnx.hpp"

namespace cytnx {
  namespace {

    // Contiguous Double tensor with known values 1..6 shaped {2,3}.
    Tensor base_double() {
      Tensor t = zeros({2, 3}, Type.Double);
      double v = 1.0;
      for (cytnx_uint64 i = 0; i < 2; i++)
        for (cytnx_uint64 j = 0; j < 3; j++) t.at<cytnx_double>({i, j}) = v++;
      return t;
    }

    // Exercise scalar-left and scalar-right overloads of every family for one scalar
    // type T. `allow_mod` is false for complex scalars (Mod rejects a complex
    // result). Cpr is element-wise equality and is defined for every scalar type.
    template <class T>
    void sweep_scalar(T s, bool allow_mod) {
      const Tensor t = base_double();
      EXPECT_NO_THROW({ EXPECT_FALSE(linalg::Add(s, t).is_void()); });
      EXPECT_NO_THROW({ EXPECT_FALSE(linalg::Add(t, s).is_void()); });
      EXPECT_NO_THROW({ EXPECT_FALSE(linalg::Sub(s, t).is_void()); });
      EXPECT_NO_THROW({ EXPECT_FALSE(linalg::Sub(t, s).is_void()); });
      EXPECT_NO_THROW({ EXPECT_FALSE(linalg::Mul(s, t).is_void()); });
      EXPECT_NO_THROW({ EXPECT_FALSE(linalg::Mul(t, s).is_void()); });
      EXPECT_NO_THROW({ EXPECT_FALSE(linalg::Div(s, t).is_void()); });
      EXPECT_NO_THROW({ EXPECT_FALSE(linalg::Div(t, s).is_void()); });
      EXPECT_NO_THROW({ EXPECT_EQ(linalg::Cpr(s, t).dtype(), (unsigned int)Type.Bool); });
      EXPECT_NO_THROW({ EXPECT_EQ(linalg::Cpr(t, s).dtype(), (unsigned int)Type.Bool); });
      if (allow_mod) {
        EXPECT_NO_THROW({ EXPECT_FALSE(linalg::Mod(s, t).is_void()); });
        EXPECT_NO_THROW({ EXPECT_FALSE(linalg::Mod(t, s).is_void()); });
      }
    }

  }  // namespace

  // Every per-dtype scalar-left / scalar-right overload of all six families.
  TEST(TypedArithCoverage, ScalarOverloadsAllDtypes) {
    sweep_scalar<cytnx_complex128>(cytnx_complex128(2, 0), /*allow_mod=*/false);
    sweep_scalar<cytnx_complex64>(cytnx_complex64(2, 0), /*allow_mod=*/false);
    sweep_scalar<cytnx_double>(2.0, true);
    sweep_scalar<cytnx_float>(2.0f, true);
    sweep_scalar<cytnx_int64>(cytnx_int64(2), true);
    sweep_scalar<cytnx_uint64>(cytnx_uint64(2), true);
    sweep_scalar<cytnx_int32>(cytnx_int32(2), true);
    sweep_scalar<cytnx_uint32>(cytnx_uint32(2), true);
    sweep_scalar<cytnx_int16>(cytnx_int16(2), true);
    sweep_scalar<cytnx_uint16>(cytnx_uint16(2), true);
    sweep_scalar<cytnx_bool>(cytnx_bool(true), true);
    sweep_scalar<Scalar>(Scalar(2.0), true);
  }

  // Independent expected values for scalar-left / scalar-right arithmetic.
  TEST(TypedArithCoverage, ScalarValues) {
    const Tensor t = base_double();  // 1..6
    // scalar-left add / scalar-right sub
    EXPECT_DOUBLE_EQ(linalg::Add(10.0, t).at<cytnx_double>({0, 0}), 11.0);
    EXPECT_DOUBLE_EQ(linalg::Add(10.0, t).at<cytnx_double>({1, 2}), 16.0);
    EXPECT_DOUBLE_EQ(linalg::Sub(t, 1.0).at<cytnx_double>({1, 2}), 5.0);
    // scalar-left sub is not commutative
    EXPECT_DOUBLE_EQ(linalg::Sub(10.0, t).at<cytnx_double>({0, 0}), 9.0);
    EXPECT_DOUBLE_EQ(linalg::Mul(t, 3.0).at<cytnx_double>({0, 1}), 6.0);
    // true division of an integer tensor by an integer scalar -> Double
    Tensor a = arange(1, 4).astype(Type.Int64);  // [1,2,3]
    Tensor q = linalg::Div(a, cytnx_int64(2));
    EXPECT_EQ(q.dtype(), (unsigned int)Type.Double);
    EXPECT_DOUBLE_EQ(q.at<cytnx_double>({0}), 0.5);
    EXPECT_DOUBLE_EQ(q.at<cytnx_double>({2}), 1.5);
    // integer Mod keeps the (promoted integer) dtype
    Tensor m = linalg::Mod(arange(0, 5).astype(Type.Int64), cytnx_int64(3));
    EXPECT_EQ(m.at<cytnx_int64>({4}), 1);  // 4 % 3
  }

  // Non-contiguous (permuted) tensor-tensor path for every family, plus in-place.
  TEST(TypedArithCoverage, NonContiguousTensorTensor) {
    Tensor a = arange(0, 12).astype(Type.Double).reshape({3, 4});
    Tensor b = (arange(0, 12) + 1.0).astype(Type.Double).reshape({3, 4});
    Tensor ap = a.permute({1, 0});  // non-contiguous {4,3}
    Tensor bp = b.permute({1, 0});
    ASSERT_FALSE(ap.is_contiguous());
    ASSERT_FALSE(bp.is_contiguous());

    const std::vector<cytnx_uint64> want{4, 3};
    EXPECT_EQ(linalg::Add(ap, bp).shape(), want);
    EXPECT_EQ(linalg::Sub(ap, bp).shape(), want);
    EXPECT_EQ(linalg::Mul(ap, bp).shape(), want);
    EXPECT_EQ(linalg::Div(ap, bp).shape(), want);
    EXPECT_EQ(linalg::Cpr(ap, bp).dtype(), (unsigned int)Type.Bool);

    // a[i,j] = 4*i+j ; ap[j,i] = a[i,j]. ap[0,0]=0, bp[0,0]=1 -> Add=1, Sub=-1.
    EXPECT_DOUBLE_EQ(linalg::Add(ap, bp).at<cytnx_double>({0, 0}), 1.0);
    EXPECT_DOUBLE_EQ(linalg::Sub(ap, bp).at<cytnx_double>({0, 0}), -1.0);

    // integer Mod on permuted views (tensor-tensor, non-contiguous; RHS has no
    // zeros so the modulus is well-defined)
    Tensor ia = arange(0, 12).astype(Type.Int64).reshape({3, 4}).permute({1, 0});
    Tensor ib = arange(1, 13).astype(Type.Int64).reshape({3, 4}).permute({1, 0});
    EXPECT_EQ(linalg::Mod(ia, ib).shape(), want);

    // in-place families on permuted views (dtype-preserving contiguous-ize path)
    Tensor x = ap.clone();
    EXPECT_NO_THROW(linalg::iAdd(x, bp));
    Tensor y = ap.clone();
    EXPECT_NO_THROW(linalg::iSub(y, bp));
    Tensor z = ap.clone();
    EXPECT_NO_THROW(linalg::iMul(z, bp));
    Tensor w = ap.clone();
    EXPECT_NO_THROW(linalg::iDiv(w, bp));
  }

}  // namespace cytnx
