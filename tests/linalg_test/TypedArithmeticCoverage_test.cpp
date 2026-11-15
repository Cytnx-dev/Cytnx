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
  namespace test {
    namespace {

      // Contiguous Double tensor with known values 1..6 shaped {2,3}.
      static Tensor base_double() {
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
      static void sweep_scalar(T s, bool allow_mod) {
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

      // Real Double [1,2,3].
      Tensor real_lhs_123() {
        Tensor t = zeros({3}, Type.Double);
        t.at<cytnx_double>({0}) = 1.0;
        t.at<cytnx_double>({1}) = 2.0;
        t.at<cytnx_double>({2}) = 3.0;
        return t;
      }

      // ComplexDouble [1+2i, 2+1i, 3+3i].
      Tensor complex_rhs_c() {
        Tensor r = zeros({3}, Type.ComplexDouble);
        r.at<cytnx_complex128>({0}) = cytnx_complex128(1, 2);
        r.at<cytnx_complex128>({1}) = cytnx_complex128(2, 1);
        r.at<cytnx_complex128>({2}) = cytnx_complex128(3, 3);
        return r;
      }

      // A genuine complex *tensor* RHS promotes a real LHS to complex in place, exactly like
      // the out-of-place op (#941/#1013). Ian's #1067 review: the pre-fix in-place guard
      // rejected EVERY real-LHS/complex-RHS combination, which contradicts this promotion
      // rule -- only a complex python *weak scalar* must be rejected (see the sibling test).
      // Independent hand-computed expected values (dtype + element values), NOT a CPU-vs-CPU
      // oracle, per CLAUDE.md.
      TEST(TypedArithCoverage, InplaceRealOpComplexTensorPromotes) {
        // iAdd: [1,2,3] += [1+2i,2+1i,3+3i] = [2+2i, 4+1i, 6+3i]
        {
          Tensor l = real_lhs_123();
          linalg::iAdd(l, complex_rhs_c());
          EXPECT_EQ(l.dtype(), (unsigned int)Type.ComplexDouble);
          EXPECT_EQ(l.at<cytnx_complex128>({0}), cytnx_complex128(2, 2));
          EXPECT_EQ(l.at<cytnx_complex128>({1}), cytnx_complex128(4, 1));
          EXPECT_EQ(l.at<cytnx_complex128>({2}), cytnx_complex128(6, 3));
        }
        // iSub: [1,2,3] -= [1+2i,2+1i,3+3i] = [0-2i, 0-1i, 0-3i]
        {
          Tensor l = real_lhs_123();
          linalg::iSub(l, complex_rhs_c());
          EXPECT_EQ(l.dtype(), (unsigned int)Type.ComplexDouble);
          EXPECT_EQ(l.at<cytnx_complex128>({0}), cytnx_complex128(0, -2));
          EXPECT_EQ(l.at<cytnx_complex128>({1}), cytnx_complex128(0, -1));
          EXPECT_EQ(l.at<cytnx_complex128>({2}), cytnx_complex128(0, -3));
        }
        // iMul: [1,2,3] *= [1+2i,2+1i,3+3i] = [1+2i, 4+2i, 9+9i]
        {
          Tensor l = real_lhs_123();
          linalg::iMul(l, complex_rhs_c());
          EXPECT_EQ(l.dtype(), (unsigned int)Type.ComplexDouble);
          EXPECT_EQ(l.at<cytnx_complex128>({0}), cytnx_complex128(1, 2));
          EXPECT_EQ(l.at<cytnx_complex128>({1}), cytnx_complex128(4, 2));
          EXPECT_EQ(l.at<cytnx_complex128>({2}), cytnx_complex128(9, 9));
        }
        // iDiv (true division in the complex field):
        // 1/(1+2i)=0.2-0.4i ; 2/(2+1i)=0.8-0.4i ; 3/(3+3i)=0.5-0.5i
        {
          Tensor l = real_lhs_123();
          linalg::iDiv(l, complex_rhs_c());
          EXPECT_EQ(l.dtype(), (unsigned int)Type.ComplexDouble);
          const cytnx_complex128 e0 = l.at<cytnx_complex128>({0});
          const cytnx_complex128 e1 = l.at<cytnx_complex128>({1});
          const cytnx_complex128 e2 = l.at<cytnx_complex128>({2});
          EXPECT_NEAR(e0.real(), 0.2, 1e-12);
          EXPECT_NEAR(e0.imag(), -0.4, 1e-12);
          EXPECT_NEAR(e1.real(), 0.8, 1e-12);
          EXPECT_NEAR(e1.imag(), -0.4, 1e-12);
          EXPECT_NEAR(e2.real(), 0.5, 1e-12);
          EXPECT_NEAR(e2.imag(), -0.5, 1e-12);
        }
        // A genuine rank-0 complex tensor RHS (Ian's #1067 example) is still a genuine
        // tensor, not a weak scalar: it broadcasts and promotes.
        // [1,2,3] += (2+1i) -> [3+1i, 4+1i, 5+1i].
        {
          Tensor l = real_lhs_123();
          Tensor r0 = zeros({1}, Type.ComplexDouble);
          r0.at<cytnx_complex128>({0}) = cytnx_complex128(2, 1);
          r0 = r0.reshape({});  // genuine rank-0 tensor (shape {}, one element)
          ASSERT_EQ(r0.shape().size(), 0u);
          linalg::iAdd(l, r0);
          EXPECT_EQ(l.dtype(), (unsigned int)Type.ComplexDouble);
          EXPECT_EQ(l.at<cytnx_complex128>({0}), cytnx_complex128(3, 1));
          EXPECT_EQ(l.at<cytnx_complex128>({1}), cytnx_complex128(4, 1));
          EXPECT_EQ(l.at<cytnx_complex128>({2}), cytnx_complex128(5, 1));
        }
      }

      // The complementary rejection: a complex python *weak scalar* into a real LHS is still
      // rejected, because numpy weak-scalar semantics (#980/#1015) preserve the LHS dtype and
      // a complex value cannot be stored in a real tensor. Exercised both via the user-facing
      // scalar operators and via the explicit rhs_is_weak_scalar flag.
      TEST(TypedArithCoverage, InplaceRealOpComplexWeakScalarRejected) {
        // user-facing: real_tensor (op)= complex scalar routes to the weak-scalar path.
        {
          Tensor t = real_lhs_123();
          EXPECT_THROW(t += cytnx_complex128(1, 1), std::logic_error);
        }
        {
          Tensor t = real_lhs_123();
          EXPECT_THROW(t -= cytnx_complex128(1, 1), std::logic_error);
        }
        {
          Tensor t = real_lhs_123();
          EXPECT_THROW(t *= cytnx_complex128(1, 1), std::logic_error);
        }
        {
          Tensor t = real_lhs_123();
          EXPECT_THROW(t /= cytnx_complex128(1, 1), std::logic_error);
        }
        // explicit weak-scalar flag with a complex tensor RHS also throws (guards the branch
        // directly); the same RHS with the default genuine-tensor flag promotes instead.
        {
          Tensor t = real_lhs_123();
          EXPECT_THROW(linalg::iAdd(t, complex_rhs_c(), /*rhs_is_weak_scalar=*/true),
                       std::logic_error);
          Tensor t2 = real_lhs_123();
          EXPECT_NO_THROW(linalg::iAdd(t2, complex_rhs_c()));
          EXPECT_EQ(t2.dtype(), (unsigned int)Type.ComplexDouble);
        }
      }

      template <class T>
      T make_value(double re, double im) {
        if constexpr (is_complex_v<T>) {
          using Real = typename T::value_type;
          return T(static_cast<Real>(re), static_cast<Real>(im));
        } else {
          return static_cast<T>(re);
        }
      }

      // A contiguous rank-1 tensor of dtype T holding values[k].
      template <class T>
      Tensor make_vector(const std::vector<T>& values) {
        Tensor t = zeros({values.size()}, Type.cy_typeid(T()));
        for (std::size_t k = 0; k < values.size(); ++k) t.at<T>({k}) = values[k];
        return t;
      }

      // Checks `lhs op= rhs` (tensor RHS) and `lhs op= rhs_scalar` (weak scalar RHS) for one
      // op, where lhs and rhs share dtype T so the kernel writes into lhs's own buffer. The
      // expected values come from T's own arithmetic, one element at a time. 37 elements cover
      // a vectorized loop body plus a scalar tail.
      template <class T, class Op>
      void check_same_dtype_inplace(Op op) {
        const std::size_t n = 37;
        std::vector<T> lhs_values(n), rhs_values(n);
        for (std::size_t k = 0; k < n; ++k) {
          lhs_values[k] = make_value<T>(0.25 * k - 4.0, 1.5 - 0.125 * k);
          rhs_values[k] = make_value<T>(0.5 + 0.125 * k, -0.75 + 0.0625 * k);
        }
        const T rhs_scalar = make_value<T>(-1.25, 0.5);

        Tensor with_tensor = make_vector(lhs_values);
        const T* tensor_buffer = with_tensor.ptr_as<T>();
        op(with_tensor, make_vector(rhs_values));
        Tensor with_scalar = make_vector(lhs_values);
        const T* scalar_buffer = with_scalar.ptr_as<T>();
        op(with_scalar, rhs_scalar);

        // Same dtype, same buffer: the kernel wrote in place rather than promoting.
        EXPECT_EQ(with_tensor.dtype(), Type.cy_typeid(T()));
        EXPECT_EQ(with_tensor.ptr_as<T>(), tensor_buffer);
        EXPECT_EQ(with_scalar.dtype(), Type.cy_typeid(T()));
        EXPECT_EQ(with_scalar.ptr_as<T>(), scalar_buffer);
        for (std::size_t k = 0; k < n; ++k) {
          T expected_tensor = lhs_values[k];
          op(expected_tensor, rhs_values[k]);
          T expected_scalar = lhs_values[k];
          op(expected_scalar, rhs_scalar);
          EXPECT_EQ(with_tensor.at<T>({k}), expected_tensor) << "element " << k;
          EXPECT_EQ(with_scalar.at<T>({k}), expected_scalar) << "element " << k;
        }
      }

      template <class T>
      void check_same_dtype_inplace_all_ops() {
        check_same_dtype_inplace<T>([](auto& a, const auto& b) { a += b; });
        check_same_dtype_inplace<T>([](auto& a, const auto& b) { a -= b; });
        check_same_dtype_inplace<T>([](auto& a, const auto& b) { a *= b; });
        check_same_dtype_inplace<T>([](auto& a, const auto& b) { a /= b; });
      }

      // In-place arithmetic whose output is the LHS buffer itself, for every op and every
      // floating-point dtype.
      TEST(TypedArithCoverage, InplaceSameDtypeWritesIntoLhs) {
        check_same_dtype_inplace_all_ops<cytnx_float>();
        check_same_dtype_inplace_all_ops<cytnx_double>();
        check_same_dtype_inplace_all_ops<cytnx_complex64>();
        check_same_dtype_inplace_all_ops<cytnx_complex128>();
      }

    }  // namespace
  }  // namespace test
}  // namespace cytnx
