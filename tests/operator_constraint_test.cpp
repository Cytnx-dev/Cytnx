#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "cytnx.hpp"

// Guard for #1003 (Ian's item 3): the free Tensor/UniTensor arithmetic and comparison operators
// (`+ - * / % ==`) are constrained to scalar-like operands via the `cytnx_scalar_like` concept, so
// arbitrary std / user-defined types are no longer viable Cytnx arithmetic operands through
// implicit conversion. These are primarily COMPILE-TIME guards; if the constraint is dropped, the
// `static_assert`s (and the whole TU) fail.
using namespace cytnx;

// The concept admits exactly the operator support surface: cytnx dtype scalars + Scalar + proxies.
static_assert(cytnx_scalar_like<cytnx_double>);
static_assert(cytnx_scalar_like<cytnx_complex128>);
static_assert(cytnx_scalar_like<cytnx_float>);
static_assert(cytnx_scalar_like<cytnx_int64>);
static_assert(cytnx_scalar_like<cytnx_bool>);
static_assert(cytnx_scalar_like<int>);  // == cytnx_int32
static_assert(cytnx_scalar_like<Scalar>);

// ...and rejects everything else.
static_assert(!cytnx_scalar_like<std::string>);
static_assert(!cytnx_scalar_like<Tensor>);
static_assert(!cytnx_scalar_like<UniTensor>);
static_assert(!cytnx_scalar_like<std::vector<double>>);
static_assert(!cytnx_scalar_like<const char*>);

// Each concept must admit exactly the operand set its operators are instantiated for, otherwise an
// accepted operand still dies at link time -- the failure mode this whole change exists to remove.
// Widening any of these without adding the matching explicit instantiation reintroduces it.

// `+ - * /` on Tensor are instantiated for both element proxies (Add.cpp instantiates
// operator+<Tensor::Tproxy> and operator+<Scalar::Sproxy>).
static_assert(cytnx_scalar_like<Tensor::Tproxy>);
static_assert(cytnx_scalar_like<Scalar::Sproxy>);

// `+ - * /` on UniTensor are instantiated for Scalar::Sproxy but never for Tensor::Tproxy.
static_assert(cytnx_unitensor_scalar_like<Scalar::Sproxy>);
static_assert(!cytnx_unitensor_scalar_like<Tensor::Tproxy>);
static_assert(cytnx_unitensor_scalar_like<cytnx_double>);
static_assert(cytnx_unitensor_scalar_like<Scalar>);
static_assert(!cytnx_unitensor_scalar_like<std::string>);

// `%` (Tensor and UniTensor) and `==` (Tensor) have no proxy instantiations at all -- see Mod.cpp
// and Cpr.cpp, which specialize only the dtype scalars and Scalar.
static_assert(cytnx_scalar_value<cytnx_double>);
static_assert(cytnx_scalar_value<Scalar>);
static_assert(!cytnx_scalar_value<Tensor::Tproxy>);
static_assert(!cytnx_scalar_value<Scalar::Sproxy>);
static_assert(!cytnx_scalar_value<std::string>);

// A type with no conversion to Tensor / UniTensor / any scalar-like type, so it can only reach the
// arithmetic operators through the (now constrained) `template <class T>` overloads. (std::string
// and containers are unsuitable here: Tensor has implicit constructors that would let them bind to
// the non-template Tensor <op> Tensor overloads instead.)
namespace {
  struct NotScalar {};
}  // namespace

// Named helper concepts so each operator-viability check is a proper (SFINAE) requires context.
template <class L, class R>
concept cy_addable = requires(L l, R r) {
  l + r;
};
template <class L, class R>
concept cy_mulable = requires(L l, R r) {
  l* r;
};
template <class L, class R>
concept cy_subable = requires(L l, R r) {
  l - r;
};
template <class L, class R>
concept cy_divable = requires(L l, R r) {
  l / r;
};
template <class L, class R>
concept cy_eqable = requires(L l, R r) {
  l == r;
};

// The scalar-on-the-left free operators (`scalar <op> Tensor/UniTensor`) can only be the free
// namespace-scope operators this PR constrains -- a scalar left operand cannot bind a member
// operator. They are no longer viable for a non-scalar operand (they were before #1003)...
static_assert(!cy_addable<NotScalar, Tensor>);
static_assert(!cy_mulable<NotScalar, Tensor>);
static_assert(!cy_eqable<NotScalar, Tensor>);
static_assert(!cy_addable<NotScalar, UniTensor>);
static_assert(!cy_mulable<NotScalar, UniTensor>);
// ...but remain viable for scalar operands.
static_assert(cy_addable<double, Tensor>);
static_assert(cy_mulable<double, Tensor>);
static_assert(cy_addable<Scalar, Tensor>);
static_assert(cy_mulable<double, UniTensor>);

// The scalar-on-the-RIGHT direction (`Tensor <op> non_scalar`) -- Ian's review item on #1093, and
// the gap that made the constraint above cosmetic. Tensor.hpp forward-declares these same free
// operators; while those forward declarations were unconstrained they formed a *distinct*,
// unconstrained template, which stayed viable for every non-scalar T and reintroduced exactly the
// undefined-reference failure the constraint was meant to turn into an overload-resolution
// failure. The forward declarations now carry the identical constraint, so this direction is
// rejected too. (They are load-bearing: Tensor's inline member templates call the free operators,
// e.g. `Tensor Mul(const T&) { return *this * rhs; }` -- operator* has no member form. Because
// `*this * rhs` is a dependent expression it binds by ADL at the point of instantiation, so the
// declarations work equally well below the class, which is where cytnx_scalar_like can name the
// now-complete Tensor::Tproxy.)
static_assert(!cy_addable<Tensor, NotScalar>);
static_assert(!cy_mulable<Tensor, NotScalar>);
static_assert(!cy_subable<Tensor, NotScalar>);
static_assert(!cy_divable<Tensor, NotScalar>);
static_assert(!cy_addable<UniTensor, NotScalar>);
static_assert(!cy_mulable<UniTensor, NotScalar>);
// ...while scalar right-operands stay viable in the same direction.
static_assert(cy_addable<Tensor, double>);
static_assert(cy_mulable<Tensor, double>);
static_assert(cy_subable<Tensor, Scalar>);
static_assert(cy_divable<Tensor, cytnx_int64>);
static_assert(cy_addable<UniTensor, double>);

TEST(OperatorConstraint, scalar_operators_still_work) {
  Tensor a = arange(0, 4, 1, Type.Double);  // [0,1,2,3]
  Tensor b = a + 1.0;
  EXPECT_DOUBLE_EQ(b.at<cytnx_double>({0}), 1.0);
  EXPECT_DOUBLE_EQ(b.at<cytnx_double>({3}), 4.0);

  Tensor c = 2.0 * a;
  EXPECT_DOUBLE_EQ(c.at<cytnx_double>({3}), 6.0);

  // Tensor <op> Tensor uses the non-template overloads and is unaffected.
  Tensor d = a + a;
  EXPECT_DOUBLE_EQ(d.at<cytnx_double>({3}), 6.0);
}
