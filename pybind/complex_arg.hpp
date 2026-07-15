#pragma once

// A thin wrapper around cytnx_complex128 that exists purely to carry a
// corrected pybind11-stubgen parameter annotation. pybind11's own
// type_caster<std::complex<T>> (pybind11/complex.h) hardcodes the
// PARAMETER (input) annotation as "typing.SupportsComplex |
// typing.SupportsFloat | typing.SupportsIndex", omitting builtin `complex`
// -- typeshed's `complex` type only gained `__complex__` in Python >= 3.11,
// so under this project's Python 3.10 floor a stub generated from that
// caster rejects a plain complex literal (e.g. `ExpH(t, 2+1j)`) even though
// the caster's own load() accepts it directly via PyComplex_AsCComplex. The
// RETURN (output) annotation is already correctly "complex".
//
// ComplexArg delegates load()/cast() entirely to pybind11's own
// type_caster<cytnx_complex128> (no behavior change) and only overrides the
// declared parameter annotation, so it is a drop-in replacement for
// cytnx_complex128 in any .def() parameter that should accept a plain
// Python complex literal.

#include <pybind11/complex.h>
#include <pybind11/pybind11.h>

#include "cytnx.hpp"

namespace pybind_cytnx {

  struct ComplexArg {
    cytnx::cytnx_complex128 value;
    ComplexArg() = default;
    ComplexArg(cytnx::cytnx_complex128 v) : value(v) {}
    operator cytnx::cytnx_complex128() const { return value; }
  };

}  // namespace pybind_cytnx

namespace pybind11 {
  namespace detail {

    template <>
    struct type_caster<pybind_cytnx::ComplexArg> {
     public:
      PYBIND11_TYPE_CASTER(pybind_cytnx::ComplexArg,
                           io_name("complex | typing.SupportsComplex | typing.SupportsFloat | "
                                   "typing.SupportsIndex",
                                   "complex"));

      bool load(handle src, bool convert) {
        make_caster<cytnx::cytnx_complex128> inner;
        if (!inner.load(src, convert)) return false;
        value = pybind_cytnx::ComplexArg(cast_op<cytnx::cytnx_complex128>(std::move(inner)));
        return true;
      }

      static handle cast(const pybind_cytnx::ComplexArg &src, return_value_policy policy,
                         handle parent) {
        return make_caster<cytnx::cytnx_complex128>::cast(src.value, policy, parent);
      }
    };

  }  // namespace detail
}  // namespace pybind11
