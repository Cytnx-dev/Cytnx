#include "utils/is.hpp"

namespace cytnx {

  bool is(const Tensor &L, const Tensor &R) { return (L._impl == R._impl); }

  bool is(const Storage &L, const Storage &R) { return (L._impl == R._impl); }
}  // namespace cytnx

namespace cytnx {
  bool is(const cytnx::Bond &L, const cytnx::Bond &R) { return (L._impl == R._impl); }

  bool is(const cytnx::Symmetry &L, const cytnx::Symmetry &R) { return (L._impl == R._impl); }
  bool is(const cytnx::UniTensor &L, const cytnx::UniTensor &R) { return (L._impl == R._impl); }
}  // namespace cytnx
