#ifndef __is__H_
#define __is__H_

#include "Type.hpp"
#include "Tensor.hpp"
#include "Storage.hpp"
#include "Bond.hpp"
#include "Symmetry.hpp"
#include "UniTensor.hpp"

namespace cytnx {

  bool is(const Tensor &L, const Tensor &R);
  bool is(const Storage &L, const Storage &R);

}  // namespace cytnx

namespace cytnx {
  bool is(const cytnx::Bond &L, const cytnx::Bond &R);
  bool is(const cytnx::Symmetry &L, const cytnx::Symmetry &R);
  bool is(const UniTensor &L, const UniTensor &R);
}  // namespace cytnx

#endif
