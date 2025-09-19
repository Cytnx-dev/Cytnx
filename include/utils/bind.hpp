#ifndef CYTNX_UTILS_BIND_H_
#define CYTNX_UTILS_BIND_H_

#include "Type.hpp"
#include "Tensor.hpp"

#include "Bond.hpp"
#include "Symmetry.hpp"
#include "UniTensor.hpp"

#ifdef BACKEND_TORCH
#else
  #include "backend/Storage.hpp"
namespace cytnx {

  bool is(const Tensor& L, const Tensor& R);
  bool is(const Storage& L, const Storage& R);

}  // namespace cytnx

namespace cytnx {
  bool is(const cytnx::Bond& L, const cytnx::Bond& R);
  bool is(const cytnx::Symmetry& L, const cytnx::Symmetry& R);
  bool is(const UniTensor& L, const UniTensor& R);
}  // namespace cytnx

#endif  // BACKEND_TORCH

#endif  // CYTNX_UTILS_BIND_H_
