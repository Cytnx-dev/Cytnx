#ifndef __is__H_
#define __is__H_

#include "Type.hpp"
#include "cytnx_error.hpp"
#include "Tensor.hpp"
#include "Storage.hpp"
#include "Bond.hpp"
#include "Symmetry.hpp"
namespace cytnx {

  bool is(const Tensor& L, const Tensor& R);
  bool is(const Storage& L, const Storage& R);
  bool is(const Bond& L, const Bond& R);
  bool is(const Symmetry& L, const Symmetry& R);

}  // namespace cytnx

#endif
