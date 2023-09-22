#ifndef __Axpy_internal_H__
#define __Axpy_internal_H__

#include <assert.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include "backend/Storage.hpp"
#include "Type.hpp"
#include "backend/Scalar.hpp"

namespace cytnx {

  namespace linalg_internal {

    /// Axpy
    void Axpy_internal_cd(const boost::intrusive_ptr<Storage_base> &x,
                          boost::intrusive_ptr<Storage_base> &y, const Scalar &a);
    void Axpy_internal_cf(const boost::intrusive_ptr<Storage_base> &x,
                          boost::intrusive_ptr<Storage_base> &y, const Scalar &a);

    void Axpy_internal_d(const boost::intrusive_ptr<Storage_base> &x,
                         boost::intrusive_ptr<Storage_base> &y, const Scalar &a);

    void Axpy_internal_f(const boost::intrusive_ptr<Storage_base> &x,
                         boost::intrusive_ptr<Storage_base> &y, const Scalar &a);

  }  // namespace linalg_internal
}  // namespace cytnx

#endif
