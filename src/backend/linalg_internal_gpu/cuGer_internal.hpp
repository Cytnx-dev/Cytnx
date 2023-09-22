#ifndef __cuGer_internal_H__
#define __cuGer_internal_H__

#include <assert.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include "backend/Storage.hpp"
#include "Type.hpp"
#include "backend/Scalar.hpp"

namespace cytnx {

  namespace linalg_internal {

    /// cuGer
    void cuGer_internal_cd(boost::intrusive_ptr<Storage_base> &A,
                           const boost::intrusive_ptr<Storage_base> &x,
                           const boost::intrusive_ptr<Storage_base> &y, const Scalar &a);

    void cuGer_internal_cf(boost::intrusive_ptr<Storage_base> &A,
                           const boost::intrusive_ptr<Storage_base> &x,
                           const boost::intrusive_ptr<Storage_base> &y, const Scalar &a);

    void cuGer_internal_d(boost::intrusive_ptr<Storage_base> &A,
                          const boost::intrusive_ptr<Storage_base> &x,
                          const boost::intrusive_ptr<Storage_base> &y, const Scalar &a);

    void cuGer_internal_f(boost::intrusive_ptr<Storage_base> &A,
                          const boost::intrusive_ptr<Storage_base> &x,
                          const boost::intrusive_ptr<Storage_base> &y, const Scalar &a);

  }  // namespace linalg_internal
}  // namespace cytnx

#endif
