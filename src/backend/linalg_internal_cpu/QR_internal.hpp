#ifndef CYTNX_BACKEND_LINALG_INTERNAL_CPU_QR_INTERNAL_H_
#define CYTNX_BACKEND_LINALG_INTERNAL_CPU_QR_INTERNAL_H_

#include <assert.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include "backend/Storage.hpp"
#include "Type.hpp"

namespace cytnx {

  namespace linalg_internal {

    /// QR
    void QR_internal_cd(const boost::intrusive_ptr<Storage_base>& in,
                        boost::intrusive_ptr<Storage_base>& Q,
                        boost::intrusive_ptr<Storage_base>& R,
                        boost::intrusive_ptr<Storage_base>& D,
                        boost::intrusive_ptr<Storage_base>& tau, const cytnx_int64& M,
                        const cytnx_int64& N, const bool& is_d);
    void QR_internal_cf(const boost::intrusive_ptr<Storage_base>& in,
                        boost::intrusive_ptr<Storage_base>& Q,
                        boost::intrusive_ptr<Storage_base>& R,
                        boost::intrusive_ptr<Storage_base>& D,
                        boost::intrusive_ptr<Storage_base>& tau, const cytnx_int64& M,
                        const cytnx_int64& N, const bool& is_d);
    void QR_internal_d(const boost::intrusive_ptr<Storage_base>& in,
                       boost::intrusive_ptr<Storage_base>& Q, boost::intrusive_ptr<Storage_base>& R,
                       boost::intrusive_ptr<Storage_base>& D,
                       boost::intrusive_ptr<Storage_base>& tau, const cytnx_int64& M,
                       const cytnx_int64& N, const bool& is_d);
    void QR_internal_f(const boost::intrusive_ptr<Storage_base>& in,
                       boost::intrusive_ptr<Storage_base>& Q, boost::intrusive_ptr<Storage_base>& R,
                       boost::intrusive_ptr<Storage_base>& D,
                       boost::intrusive_ptr<Storage_base>& tau, const cytnx_int64& M,
                       const cytnx_int64& N, const bool& is_d);

  }  // namespace linalg_internal
}  // namespace cytnx

#endif  // CYTNX_BACKEND_LINALG_INTERNAL_CPU_QR_INTERNAL_H_
