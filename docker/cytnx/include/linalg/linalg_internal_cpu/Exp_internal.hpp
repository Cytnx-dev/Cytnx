#ifndef __Exp_internal_H__
#define __Exp_internal_H__

#include <assert.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include "Storage.hpp"
#include "Type.hpp"

namespace cytnx {
  namespace linalg_internal {

    void Exp_internal_d(boost::intrusive_ptr<Storage_base> &out,
                        const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &Nelem);

    void Exp_internal_f(boost::intrusive_ptr<Storage_base> &out,
                        const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &Nelem);

    void Exp_internal_cd(boost::intrusive_ptr<Storage_base> &out,
                         const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &Nelem);

    void Exp_internal_cf(boost::intrusive_ptr<Storage_base> &out,
                         const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &Nelem);

  }  // namespace linalg_internal

}  // namespace cytnx

#endif
