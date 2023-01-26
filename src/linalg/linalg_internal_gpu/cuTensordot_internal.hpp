#ifndef __cuTensordot_internal_H__
#define __cuTensordot_internal_H__

#include <assert.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include "Storage.hpp"
#include "Type.hpp"

namespace cytnx {

  namespace linalg_internal {

    /// cuTensordot
    void cuTensordot_internal_cd(boost::intrusive_ptr<Storage_base> &out,
                                 const boost::intrusive_ptr<Storage_base> &Lin,
                                 const boost::intrusive_ptr<Storage_base> &Rin,
                                 const unsigned long long &len, const bool &is_conj);
    void cuTensordot_internal_cf(boost::intrusive_ptr<Storage_base> &out,
                                 const boost::intrusive_ptr<Storage_base> &Lin,
                                 const boost::intrusive_ptr<Storage_base> &Rin,
                                 const unsigned long long &len, const bool &is_conj);
    void cuTensordot_internal_d(boost::intrusive_ptr<Storage_base> &out,
                                const boost::intrusive_ptr<Storage_base> &Lin,
                                const boost::intrusive_ptr<Storage_base> &Rin,
                                const unsigned long long &len, const bool &is_conj);
    void cuTensordot_internal_f(boost::intrusive_ptr<Storage_base> &out,
                                const boost::intrusive_ptr<Storage_base> &Lin,
                                const boost::intrusive_ptr<Storage_base> &Rin,
                                const unsigned long long &len, const bool &is_conj);

    


  }  // namespace linalg_internal
}  // namespace cytnx

#endif
