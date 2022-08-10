#ifndef __Vectordot_internal_H__
#define __Vectordot_internal_H__

#include <assert.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include "Storage.hpp"
#include "Type.hpp"

namespace cytnx {

  namespace linalg_internal {

    void Vectordot_internal_cd(boost::intrusive_ptr<Storage_base> &out,
                               const boost::intrusive_ptr<Storage_base> &Lin,
                               const boost::intrusive_ptr<Storage_base> &Rin,
                               const unsigned long long &len, const bool &is_conj);
    void Vectordot_internal_cf(boost::intrusive_ptr<Storage_base> &out,
                               const boost::intrusive_ptr<Storage_base> &Lin,
                               const boost::intrusive_ptr<Storage_base> &Rin,
                               const unsigned long long &len, const bool &is_conj);
    void Vectordot_internal_d(boost::intrusive_ptr<Storage_base> &out,
                              const boost::intrusive_ptr<Storage_base> &Lin,
                              const boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const bool &is_conj);
    void Vectordot_internal_f(boost::intrusive_ptr<Storage_base> &out,
                              const boost::intrusive_ptr<Storage_base> &Lin,
                              const boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const bool &is_conj);
    void Vectordot_internal_i64(boost::intrusive_ptr<Storage_base> &out,
                                const boost::intrusive_ptr<Storage_base> &Lin,
                                const boost::intrusive_ptr<Storage_base> &Rin,
                                const unsigned long long &len, const bool &is_conj);
    void Vectordot_internal_u64(boost::intrusive_ptr<Storage_base> &out,
                                const boost::intrusive_ptr<Storage_base> &Lin,
                                const boost::intrusive_ptr<Storage_base> &Rin,
                                const unsigned long long &len, const bool &is_conj);
    void Vectordot_internal_i32(boost::intrusive_ptr<Storage_base> &out,
                                const boost::intrusive_ptr<Storage_base> &Lin,
                                const boost::intrusive_ptr<Storage_base> &Rin,
                                const unsigned long long &len, const bool &is_conj);
    void Vectordot_internal_u32(boost::intrusive_ptr<Storage_base> &out,
                                const boost::intrusive_ptr<Storage_base> &Lin,
                                const boost::intrusive_ptr<Storage_base> &Rin,
                                const unsigned long long &len, const bool &is_conj);
    void Vectordot_internal_i16(boost::intrusive_ptr<Storage_base> &out,
                                const boost::intrusive_ptr<Storage_base> &Lin,
                                const boost::intrusive_ptr<Storage_base> &Rin,
                                const unsigned long long &len, const bool &is_conj);
    void Vectordot_internal_u16(boost::intrusive_ptr<Storage_base> &out,
                                const boost::intrusive_ptr<Storage_base> &Lin,
                                const boost::intrusive_ptr<Storage_base> &Rin,
                                const unsigned long long &len, const bool &is_conj);
    void Vectordot_internal_b(boost::intrusive_ptr<Storage_base> &out,
                              const boost::intrusive_ptr<Storage_base> &Lin,
                              const boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const bool &is_conj);

  }  // namespace linalg_internal
}  // namespace cytnx

#endif
