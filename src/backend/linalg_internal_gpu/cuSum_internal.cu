#include "backend/linalg_internal_gpu/cuSum_internal.hpp"

#include "boost/smart_ptr/intrusive_ptr.hpp"

#include "backend/Storage.hpp"
#include "backend/utils_internal_gpu/cuReduce_gpu.hpp"
#include "cytnx_error.hpp"
#include "Type.hpp"

namespace cytnx {
  namespace linalg_internal {
    void cuSum_internal_cd(boost::intrusive_ptr<Storage_base> &out,
                           const boost::intrusive_ptr<Storage_base> &ten,
                           const cytnx_uint64 &Nelem) {
      utils_internal::cuReduce_gpu((cytnx_complex128 *)out->data(), (cytnx_complex128 *)ten->data(),
                                   Nelem);
    }
    void cuSum_internal_cf(boost::intrusive_ptr<Storage_base> &out,
                           const boost::intrusive_ptr<Storage_base> &ten,
                           const cytnx_uint64 &Nelem) {
      utils_internal::cuReduce_gpu((cytnx_complex64 *)out->data(), (cytnx_complex64 *)ten->data(),
                                   Nelem);
    }
    void cuSum_internal_d(boost::intrusive_ptr<Storage_base> &out,
                          const boost::intrusive_ptr<Storage_base> &ten,
                          const cytnx_uint64 &Nelem) {
      utils_internal::cuReduce_gpu((cytnx_double *)out->data(), (cytnx_double *)ten->data(), Nelem);
    }
    void cuSum_internal_f(boost::intrusive_ptr<Storage_base> &out,
                          const boost::intrusive_ptr<Storage_base> &ten,
                          const cytnx_uint64 &Nelem) {
      utils_internal::cuReduce_gpu((cytnx_float *)out->data(), (cytnx_float *)ten->data(), Nelem);
    }
    void cuSum_internal_i64(boost::intrusive_ptr<Storage_base> &out,
                            const boost::intrusive_ptr<Storage_base> &ten,
                            const cytnx_uint64 &Nelem) {
      utils_internal::cuReduce_gpu((cytnx_int64 *)out->data(), (cytnx_int64 *)ten->data(), Nelem);
    }
    void cuSum_internal_u64(boost::intrusive_ptr<Storage_base> &out,
                            const boost::intrusive_ptr<Storage_base> &ten,
                            const cytnx_uint64 &Nelem) {
      utils_internal::cuReduce_gpu((cytnx_uint64 *)out->data(), (cytnx_uint64 *)ten->data(), Nelem);
    }
    void cuSum_internal_i32(boost::intrusive_ptr<Storage_base> &out,
                            const boost::intrusive_ptr<Storage_base> &ten,
                            const cytnx_uint64 &Nelem) {
      utils_internal::cuReduce_gpu((cytnx_int32 *)out->data(), (cytnx_int32 *)ten->data(), Nelem);
    }
    void cuSum_internal_u32(boost::intrusive_ptr<Storage_base> &out,
                            const boost::intrusive_ptr<Storage_base> &ten,
                            const cytnx_uint64 &Nelem) {
      utils_internal::cuReduce_gpu((cytnx_uint32 *)out->data(), (cytnx_uint32 *)ten->data(), Nelem);
    }
    void cuSum_internal_i16(boost::intrusive_ptr<Storage_base> &out,
                            const boost::intrusive_ptr<Storage_base> &ten,
                            const cytnx_uint64 &Nelem) {
      utils_internal::cuReduce_gpu((cytnx_int16 *)out->data(), (cytnx_int16 *)ten->data(), Nelem);
    }
    void cuSum_internal_u16(boost::intrusive_ptr<Storage_base> &out,
                            const boost::intrusive_ptr<Storage_base> &ten,
                            const cytnx_uint64 &Nelem) {
      utils_internal::cuReduce_gpu((cytnx_uint16 *)out->data(), (cytnx_uint16 *)ten->data(), Nelem);
    }
    void cuSum_internal_b(boost::intrusive_ptr<Storage_base> &out,
                          const boost::intrusive_ptr<Storage_base> &ten,
                          const cytnx_uint64 &Nelem) {
      cytnx_error_msg(
        true,
        "[ERROR] Sum cannot perform on Bool type. use astype() to convert to desire type first.%s",
        "\n");
    }

  }  // namespace linalg_internal
}  // namespace cytnx
