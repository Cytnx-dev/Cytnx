#include "cuSum_internal.hpp"
#include "cytnx_error.hpp"
#include "Type.hpp"
#include "backend/lapack_wrapper.hpp"
#include "../utils_internal_gpu/cuReduce_gpu.hpp"

namespace cytnx {

  namespace linalg_internal {
    using namespace std;
    void cuSum_internal_cd(boost::intrusive_ptr<Storage_base> &out,
                           const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &Nelem,
                           const char &type) {
      utils_internal::cuReduce_gpu_cd((cytnx_complex128 *)out->Mem, (cytnx_complex128 *)ten->Mem,
                                      Nelem);
    }
    void cuSum_internal_cf(boost::intrusive_ptr<Storage_base> &out,
                           const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &Nelem,
                           const char &type) {
      utils_internal::cuReduce_gpu_cf((cytnx_complex64 *)out->Mem, (cytnx_complex64 *)ten->Mem,
                                      Nelem);
    }
    void cuSum_internal_d(boost::intrusive_ptr<Storage_base> &out,
                          const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &Nelem,
                          const char &type) {
      utils_internal::cuReduce_gpu_d((cytnx_double *)out->Mem, (cytnx_double *)ten->Mem, Nelem);
    }
    void cuSum_internal_f(boost::intrusive_ptr<Storage_base> &out,
                          const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &Nelem,
                          const char &type) {
      utils_internal::cuReduce_gpu_f((cytnx_float *)out->Mem, (cytnx_float *)ten->Mem, Nelem);
    }
    void cuSum_internal_i64(boost::intrusive_ptr<Storage_base> &out,
                            const boost::intrusive_ptr<Storage_base> &ten,
                            const cytnx_uint64 &Nelem, const char &type) {
      utils_internal::cuReduce_gpu_i64((cytnx_int64 *)out->Mem, (cytnx_int64 *)ten->Mem, Nelem);
    }
    void cuSum_internal_u64(boost::intrusive_ptr<Storage_base> &out,
                            const boost::intrusive_ptr<Storage_base> &ten,
                            const cytnx_uint64 &Nelem, const char &type) {
      utils_internal::cuReduce_gpu_u64((cytnx_uint64 *)out->Mem, (cytnx_uint64 *)ten->Mem, Nelem);
    }
    void cuSum_internal_i32(boost::intrusive_ptr<Storage_base> &out,
                            const boost::intrusive_ptr<Storage_base> &ten,
                            const cytnx_uint64 &Nelem, const char &type) {
      utils_internal::cuReduce_gpu_i32((cytnx_int32 *)out->Mem, (cytnx_int32 *)ten->Mem, Nelem);
    }
    void cuSum_internal_u32(boost::intrusive_ptr<Storage_base> &out,
                            const boost::intrusive_ptr<Storage_base> &ten,
                            const cytnx_uint64 &Nelem, const char &type) {
      utils_internal::cuReduce_gpu_u32((cytnx_uint32 *)out->Mem, (cytnx_uint32 *)ten->Mem, Nelem);
    }
    void cuSum_internal_i16(boost::intrusive_ptr<Storage_base> &out,
                            const boost::intrusive_ptr<Storage_base> &ten,
                            const cytnx_uint64 &Nelem, const char &type) {
      utils_internal::cuReduce_gpu_i16((cytnx_int16 *)out->Mem, (cytnx_int16 *)ten->Mem, Nelem);
    }
    void cuSum_internal_u16(boost::intrusive_ptr<Storage_base> &out,
                            const boost::intrusive_ptr<Storage_base> &ten,
                            const cytnx_uint64 &Nelem, const char &type) {
      utils_internal::cuReduce_gpu_u16((cytnx_uint16 *)out->Mem, (cytnx_uint16 *)ten->Mem, Nelem);
    }
    void cuSum_internal_b(boost::intrusive_ptr<Storage_base> &out,
                          const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &Nelem,
                          const char &type) {
      cytnx_error_msg(
        true,
        "[ERROR] Sum cannot perform on Bool type. use astype() to convert to desire type first.%s",
        "\n");
    }

  }  // namespace linalg_internal
}  // namespace cytnx
