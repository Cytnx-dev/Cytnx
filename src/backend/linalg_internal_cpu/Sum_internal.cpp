#include "Sum_internal.hpp"

#include "boost/smart_ptr/intrusive_ptr.hpp"

#include "backend/Storage.hpp"
#include "cytnx_error.hpp"
#include "Type.hpp"

using namespace std;

namespace cytnx {
  namespace linalg_internal {

    void Sum_internal_u64(boost::intrusive_ptr<Storage_base>& out,
                          const boost::intrusive_ptr<Storage_base>& ten,
                          const cytnx_uint64& Nelem) {
      cytnx_uint64* _ten = (cytnx_uint64*)ten->data();
      cytnx_uint64* _out = (cytnx_uint64*)out->data();

      _out[0] = 0;
      for (cytnx_uint64 n = 0; n < Nelem; n++) {
        _out[0] += _ten[n];
      }
    }

    void Sum_internal_i64(boost::intrusive_ptr<Storage_base>& out,
                          const boost::intrusive_ptr<Storage_base>& ten,
                          const cytnx_uint64& Nelem) {
      cytnx_int64* _ten = (cytnx_int64*)ten->data();
      cytnx_int64* _out = (cytnx_int64*)out->data();

      _out[0] = 0;
      for (cytnx_uint64 n = 0; n < Nelem; n++) {
        _out[0] += _ten[n];
      }
    }

    void Sum_internal_u32(boost::intrusive_ptr<Storage_base>& out,
                          const boost::intrusive_ptr<Storage_base>& ten,
                          const cytnx_uint64& Nelem) {
      cytnx_uint32* _ten = (cytnx_uint32*)ten->data();
      cytnx_uint32* _out = (cytnx_uint32*)out->data();

      _out[0] = 0;
      for (cytnx_uint64 n = 0; n < Nelem; n++) {
        _out[0] += _ten[n];
      }
    }

    void Sum_internal_i32(boost::intrusive_ptr<Storage_base>& out,
                          const boost::intrusive_ptr<Storage_base>& ten,
                          const cytnx_uint64& Nelem) {
      cytnx_int32* _ten = (cytnx_int32*)ten->data();
      cytnx_int32* _out = (cytnx_int32*)out->data();

      _out[0] = 0;
      for (cytnx_uint64 n = 0; n < Nelem; n++) {
        _out[0] += _ten[n];
      }
    }

    void Sum_internal_u16(boost::intrusive_ptr<Storage_base>& out,
                          const boost::intrusive_ptr<Storage_base>& ten,
                          const cytnx_uint64& Nelem) {
      cytnx_uint16* _ten = (cytnx_uint16*)ten->data();
      cytnx_uint16* _out = (cytnx_uint16*)out->data();

      _out[0] = 0;
      for (cytnx_uint64 n = 0; n < Nelem; n++) {
        _out[0] += _ten[n];
      }
    }

    void Sum_internal_i16(boost::intrusive_ptr<Storage_base>& out,
                          const boost::intrusive_ptr<Storage_base>& ten,
                          const cytnx_uint64& Nelem) {
      cytnx_int16* _ten = (cytnx_int16*)ten->data();
      cytnx_int16* _out = (cytnx_int16*)out->data();

      _out[0] = 0;
      for (cytnx_uint64 n = 0; n < Nelem; n++) {
        _out[0] += _ten[n];
      }
    }

    void Sum_internal_d(boost::intrusive_ptr<Storage_base>& out,
                        const boost::intrusive_ptr<Storage_base>& ten, const cytnx_uint64& Nelem) {
      cytnx_double* _ten = (cytnx_double*)ten->data();
      cytnx_double* _out = (cytnx_double*)out->data();

      _out[0] = 0;
      for (cytnx_uint64 n = 0; n < Nelem; n++) {
        _out[0] += _ten[n];
      }
    }

    void Sum_internal_f(boost::intrusive_ptr<Storage_base>& out,
                        const boost::intrusive_ptr<Storage_base>& ten, const cytnx_uint64& Nelem) {
      cytnx_float* _ten = (cytnx_float*)ten->data();
      cytnx_float* _out = (cytnx_float*)out->data();

      _out[0] = 0;
      for (cytnx_uint64 n = 0; n < Nelem; n++) {
        _out[0] += _ten[n];
      }
    }
    void Sum_internal_cd(boost::intrusive_ptr<Storage_base>& out,
                         const boost::intrusive_ptr<Storage_base>& ten, const cytnx_uint64& Nelem) {
      cytnx_complex128* _ten = (cytnx_complex128*)ten->data();
      cytnx_complex128* _out = (cytnx_complex128*)out->data();

      _out[0] = 0;
      for (cytnx_uint64 n = 0; n < Nelem; n++) {
        _out[0] += _ten[n];
      }
    }

    void Sum_internal_cf(boost::intrusive_ptr<Storage_base>& out,
                         const boost::intrusive_ptr<Storage_base>& ten, const cytnx_uint64& Nelem) {
      cytnx_complex64* _ten = (cytnx_complex64*)ten->data();
      cytnx_complex64* _out = (cytnx_complex64*)out->data();

      _out[0] = 0;
      for (cytnx_uint64 n = 0; n < Nelem; n++) {
        _out[0] += _ten[n];
      }
    }

    void Sum_internal_b(boost::intrusive_ptr<Storage_base>& out,
                        const boost::intrusive_ptr<Storage_base>& ten, const cytnx_uint64& Nelem) {
      cytnx_error_msg(true, "[ERROR][Sum_internal] bool does not have Sum support.%s", "\n");
    }

  }  // namespace linalg_internal

}  // namespace cytnx
