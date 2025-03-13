#include "MaxMin_internal.hpp"
#include "cytnx_error.hpp"
#include "backend/lapack_wrapper.hpp"
#include <cfloat>
#include <limits>

using namespace std;

namespace cytnx {
  namespace linalg_internal {

    void MaxMin_internal_u64(boost::intrusive_ptr<Storage_base> &out,
                             const boost::intrusive_ptr<Storage_base> &ten,
                             const cytnx_uint64 &Nelem, const char &type) {
      cytnx_uint64 *_ten = (cytnx_uint64 *)ten->data();
      cytnx_uint64 *_out = (cytnx_uint64 *)out->data();

      if (type == 'x') {
        _out[0] = numeric_limits<cytnx_uint64>::min();
        for (cytnx_uint64 n = 0; n < Nelem; n++) {
          if (_ten[n] > _out[0]) _out[0] = _ten[n];
        }
      } else if (type == 'n') {
        _out[0] = numeric_limits<cytnx_uint64>::max();
        for (cytnx_uint64 n = 0; n < Nelem; n++) {
          if (_ten[n] < _out[0]) _out[0] = _ten[n];
        }
      }
    }

    void MaxMin_internal_i64(boost::intrusive_ptr<Storage_base> &out,
                             const boost::intrusive_ptr<Storage_base> &ten,
                             const cytnx_uint64 &Nelem, const char &type) {
      cytnx_int64 *_ten = (cytnx_int64 *)ten->data();
      cytnx_int64 *_out = (cytnx_int64 *)out->data();

      if (type == 'x') {
        _out[0] = numeric_limits<cytnx_int64>::min();
        for (cytnx_uint64 n = 0; n < Nelem; n++) {
          if (_ten[n] > _out[0]) _out[0] = _ten[n];
        }
      } else if (type == 'n') {
        _out[0] = numeric_limits<cytnx_int64>::max();
        for (cytnx_uint64 n = 0; n < Nelem; n++) {
          if (_ten[n] < _out[0]) _out[0] = _ten[n];
        }
      }
    }

    void MaxMin_internal_u32(boost::intrusive_ptr<Storage_base> &out,
                             const boost::intrusive_ptr<Storage_base> &ten,
                             const cytnx_uint64 &Nelem, const char &type) {
      cytnx_uint32 *_ten = (cytnx_uint32 *)ten->data();
      cytnx_uint32 *_out = (cytnx_uint32 *)out->data();

      if (type == 'x') {
        _out[0] = numeric_limits<cytnx_uint32>::min();
        for (cytnx_uint64 n = 0; n < Nelem; n++) {
          if (_ten[n] > _out[0]) _out[0] = _ten[n];
        }
      } else if (type == 'n') {
        _out[0] = numeric_limits<cytnx_uint32>::max();
        for (cytnx_uint64 n = 0; n < Nelem; n++) {
          if (_ten[n] < _out[0]) _out[0] = _ten[n];
        }
      }
    }

    void MaxMin_internal_i32(boost::intrusive_ptr<Storage_base> &out,
                             const boost::intrusive_ptr<Storage_base> &ten,
                             const cytnx_uint64 &Nelem, const char &type) {
      cytnx_int32 *_ten = (cytnx_int32 *)ten->data();
      cytnx_int32 *_out = (cytnx_int32 *)out->data();

      if (type == 'x') {
        _out[0] = numeric_limits<cytnx_int32>::min();
        for (cytnx_uint64 n = 0; n < Nelem; n++) {
          if (_ten[n] > _out[0]) _out[0] = _ten[n];
        }
      } else if (type == 'n') {
        _out[0] = numeric_limits<cytnx_int32>::max();
        for (cytnx_uint64 n = 0; n < Nelem; n++) {
          if (_ten[n] < _out[0]) _out[0] = _ten[n];
        }
      }
    }

    void MaxMin_internal_u16(boost::intrusive_ptr<Storage_base> &out,
                             const boost::intrusive_ptr<Storage_base> &ten,
                             const cytnx_uint64 &Nelem, const char &type) {
      cytnx_uint16 *_ten = (cytnx_uint16 *)ten->data();
      cytnx_uint16 *_out = (cytnx_uint16 *)out->data();

      if (type == 'x') {
        _out[0] = numeric_limits<cytnx_uint16>::min();
        for (cytnx_uint64 n = 0; n < Nelem; n++) {
          if (_ten[n] > _out[0]) _out[0] = _ten[n];
        }
      } else if (type == 'n') {
        _out[0] = numeric_limits<cytnx_uint16>::max();
        for (cytnx_uint64 n = 0; n < Nelem; n++) {
          if (_ten[n] < _out[0]) _out[0] = _ten[n];
        }
      }
    }

    void MaxMin_internal_i16(boost::intrusive_ptr<Storage_base> &out,
                             const boost::intrusive_ptr<Storage_base> &ten,
                             const cytnx_uint64 &Nelem, const char &type) {
      cytnx_int16 *_ten = (cytnx_int16 *)ten->data();
      cytnx_int16 *_out = (cytnx_int16 *)out->data();

      if (type == 'x') {
        _out[0] = numeric_limits<cytnx_int16>::min();
        for (cytnx_uint64 n = 0; n < Nelem; n++) {
          if (_ten[n] > _out[0]) _out[0] = _ten[n];
        }
      } else if (type == 'n') {
        _out[0] = numeric_limits<cytnx_int16>::max();
        for (cytnx_uint64 n = 0; n < Nelem; n++) {
          if (_ten[n] < _out[0]) _out[0] = _ten[n];
        }
      }
    }

    void MaxMin_internal_d(boost::intrusive_ptr<Storage_base> &out,
                           const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &Nelem,
                           const char &type) {
      cytnx_double *_ten = (cytnx_double *)ten->data();
      cytnx_double *_out = (cytnx_double *)out->data();

      if (type == 'x') {
        _out[0] = DBL_MIN;
        for (cytnx_uint64 n = 0; n < Nelem; n++) {
          if (_ten[n] > _out[0]) _out[0] = _ten[n];
        }
      } else if (type == 'n') {
        _out[0] = DBL_MAX;
        for (cytnx_uint64 n = 0; n < Nelem; n++) {
          if (_ten[n] < _out[0]) _out[0] = _ten[n];
        }
      }
    }

    void MaxMin_internal_f(boost::intrusive_ptr<Storage_base> &out,
                           const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &Nelem,
                           const char &type) {
      cytnx_float *_ten = (cytnx_float *)ten->data();
      cytnx_float *_out = (cytnx_float *)out->data();

      if (type == 'x') {
        _out[0] = FLT_MIN;
        for (cytnx_uint64 n = 0; n < Nelem; n++) {
          if (_ten[n] > _out[0]) _out[0] = _ten[n];
        }
      } else if (type == 'n') {
        _out[0] = FLT_MAX;
        for (cytnx_uint64 n = 0; n < Nelem; n++) {
          if (_ten[n] < _out[0]) _out[0] = _ten[n];
        }
      }
    }
    void MaxMin_internal_cd(boost::intrusive_ptr<Storage_base> &out,
                            const boost::intrusive_ptr<Storage_base> &ten,
                            const cytnx_uint64 &Nelem, const char &type) {
      cytnx_complex128 *_ten = (cytnx_complex128 *)ten->data();
      cytnx_complex128 *_out = (cytnx_complex128 *)out->data();

      if (type == 'x') {
        _out[0] = cytnx_complex128(DBL_MIN, DBL_MIN);
        for (cytnx_uint64 n = 0; n < Nelem; n++) {
          if (_ten[n].real() >= _out[0].real()) _out[0] = _ten[n];
        }
      } else if (type == 'n') {
        _out[0] = cytnx_complex128(DBL_MAX, DBL_MAX);
        for (cytnx_uint64 n = 0; n < Nelem; n++) {
          if (_ten[n].real() <= _out[0].real()) _out[0] = _ten[n];
        }
      }
    }

    void MaxMin_internal_cf(boost::intrusive_ptr<Storage_base> &out,
                            const boost::intrusive_ptr<Storage_base> &ten,
                            const cytnx_uint64 &Nelem, const char &type) {
      cytnx_complex64 *_ten = (cytnx_complex64 *)ten->data();
      cytnx_complex64 *_out = (cytnx_complex64 *)out->data();

      if (type == 'x') {
        _out[0] = cytnx_complex64(FLT_MIN, FLT_MIN);
        for (cytnx_uint64 n = 0; n < Nelem; n++) {
          if (_ten[n].real() >= _out[0].real()) _out[0] = _ten[n];
        }
      } else if (type == 'n') {
        _out[0] = cytnx_complex64(FLT_MAX, FLT_MAX);
        for (cytnx_uint64 n = 0; n < Nelem; n++) {
          if (_ten[n].real() <= _out[0].real()) _out[0] = _ten[n];
        }
      }
    }

    void MaxMin_internal_b(boost::intrusive_ptr<Storage_base> &out,
                           const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &Nelem,
                           const char &type) {
      cytnx_error_msg(true, "[ERROR][Maxmin_internal] bool does not have Min/Max support.%s", "\n");
    }

  }  // namespace linalg_internal

}  // namespace cytnx
