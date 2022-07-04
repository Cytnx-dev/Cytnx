#include "linalg/linalg_internal_cpu/Mul_internal.hpp"
#include "utils/utils_internal_interface.hpp"
#include "utils/utils.hpp"
#ifdef UNI_OMP
  #include <omp.h>
#endif

namespace cytnx {

  namespace linalg_internal {

    /// Mul
    void Mul_internal_cdtcd(boost::intrusive_ptr<Storage_base> &out,
                            boost::intrusive_ptr<Storage_base> &Lin,
                            boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                            const std::vector<cytnx_uint64> &shape,
                            const std::vector<cytnx_uint64> &invmapper_L,
                            const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_complex128 *_out = (cytnx_complex128 *)out->Mem;
      cytnx_complex128 *_Lin = (cytnx_complex128 *)Lin->Mem;
      cytnx_complex128 *_Rin = (cytnx_complex128 *)Rin->Mem;
      if (Lin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[0] * _Rin[i];
        }
      } else if (Rin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[i] * _Rin[0];
        }
      } else {
        if (shape.size() == 0) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (unsigned long long i = 0; i < len; i++) {
            _out[i] = _Lin[i] * _Rin[i];
          }
        } else {
          /// handle non-contiguous:
          std::vector<cytnx_uint64> accu_shape(shape.size());
          std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
          cytnx_uint64 tmp1 = 1, tmp2 = 1, tmp3 = 1;
          for (cytnx_uint64 i = 0; i < shape.size(); i++) {
            accu_shape[shape.size() - 1 - i] = tmp1;
            tmp1 *= shape[shape.size() - 1 - i];

            old_accu_shapeL[shape.size() - 1 - i] = tmp2;
            tmp2 *= shape[invmapper_L[shape.size() - 1 - i]];

            old_accu_shapeR[shape.size() - 1 - i] = tmp3;
            tmp3 *= shape[invmapper_R[shape.size() - 1 - i]];
          }

// handle non-contiguous
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (cytnx_uint64 i = 0; i < len; i++) {
            std::vector<cytnx_uint64> tmpv = c2cartesian(i, accu_shape);
            _out[i] = _Lin[cartesian2c(vec_map(tmpv, invmapper_L), old_accu_shapeL)] *
                      _Rin[cartesian2c(vec_map(tmpv, invmapper_R), old_accu_shapeR)];
          }
        }
      }
    }
    void Mul_internal_cdtcf(boost::intrusive_ptr<Storage_base> &out,
                            boost::intrusive_ptr<Storage_base> &Lin,
                            boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                            const std::vector<cytnx_uint64> &shape,
                            const std::vector<cytnx_uint64> &invmapper_L,
                            const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_complex128 *_out = (cytnx_complex128 *)out->Mem;
      cytnx_complex128 *_Lin = (cytnx_complex128 *)Lin->Mem;
      cytnx_complex64 *_Rin = (cytnx_complex64 *)Rin->Mem;

      if (Lin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[0] * _Rin[i];
        }
      } else if (Rin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[i] * _Rin[0];
        }
      } else {
        if (shape.size() == 0) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (unsigned long long i = 0; i < len; i++) {
            _out[i] = _Lin[i] * _Rin[i];
          }
        } else {
          /// handle non-contiguous:
          std::vector<cytnx_uint64> accu_shape(shape.size());
          std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
          cytnx_uint64 tmp1 = 1, tmp2 = 1, tmp3 = 1;
          for (cytnx_uint64 i = 0; i < shape.size(); i++) {
            accu_shape[shape.size() - 1 - i] = tmp1;
            tmp1 *= shape[shape.size() - 1 - i];

            old_accu_shapeL[shape.size() - 1 - i] = tmp2;
            tmp2 *= shape[invmapper_L[shape.size() - 1 - i]];

            old_accu_shapeR[shape.size() - 1 - i] = tmp3;
            tmp3 *= shape[invmapper_R[shape.size() - 1 - i]];
          }

// handle non-contiguous
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (cytnx_uint64 i = 0; i < len; i++) {
            std::vector<cytnx_uint64> tmpv = c2cartesian(i, accu_shape);
            _out[i] = _Lin[cartesian2c(vec_map(tmpv, invmapper_L), old_accu_shapeL)] *
                      _Rin[cartesian2c(vec_map(tmpv, invmapper_R), old_accu_shapeR)];
          }
        }
      }
    }
    void Mul_internal_cdtd(boost::intrusive_ptr<Storage_base> &out,
                           boost::intrusive_ptr<Storage_base> &Lin,
                           boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                           const std::vector<cytnx_uint64> &shape,
                           const std::vector<cytnx_uint64> &invmapper_L,
                           const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_complex128 *_out = (cytnx_complex128 *)out->Mem;
      cytnx_complex128 *_Lin = (cytnx_complex128 *)Lin->Mem;
      cytnx_double *_Rin = (cytnx_double *)Rin->Mem;

      if (Lin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[0] * _Rin[i];
        }
      } else if (Rin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[i] * _Rin[0];
        }
      } else {
        if (shape.size() == 0) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (unsigned long long i = 0; i < len; i++) {
            _out[i] = _Lin[i] * _Rin[i];
          }
        } else {
          /// handle non-contiguous:
          std::vector<cytnx_uint64> accu_shape(shape.size());
          std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
          cytnx_uint64 tmp1 = 1, tmp2 = 1, tmp3 = 1;
          for (cytnx_uint64 i = 0; i < shape.size(); i++) {
            accu_shape[shape.size() - 1 - i] = tmp1;
            tmp1 *= shape[shape.size() - 1 - i];

            old_accu_shapeL[shape.size() - 1 - i] = tmp2;
            tmp2 *= shape[invmapper_L[shape.size() - 1 - i]];

            old_accu_shapeR[shape.size() - 1 - i] = tmp3;
            tmp3 *= shape[invmapper_R[shape.size() - 1 - i]];
          }

// handle non-contiguous
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (cytnx_uint64 i = 0; i < len; i++) {
            std::vector<cytnx_uint64> tmpv = c2cartesian(i, accu_shape);
            _out[i] = _Lin[cartesian2c(vec_map(tmpv, invmapper_L), old_accu_shapeL)] *
                      _Rin[cartesian2c(vec_map(tmpv, invmapper_R), old_accu_shapeR)];
          }
        }
      }
    }
    void Mul_internal_cdtf(boost::intrusive_ptr<Storage_base> &out,
                           boost::intrusive_ptr<Storage_base> &Lin,
                           boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                           const std::vector<cytnx_uint64> &shape,
                           const std::vector<cytnx_uint64> &invmapper_L,
                           const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_complex128 *_out = (cytnx_complex128 *)out->Mem;
      cytnx_complex128 *_Lin = (cytnx_complex128 *)Lin->Mem;
      cytnx_float *_Rin = (cytnx_float *)Rin->Mem;

      if (Lin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[0] * _Rin[i];
        }
      } else if (Rin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[i] * _Rin[0];
        }
      } else {
        if (shape.size() == 0) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (unsigned long long i = 0; i < len; i++) {
            _out[i] = _Lin[i] * _Rin[i];
          }
        } else {
          /// handle non-contiguous:
          std::vector<cytnx_uint64> accu_shape(shape.size());
          std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
          cytnx_uint64 tmp1 = 1, tmp2 = 1, tmp3 = 1;
          for (cytnx_uint64 i = 0; i < shape.size(); i++) {
            accu_shape[shape.size() - 1 - i] = tmp1;
            tmp1 *= shape[shape.size() - 1 - i];

            old_accu_shapeL[shape.size() - 1 - i] = tmp2;
            tmp2 *= shape[invmapper_L[shape.size() - 1 - i]];

            old_accu_shapeR[shape.size() - 1 - i] = tmp3;
            tmp3 *= shape[invmapper_R[shape.size() - 1 - i]];
          }

// handle non-contiguous
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (cytnx_uint64 i = 0; i < len; i++) {
            std::vector<cytnx_uint64> tmpv = c2cartesian(i, accu_shape);
            _out[i] = _Lin[cartesian2c(vec_map(tmpv, invmapper_L), old_accu_shapeL)] *
                      _Rin[cartesian2c(vec_map(tmpv, invmapper_R), old_accu_shapeR)];
          }
        }
      }
    }
    void Mul_internal_cdtu64(boost::intrusive_ptr<Storage_base> &out,
                             boost::intrusive_ptr<Storage_base> &Lin,
                             boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                             const std::vector<cytnx_uint64> &shape,
                             const std::vector<cytnx_uint64> &invmapper_L,
                             const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_complex128 *_out = (cytnx_complex128 *)out->Mem;
      cytnx_complex128 *_Lin = (cytnx_complex128 *)Lin->Mem;
      cytnx_uint64 *_Rin = (cytnx_uint64 *)Rin->Mem;

      if (Lin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[0] * _Rin[i];
        }
      } else if (Rin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[i] * _Rin[0];
        }
      } else {
        if (shape.size() == 0) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (unsigned long long i = 0; i < len; i++) {
            _out[i] = _Lin[i] * _Rin[i];
          }
        } else {
          /// handle non-contiguous:
          std::vector<cytnx_uint64> accu_shape(shape.size());
          std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
          cytnx_uint64 tmp1 = 1, tmp2 = 1, tmp3 = 1;
          for (cytnx_uint64 i = 0; i < shape.size(); i++) {
            accu_shape[shape.size() - 1 - i] = tmp1;
            tmp1 *= shape[shape.size() - 1 - i];

            old_accu_shapeL[shape.size() - 1 - i] = tmp2;
            tmp2 *= shape[invmapper_L[shape.size() - 1 - i]];

            old_accu_shapeR[shape.size() - 1 - i] = tmp3;
            tmp3 *= shape[invmapper_R[shape.size() - 1 - i]];
          }

// handle non-contiguous
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (cytnx_uint64 i = 0; i < len; i++) {
            std::vector<cytnx_uint64> tmpv = c2cartesian(i, accu_shape);
            _out[i] = _Lin[cartesian2c(vec_map(tmpv, invmapper_L), old_accu_shapeL)] *
                      _Rin[cartesian2c(vec_map(tmpv, invmapper_R), old_accu_shapeR)];
          }
        }
      }
    }
    void Mul_internal_cdtu32(boost::intrusive_ptr<Storage_base> &out,
                             boost::intrusive_ptr<Storage_base> &Lin,
                             boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                             const std::vector<cytnx_uint64> &shape,
                             const std::vector<cytnx_uint64> &invmapper_L,
                             const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_complex128 *_out = (cytnx_complex128 *)out->Mem;
      cytnx_complex128 *_Lin = (cytnx_complex128 *)Lin->Mem;
      cytnx_uint32 *_Rin = (cytnx_uint32 *)Rin->Mem;

      if (Lin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[0] * _Rin[i];
        }
      } else if (Rin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[i] * _Rin[0];
        }
      } else {
        if (shape.size() == 0) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (unsigned long long i = 0; i < len; i++) {
            _out[i] = _Lin[i] * _Rin[i];
          }
        } else {
          /// handle non-contiguous:
          std::vector<cytnx_uint64> accu_shape(shape.size());
          std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
          cytnx_uint64 tmp1 = 1, tmp2 = 1, tmp3 = 1;
          for (cytnx_uint64 i = 0; i < shape.size(); i++) {
            accu_shape[shape.size() - 1 - i] = tmp1;
            tmp1 *= shape[shape.size() - 1 - i];

            old_accu_shapeL[shape.size() - 1 - i] = tmp2;
            tmp2 *= shape[invmapper_L[shape.size() - 1 - i]];

            old_accu_shapeR[shape.size() - 1 - i] = tmp3;
            tmp3 *= shape[invmapper_R[shape.size() - 1 - i]];
          }

// handle non-contiguous
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (cytnx_uint64 i = 0; i < len; i++) {
            std::vector<cytnx_uint64> tmpv = c2cartesian(i, accu_shape);
            _out[i] = _Lin[cartesian2c(vec_map(tmpv, invmapper_L), old_accu_shapeL)] *
                      _Rin[cartesian2c(vec_map(tmpv, invmapper_R), old_accu_shapeR)];
          }
        }
      }
    }
    void Mul_internal_cdti64(boost::intrusive_ptr<Storage_base> &out,
                             boost::intrusive_ptr<Storage_base> &Lin,
                             boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                             const std::vector<cytnx_uint64> &shape,
                             const std::vector<cytnx_uint64> &invmapper_L,
                             const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_complex128 *_out = (cytnx_complex128 *)out->Mem;
      cytnx_complex128 *_Lin = (cytnx_complex128 *)Lin->Mem;
      cytnx_int64 *_Rin = (cytnx_int64 *)Rin->Mem;

      if (Lin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[0] * _Rin[i];
        }
      } else if (Rin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[i] * _Rin[0];
        }
      } else {
        if (shape.size() == 0) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (unsigned long long i = 0; i < len; i++) {
            _out[i] = _Lin[i] * _Rin[i];
          }
        } else {
          /// handle non-contiguous:
          std::vector<cytnx_uint64> accu_shape(shape.size());
          std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
          cytnx_uint64 tmp1 = 1, tmp2 = 1, tmp3 = 1;
          for (cytnx_uint64 i = 0; i < shape.size(); i++) {
            accu_shape[shape.size() - 1 - i] = tmp1;
            tmp1 *= shape[shape.size() - 1 - i];

            old_accu_shapeL[shape.size() - 1 - i] = tmp2;
            tmp2 *= shape[invmapper_L[shape.size() - 1 - i]];

            old_accu_shapeR[shape.size() - 1 - i] = tmp3;
            tmp3 *= shape[invmapper_R[shape.size() - 1 - i]];
          }

// handle non-contiguous
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (cytnx_uint64 i = 0; i < len; i++) {
            std::vector<cytnx_uint64> tmpv = c2cartesian(i, accu_shape);
            _out[i] = _Lin[cartesian2c(vec_map(tmpv, invmapper_L), old_accu_shapeL)] *
                      _Rin[cartesian2c(vec_map(tmpv, invmapper_R), old_accu_shapeR)];
          }
        }
      }
    }
    void Mul_internal_cdti32(boost::intrusive_ptr<Storage_base> &out,
                             boost::intrusive_ptr<Storage_base> &Lin,
                             boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                             const std::vector<cytnx_uint64> &shape,
                             const std::vector<cytnx_uint64> &invmapper_L,
                             const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_complex128 *_out = (cytnx_complex128 *)out->Mem;
      cytnx_complex128 *_Lin = (cytnx_complex128 *)Lin->Mem;
      cytnx_int32 *_Rin = (cytnx_int32 *)Rin->Mem;

      if (Lin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[0] * _Rin[i];
        }
      } else if (Rin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[i] * _Rin[0];
        }
      } else {
        if (shape.size() == 0) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (unsigned long long i = 0; i < len; i++) {
            _out[i] = _Lin[i] * _Rin[i];
          }
        } else {
          /// handle non-contiguous:
          std::vector<cytnx_uint64> accu_shape(shape.size());
          std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
          cytnx_uint64 tmp1 = 1, tmp2 = 1, tmp3 = 1;
          for (cytnx_uint64 i = 0; i < shape.size(); i++) {
            accu_shape[shape.size() - 1 - i] = tmp1;
            tmp1 *= shape[shape.size() - 1 - i];

            old_accu_shapeL[shape.size() - 1 - i] = tmp2;
            tmp2 *= shape[invmapper_L[shape.size() - 1 - i]];

            old_accu_shapeR[shape.size() - 1 - i] = tmp3;
            tmp3 *= shape[invmapper_R[shape.size() - 1 - i]];
          }

// handle non-contiguous
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (cytnx_uint64 i = 0; i < len; i++) {
            std::vector<cytnx_uint64> tmpv = c2cartesian(i, accu_shape);
            _out[i] = _Lin[cartesian2c(vec_map(tmpv, invmapper_L), old_accu_shapeL)] *
                      _Rin[cartesian2c(vec_map(tmpv, invmapper_R), old_accu_shapeR)];
          }
        }
      }
    }
    void Mul_internal_cdti16(boost::intrusive_ptr<Storage_base> &out,
                             boost::intrusive_ptr<Storage_base> &Lin,
                             boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                             const std::vector<cytnx_uint64> &shape,
                             const std::vector<cytnx_uint64> &invmapper_L,
                             const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_complex128 *_out = (cytnx_complex128 *)out->Mem;
      cytnx_complex128 *_Lin = (cytnx_complex128 *)Lin->Mem;
      cytnx_int16 *_Rin = (cytnx_int16 *)Rin->Mem;

      if (Lin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[0] * cytnx_complex128(_Rin[i], 0);
        }
      } else if (Rin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[i] * cytnx_complex128(_Rin[0], 0);
        }
      } else {
        if (shape.size() == 0) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (unsigned long long i = 0; i < len; i++) {
            _out[i] = _Lin[i] * cytnx_complex128(_Rin[i], 0);
          }
        } else {
          /// handle non-contiguous:
          std::vector<cytnx_uint64> accu_shape(shape.size());
          std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
          cytnx_uint64 tmp1 = 1, tmp2 = 1, tmp3 = 1;
          for (cytnx_uint64 i = 0; i < shape.size(); i++) {
            accu_shape[shape.size() - 1 - i] = tmp1;
            tmp1 *= shape[shape.size() - 1 - i];

            old_accu_shapeL[shape.size() - 1 - i] = tmp2;
            tmp2 *= shape[invmapper_L[shape.size() - 1 - i]];

            old_accu_shapeR[shape.size() - 1 - i] = tmp3;
            tmp3 *= shape[invmapper_R[shape.size() - 1 - i]];
          }

// handle non-contiguous
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (cytnx_uint64 i = 0; i < len; i++) {
            std::vector<cytnx_uint64> tmpv = c2cartesian(i, accu_shape);
            _out[i] =
              _Lin[cartesian2c(vec_map(tmpv, invmapper_L), old_accu_shapeL)] *
              cytnx_complex128(_Rin[cartesian2c(vec_map(tmpv, invmapper_R), old_accu_shapeR)], 0);
          }
        }
      }
    }
    void Mul_internal_cdtu16(boost::intrusive_ptr<Storage_base> &out,
                             boost::intrusive_ptr<Storage_base> &Lin,
                             boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                             const std::vector<cytnx_uint64> &shape,
                             const std::vector<cytnx_uint64> &invmapper_L,
                             const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_complex128 *_out = (cytnx_complex128 *)out->Mem;
      cytnx_complex128 *_Lin = (cytnx_complex128 *)Lin->Mem;
      cytnx_uint16 *_Rin = (cytnx_uint16 *)Rin->Mem;

      if (Lin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[0] * cytnx_complex128(_Rin[i], 0);
        }
      } else if (Rin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[i] * cytnx_complex128(_Rin[0], 0);
        }
      } else {
        if (shape.size() == 0) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (unsigned long long i = 0; i < len; i++) {
            _out[i] = _Lin[i] * cytnx_complex128(_Rin[i], 0);
          }
        } else {
          /// handle non-contiguous:
          std::vector<cytnx_uint64> accu_shape(shape.size());
          std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
          cytnx_uint64 tmp1 = 1, tmp2 = 1, tmp3 = 1;
          for (cytnx_uint64 i = 0; i < shape.size(); i++) {
            accu_shape[shape.size() - 1 - i] = tmp1;
            tmp1 *= shape[shape.size() - 1 - i];

            old_accu_shapeL[shape.size() - 1 - i] = tmp2;
            tmp2 *= shape[invmapper_L[shape.size() - 1 - i]];

            old_accu_shapeR[shape.size() - 1 - i] = tmp3;
            tmp3 *= shape[invmapper_R[shape.size() - 1 - i]];
          }

// handle non-contiguous
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (cytnx_uint64 i = 0; i < len; i++) {
            std::vector<cytnx_uint64> tmpv = c2cartesian(i, accu_shape);
            _out[i] =
              _Lin[cartesian2c(vec_map(tmpv, invmapper_L), old_accu_shapeL)] *
              cytnx_complex128(_Rin[cartesian2c(vec_map(tmpv, invmapper_R), old_accu_shapeR)], 0);
          }
        }
      }
    }
    void Mul_internal_cdtb(boost::intrusive_ptr<Storage_base> &out,
                           boost::intrusive_ptr<Storage_base> &Lin,
                           boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                           const std::vector<cytnx_uint64> &shape,
                           const std::vector<cytnx_uint64> &invmapper_L,
                           const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_complex128 *_out = (cytnx_complex128 *)out->Mem;
      cytnx_complex128 *_Lin = (cytnx_complex128 *)Lin->Mem;
      cytnx_bool *_Rin = (cytnx_bool *)Rin->Mem;

      if (Lin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[0] * cytnx_complex128(_Rin[i], 0);
        }
      } else if (Rin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[i] * cytnx_complex128(_Rin[0], 0);
        }
      } else {
        if (shape.size() == 0) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (unsigned long long i = 0; i < len; i++) {
            _out[i] = _Lin[i] * cytnx_complex128(_Rin[i], 0);
          }
        } else {
          /// handle non-contiguous:
          std::vector<cytnx_uint64> accu_shape(shape.size());
          std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
          cytnx_uint64 tmp1 = 1, tmp2 = 1, tmp3 = 1;
          for (cytnx_uint64 i = 0; i < shape.size(); i++) {
            accu_shape[shape.size() - 1 - i] = tmp1;
            tmp1 *= shape[shape.size() - 1 - i];

            old_accu_shapeL[shape.size() - 1 - i] = tmp2;
            tmp2 *= shape[invmapper_L[shape.size() - 1 - i]];

            old_accu_shapeR[shape.size() - 1 - i] = tmp3;
            tmp3 *= shape[invmapper_R[shape.size() - 1 - i]];
          }

// handle non-contiguous
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (cytnx_uint64 i = 0; i < len; i++) {
            std::vector<cytnx_uint64> tmpv = c2cartesian(i, accu_shape);
            _out[i] =
              _Lin[cartesian2c(vec_map(tmpv, invmapper_L), old_accu_shapeL)] *
              cytnx_complex128(_Rin[cartesian2c(vec_map(tmpv, invmapper_R), old_accu_shapeR)], 0);
          }
        }
      }
    }
    //--------------------------
    void Mul_internal_cftcd(boost::intrusive_ptr<Storage_base> &out,
                            boost::intrusive_ptr<Storage_base> &Lin,
                            boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                            const std::vector<cytnx_uint64> &shape,
                            const std::vector<cytnx_uint64> &invmapper_L,
                            const std::vector<cytnx_uint64> &invmapper_R) {
      Mul_internal_cdtcf(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void Mul_internal_cftcf(boost::intrusive_ptr<Storage_base> &out,
                            boost::intrusive_ptr<Storage_base> &Lin,
                            boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                            const std::vector<cytnx_uint64> &shape,
                            const std::vector<cytnx_uint64> &invmapper_L,
                            const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_complex64 *_out = (cytnx_complex64 *)out->Mem;
      cytnx_complex64 *_Lin = (cytnx_complex64 *)Lin->Mem;
      cytnx_complex64 *_Rin = (cytnx_complex64 *)Rin->Mem;

      if (Lin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[0] * _Rin[i];
        }
      } else if (Rin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[i] * _Rin[0];
        }
      } else {
        if (shape.size() == 0) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (unsigned long long i = 0; i < len; i++) {
            _out[i] = _Lin[i] * _Rin[i];
          }
        } else {
          /// handle non-contiguous:
          std::vector<cytnx_uint64> accu_shape(shape.size());
          std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
          cytnx_uint64 tmp1 = 1, tmp2 = 1, tmp3 = 1;
          for (cytnx_uint64 i = 0; i < shape.size(); i++) {
            accu_shape[shape.size() - 1 - i] = tmp1;
            tmp1 *= shape[shape.size() - 1 - i];

            old_accu_shapeL[shape.size() - 1 - i] = tmp2;
            tmp2 *= shape[invmapper_L[shape.size() - 1 - i]];

            old_accu_shapeR[shape.size() - 1 - i] = tmp3;
            tmp3 *= shape[invmapper_R[shape.size() - 1 - i]];
          }

// handle non-contiguous
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (cytnx_uint64 i = 0; i < len; i++) {
            std::vector<cytnx_uint64> tmpv = c2cartesian(i, accu_shape);
            _out[i] = _Lin[cartesian2c(vec_map(tmpv, invmapper_L), old_accu_shapeL)] *
                      _Rin[cartesian2c(vec_map(tmpv, invmapper_R), old_accu_shapeR)];
          }
        }
      }
    }
    void Mul_internal_cftd(boost::intrusive_ptr<Storage_base> &out,
                           boost::intrusive_ptr<Storage_base> &Lin,
                           boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                           const std::vector<cytnx_uint64> &shape,
                           const std::vector<cytnx_uint64> &invmapper_L,
                           const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_complex64 *_out = (cytnx_complex64 *)out->Mem;
      cytnx_complex64 *_Lin = (cytnx_complex64 *)Lin->Mem;
      cytnx_double *_Rin = (cytnx_double *)Rin->Mem;

      if (Lin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[0] * _Rin[i];
        }
      } else if (Rin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[i] * _Rin[0];
        }
      } else {
        if (shape.size() == 0) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (unsigned long long i = 0; i < len; i++) {
            _out[i] = _Lin[i] * _Rin[i];
          }
        } else {
          /// handle non-contiguous:
          std::vector<cytnx_uint64> accu_shape(shape.size());
          std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
          cytnx_uint64 tmp1 = 1, tmp2 = 1, tmp3 = 1;
          for (cytnx_uint64 i = 0; i < shape.size(); i++) {
            accu_shape[shape.size() - 1 - i] = tmp1;
            tmp1 *= shape[shape.size() - 1 - i];

            old_accu_shapeL[shape.size() - 1 - i] = tmp2;
            tmp2 *= shape[invmapper_L[shape.size() - 1 - i]];

            old_accu_shapeR[shape.size() - 1 - i] = tmp3;
            tmp3 *= shape[invmapper_R[shape.size() - 1 - i]];
          }

// handle non-contiguous
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (cytnx_uint64 i = 0; i < len; i++) {
            std::vector<cytnx_uint64> tmpv = c2cartesian(i, accu_shape);
            _out[i] = _Lin[cartesian2c(vec_map(tmpv, invmapper_L), old_accu_shapeL)] *
                      _Rin[cartesian2c(vec_map(tmpv, invmapper_R), old_accu_shapeR)];
          }
        }
      }
    }
    void Mul_internal_cftf(boost::intrusive_ptr<Storage_base> &out,
                           boost::intrusive_ptr<Storage_base> &Lin,
                           boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                           const std::vector<cytnx_uint64> &shape,
                           const std::vector<cytnx_uint64> &invmapper_L,
                           const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_complex64 *_out = (cytnx_complex64 *)out->Mem;
      cytnx_complex64 *_Lin = (cytnx_complex64 *)Lin->Mem;
      cytnx_float *_Rin = (cytnx_float *)Rin->Mem;

      if (Lin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[0] * _Rin[i];
        }
      } else if (Rin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[i] * _Rin[0];
        }
      } else {
        if (shape.size() == 0) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (unsigned long long i = 0; i < len; i++) {
            _out[i] = _Lin[i] * _Rin[i];
          }
        } else {
          /// handle non-contiguous:
          std::vector<cytnx_uint64> accu_shape(shape.size());
          std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
          cytnx_uint64 tmp1 = 1, tmp2 = 1, tmp3 = 1;
          for (cytnx_uint64 i = 0; i < shape.size(); i++) {
            accu_shape[shape.size() - 1 - i] = tmp1;
            tmp1 *= shape[shape.size() - 1 - i];

            old_accu_shapeL[shape.size() - 1 - i] = tmp2;
            tmp2 *= shape[invmapper_L[shape.size() - 1 - i]];

            old_accu_shapeR[shape.size() - 1 - i] = tmp3;
            tmp3 *= shape[invmapper_R[shape.size() - 1 - i]];
          }

// handle non-contiguous
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (cytnx_uint64 i = 0; i < len; i++) {
            std::vector<cytnx_uint64> tmpv = c2cartesian(i, accu_shape);
            _out[i] = _Lin[cartesian2c(vec_map(tmpv, invmapper_L), old_accu_shapeL)] *
                      _Rin[cartesian2c(vec_map(tmpv, invmapper_R), old_accu_shapeR)];
          }
        }
      }
    }
    void Mul_internal_cftu64(boost::intrusive_ptr<Storage_base> &out,
                             boost::intrusive_ptr<Storage_base> &Lin,
                             boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                             const std::vector<cytnx_uint64> &shape,
                             const std::vector<cytnx_uint64> &invmapper_L,
                             const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_complex64 *_out = (cytnx_complex64 *)out->Mem;
      cytnx_complex64 *_Lin = (cytnx_complex64 *)Lin->Mem;
      cytnx_uint64 *_Rin = (cytnx_uint64 *)Rin->Mem;

      if (Lin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[0] * _Rin[i];
        }
      } else if (Rin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[i] * _Rin[0];
        }
      } else {
        if (shape.size() == 0) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (unsigned long long i = 0; i < len; i++) {
            _out[i] = _Lin[i] * _Rin[i];
          }
        } else {
          /// handle non-contiguous:
          std::vector<cytnx_uint64> accu_shape(shape.size());
          std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
          cytnx_uint64 tmp1 = 1, tmp2 = 1, tmp3 = 1;
          for (cytnx_uint64 i = 0; i < shape.size(); i++) {
            accu_shape[shape.size() - 1 - i] = tmp1;
            tmp1 *= shape[shape.size() - 1 - i];

            old_accu_shapeL[shape.size() - 1 - i] = tmp2;
            tmp2 *= shape[invmapper_L[shape.size() - 1 - i]];

            old_accu_shapeR[shape.size() - 1 - i] = tmp3;
            tmp3 *= shape[invmapper_R[shape.size() - 1 - i]];
          }

// handle non-contiguous
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (cytnx_uint64 i = 0; i < len; i++) {
            std::vector<cytnx_uint64> tmpv = c2cartesian(i, accu_shape);
            _out[i] = _Lin[cartesian2c(vec_map(tmpv, invmapper_L), old_accu_shapeL)] *
                      _Rin[cartesian2c(vec_map(tmpv, invmapper_R), old_accu_shapeR)];
          }
        }
      }
    }
    void Mul_internal_cftu32(boost::intrusive_ptr<Storage_base> &out,
                             boost::intrusive_ptr<Storage_base> &Lin,
                             boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                             const std::vector<cytnx_uint64> &shape,
                             const std::vector<cytnx_uint64> &invmapper_L,
                             const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_complex64 *_out = (cytnx_complex64 *)out->Mem;
      cytnx_complex64 *_Lin = (cytnx_complex64 *)Lin->Mem;
      cytnx_uint32 *_Rin = (cytnx_uint32 *)Rin->Mem;

      if (Lin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[0] * _Rin[i];
        }
      } else if (Rin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[i] * _Rin[0];
        }
      } else {
        if (shape.size() == 0) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (unsigned long long i = 0; i < len; i++) {
            _out[i] = _Lin[i] * _Rin[i];
          }
        } else {
          /// handle non-contiguous:
          std::vector<cytnx_uint64> accu_shape(shape.size());
          std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
          cytnx_uint64 tmp1 = 1, tmp2 = 1, tmp3 = 1;
          for (cytnx_uint64 i = 0; i < shape.size(); i++) {
            accu_shape[shape.size() - 1 - i] = tmp1;
            tmp1 *= shape[shape.size() - 1 - i];

            old_accu_shapeL[shape.size() - 1 - i] = tmp2;
            tmp2 *= shape[invmapper_L[shape.size() - 1 - i]];

            old_accu_shapeR[shape.size() - 1 - i] = tmp3;
            tmp3 *= shape[invmapper_R[shape.size() - 1 - i]];
          }

// handle non-contiguous
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (cytnx_uint64 i = 0; i < len; i++) {
            std::vector<cytnx_uint64> tmpv = c2cartesian(i, accu_shape);
            _out[i] = _Lin[cartesian2c(vec_map(tmpv, invmapper_L), old_accu_shapeL)] *
                      _Rin[cartesian2c(vec_map(tmpv, invmapper_R), old_accu_shapeR)];
          }
        }
      }
    }
    void Mul_internal_cfti64(boost::intrusive_ptr<Storage_base> &out,
                             boost::intrusive_ptr<Storage_base> &Lin,
                             boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                             const std::vector<cytnx_uint64> &shape,
                             const std::vector<cytnx_uint64> &invmapper_L,
                             const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_complex64 *_out = (cytnx_complex64 *)out->Mem;
      cytnx_complex64 *_Lin = (cytnx_complex64 *)Lin->Mem;
      cytnx_int64 *_Rin = (cytnx_int64 *)Rin->Mem;

      if (Lin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[0] * _Rin[i];
        }
      } else if (Rin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[i] * _Rin[0];
        }
      } else {
        if (shape.size() == 0) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (unsigned long long i = 0; i < len; i++) {
            _out[i] = _Lin[i] * _Rin[i];
          }
        } else {
          /// handle non-contiguous:
          std::vector<cytnx_uint64> accu_shape(shape.size());
          std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
          cytnx_uint64 tmp1 = 1, tmp2 = 1, tmp3 = 1;
          for (cytnx_uint64 i = 0; i < shape.size(); i++) {
            accu_shape[shape.size() - 1 - i] = tmp1;
            tmp1 *= shape[shape.size() - 1 - i];

            old_accu_shapeL[shape.size() - 1 - i] = tmp2;
            tmp2 *= shape[invmapper_L[shape.size() - 1 - i]];

            old_accu_shapeR[shape.size() - 1 - i] = tmp3;
            tmp3 *= shape[invmapper_R[shape.size() - 1 - i]];
          }

// handle non-contiguous
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (cytnx_uint64 i = 0; i < len; i++) {
            std::vector<cytnx_uint64> tmpv = c2cartesian(i, accu_shape);
            _out[i] = _Lin[cartesian2c(vec_map(tmpv, invmapper_L), old_accu_shapeL)] *
                      _Rin[cartesian2c(vec_map(tmpv, invmapper_R), old_accu_shapeR)];
          }
        }
      }
    }
    void Mul_internal_cfti32(boost::intrusive_ptr<Storage_base> &out,
                             boost::intrusive_ptr<Storage_base> &Lin,
                             boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                             const std::vector<cytnx_uint64> &shape,
                             const std::vector<cytnx_uint64> &invmapper_L,
                             const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_complex64 *_out = (cytnx_complex64 *)out->Mem;
      cytnx_complex64 *_Lin = (cytnx_complex64 *)Lin->Mem;
      cytnx_int32 *_Rin = (cytnx_int32 *)Rin->Mem;

      if (Lin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[0] * _Rin[i];
        }
      } else if (Rin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[i] * _Rin[0];
        }
      } else {
        if (shape.size() == 0) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (unsigned long long i = 0; i < len; i++) {
            _out[i] = _Lin[i] * _Rin[i];
          }
        } else {
          /// handle non-contiguous:
          std::vector<cytnx_uint64> accu_shape(shape.size());
          std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
          cytnx_uint64 tmp1 = 1, tmp2 = 1, tmp3 = 1;
          for (cytnx_uint64 i = 0; i < shape.size(); i++) {
            accu_shape[shape.size() - 1 - i] = tmp1;
            tmp1 *= shape[shape.size() - 1 - i];

            old_accu_shapeL[shape.size() - 1 - i] = tmp2;
            tmp2 *= shape[invmapper_L[shape.size() - 1 - i]];

            old_accu_shapeR[shape.size() - 1 - i] = tmp3;
            tmp3 *= shape[invmapper_R[shape.size() - 1 - i]];
          }

// handle non-contiguous
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (cytnx_uint64 i = 0; i < len; i++) {
            std::vector<cytnx_uint64> tmpv = c2cartesian(i, accu_shape);
            _out[i] = _Lin[cartesian2c(vec_map(tmpv, invmapper_L), old_accu_shapeL)] *
                      _Rin[cartesian2c(vec_map(tmpv, invmapper_R), old_accu_shapeR)];
          }
        }
      }
    }
    void Mul_internal_cfti16(boost::intrusive_ptr<Storage_base> &out,
                             boost::intrusive_ptr<Storage_base> &Lin,
                             boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                             const std::vector<cytnx_uint64> &shape,
                             const std::vector<cytnx_uint64> &invmapper_L,
                             const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_complex64 *_out = (cytnx_complex64 *)out->Mem;
      cytnx_complex64 *_Lin = (cytnx_complex64 *)Lin->Mem;
      cytnx_int16 *_Rin = (cytnx_int16 *)Rin->Mem;

      if (Lin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[0] * _Rin[i];
        }
      } else if (Rin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[i] * _Rin[0];
        }
      } else {
        if (shape.size() == 0) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (unsigned long long i = 0; i < len; i++) {
            _out[i] = _Lin[i] * _Rin[i];
          }
        } else {
          /// handle non-contiguous:
          std::vector<cytnx_uint64> accu_shape(shape.size());
          std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
          cytnx_uint64 tmp1 = 1, tmp2 = 1, tmp3 = 1;
          for (cytnx_uint64 i = 0; i < shape.size(); i++) {
            accu_shape[shape.size() - 1 - i] = tmp1;
            tmp1 *= shape[shape.size() - 1 - i];

            old_accu_shapeL[shape.size() - 1 - i] = tmp2;
            tmp2 *= shape[invmapper_L[shape.size() - 1 - i]];

            old_accu_shapeR[shape.size() - 1 - i] = tmp3;
            tmp3 *= shape[invmapper_R[shape.size() - 1 - i]];
          }

// handle non-contiguous
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (cytnx_uint64 i = 0; i < len; i++) {
            std::vector<cytnx_uint64> tmpv = c2cartesian(i, accu_shape);
            _out[i] = _Lin[cartesian2c(vec_map(tmpv, invmapper_L), old_accu_shapeL)] *
                      _Rin[cartesian2c(vec_map(tmpv, invmapper_R), old_accu_shapeR)];
          }
        }
      }
    }
    void Mul_internal_cftu16(boost::intrusive_ptr<Storage_base> &out,
                             boost::intrusive_ptr<Storage_base> &Lin,
                             boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                             const std::vector<cytnx_uint64> &shape,
                             const std::vector<cytnx_uint64> &invmapper_L,
                             const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_complex64 *_out = (cytnx_complex64 *)out->Mem;
      cytnx_complex64 *_Lin = (cytnx_complex64 *)Lin->Mem;
      cytnx_uint16 *_Rin = (cytnx_uint16 *)Rin->Mem;

      if (Lin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[0] * _Rin[i];
        }
      } else if (Rin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[i] * _Rin[0];
        }
      } else {
        if (shape.size() == 0) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (unsigned long long i = 0; i < len; i++) {
            _out[i] = _Lin[i] * _Rin[i];
          }
        } else {
          /// handle non-contiguous:
          std::vector<cytnx_uint64> accu_shape(shape.size());
          std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
          cytnx_uint64 tmp1 = 1, tmp2 = 1, tmp3 = 1;
          for (cytnx_uint64 i = 0; i < shape.size(); i++) {
            accu_shape[shape.size() - 1 - i] = tmp1;
            tmp1 *= shape[shape.size() - 1 - i];

            old_accu_shapeL[shape.size() - 1 - i] = tmp2;
            tmp2 *= shape[invmapper_L[shape.size() - 1 - i]];

            old_accu_shapeR[shape.size() - 1 - i] = tmp3;
            tmp3 *= shape[invmapper_R[shape.size() - 1 - i]];
          }

// handle non-contiguous
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (cytnx_uint64 i = 0; i < len; i++) {
            std::vector<cytnx_uint64> tmpv = c2cartesian(i, accu_shape);
            _out[i] = _Lin[cartesian2c(vec_map(tmpv, invmapper_L), old_accu_shapeL)] *
                      _Rin[cartesian2c(vec_map(tmpv, invmapper_R), old_accu_shapeR)];
          }
        }
      }
    }
    void Mul_internal_cftb(boost::intrusive_ptr<Storage_base> &out,
                           boost::intrusive_ptr<Storage_base> &Lin,
                           boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                           const std::vector<cytnx_uint64> &shape,
                           const std::vector<cytnx_uint64> &invmapper_L,
                           const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_complex64 *_out = (cytnx_complex64 *)out->Mem;
      cytnx_complex64 *_Lin = (cytnx_complex64 *)Lin->Mem;
      cytnx_bool *_Rin = (cytnx_bool *)Rin->Mem;

      if (Lin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[0] * cytnx_complex64(_Rin[i], 0);
        }
      } else if (Rin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[i] * cytnx_complex64(_Rin[0], 0);
        }
      } else {
        if (shape.size() == 0) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (unsigned long long i = 0; i < len; i++) {
            _out[i] = _Lin[i] * cytnx_complex64(_Rin[i], 0);
          }
        } else {
          /// handle non-contiguous:
          std::vector<cytnx_uint64> accu_shape(shape.size());
          std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
          cytnx_uint64 tmp1 = 1, tmp2 = 1, tmp3 = 1;
          for (cytnx_uint64 i = 0; i < shape.size(); i++) {
            accu_shape[shape.size() - 1 - i] = tmp1;
            tmp1 *= shape[shape.size() - 1 - i];

            old_accu_shapeL[shape.size() - 1 - i] = tmp2;
            tmp2 *= shape[invmapper_L[shape.size() - 1 - i]];

            old_accu_shapeR[shape.size() - 1 - i] = tmp3;
            tmp3 *= shape[invmapper_R[shape.size() - 1 - i]];
          }

// handle non-contiguous
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (cytnx_uint64 i = 0; i < len; i++) {
            std::vector<cytnx_uint64> tmpv = c2cartesian(i, accu_shape);
            _out[i] =
              _Lin[cartesian2c(vec_map(tmpv, invmapper_L), old_accu_shapeL)] *
              cytnx_complex64(_Rin[cartesian2c(vec_map(tmpv, invmapper_R), old_accu_shapeR)], 0);
          }
        }
      }
    }
    //------------------------------
    void Mul_internal_dtcd(boost::intrusive_ptr<Storage_base> &out,
                           boost::intrusive_ptr<Storage_base> &Lin,
                           boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                           const std::vector<cytnx_uint64> &shape,
                           const std::vector<cytnx_uint64> &invmapper_L,
                           const std::vector<cytnx_uint64> &invmapper_R) {
      Mul_internal_cdtd(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void Mul_internal_dtcf(boost::intrusive_ptr<Storage_base> &out,
                           boost::intrusive_ptr<Storage_base> &Lin,
                           boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                           const std::vector<cytnx_uint64> &shape,
                           const std::vector<cytnx_uint64> &invmapper_L,
                           const std::vector<cytnx_uint64> &invmapper_R) {
      Mul_internal_cftd(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void Mul_internal_dtd(boost::intrusive_ptr<Storage_base> &out,
                          boost::intrusive_ptr<Storage_base> &Lin,
                          boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                          const std::vector<cytnx_uint64> &shape,
                          const std::vector<cytnx_uint64> &invmapper_L,
                          const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_double *_out = (cytnx_double *)out->Mem;
      cytnx_double *_Lin = (cytnx_double *)Lin->Mem;
      cytnx_double *_Rin = (cytnx_double *)Rin->Mem;

      if (Lin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[0] * _Rin[i];
        }
      } else if (Rin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[i] * _Rin[0];
        }
      } else {
        if (shape.size() == 0) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (unsigned long long i = 0; i < len; i++) {
            _out[i] = _Lin[i] * _Rin[i];
          }
        } else {
          /// handle non-contiguous:
          std::vector<cytnx_uint64> accu_shape(shape.size());
          std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
          cytnx_uint64 tmp1 = 1, tmp2 = 1, tmp3 = 1;
          for (cytnx_uint64 i = 0; i < shape.size(); i++) {
            accu_shape[shape.size() - 1 - i] = tmp1;
            tmp1 *= shape[shape.size() - 1 - i];

            old_accu_shapeL[shape.size() - 1 - i] = tmp2;
            tmp2 *= shape[invmapper_L[shape.size() - 1 - i]];

            old_accu_shapeR[shape.size() - 1 - i] = tmp3;
            tmp3 *= shape[invmapper_R[shape.size() - 1 - i]];
          }

// handle non-contiguous
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (cytnx_uint64 i = 0; i < len; i++) {
            std::vector<cytnx_uint64> tmpv = c2cartesian(i, accu_shape);
            _out[i] = _Lin[cartesian2c(vec_map(tmpv, invmapper_L), old_accu_shapeL)] *
                      _Rin[cartesian2c(vec_map(tmpv, invmapper_R), old_accu_shapeR)];
          }
        }
      }
    }
    void Mul_internal_dtf(boost::intrusive_ptr<Storage_base> &out,
                          boost::intrusive_ptr<Storage_base> &Lin,
                          boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                          const std::vector<cytnx_uint64> &shape,
                          const std::vector<cytnx_uint64> &invmapper_L,
                          const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_double *_out = (cytnx_double *)out->Mem;
      cytnx_double *_Lin = (cytnx_double *)Lin->Mem;
      cytnx_float *_Rin = (cytnx_float *)Rin->Mem;

      if (Lin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[0] * _Rin[i];
        }
      } else if (Rin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[i] * _Rin[0];
        }
      } else {
        if (shape.size() == 0) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (unsigned long long i = 0; i < len; i++) {
            _out[i] = _Lin[i] * _Rin[i];
          }
        } else {
          /// handle non-contiguous:
          std::vector<cytnx_uint64> accu_shape(shape.size());
          std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
          cytnx_uint64 tmp1 = 1, tmp2 = 1, tmp3 = 1;
          for (cytnx_uint64 i = 0; i < shape.size(); i++) {
            accu_shape[shape.size() - 1 - i] = tmp1;
            tmp1 *= shape[shape.size() - 1 - i];

            old_accu_shapeL[shape.size() - 1 - i] = tmp2;
            tmp2 *= shape[invmapper_L[shape.size() - 1 - i]];

            old_accu_shapeR[shape.size() - 1 - i] = tmp3;
            tmp3 *= shape[invmapper_R[shape.size() - 1 - i]];
          }

// handle non-contiguous
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (cytnx_uint64 i = 0; i < len; i++) {
            std::vector<cytnx_uint64> tmpv = c2cartesian(i, accu_shape);
            _out[i] = _Lin[cartesian2c(vec_map(tmpv, invmapper_L), old_accu_shapeL)] *
                      _Rin[cartesian2c(vec_map(tmpv, invmapper_R), old_accu_shapeR)];
          }
        }
      }
    }
    void Mul_internal_dtu64(boost::intrusive_ptr<Storage_base> &out,
                            boost::intrusive_ptr<Storage_base> &Lin,
                            boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                            const std::vector<cytnx_uint64> &shape,
                            const std::vector<cytnx_uint64> &invmapper_L,
                            const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_double *_out = (cytnx_double *)out->Mem;
      cytnx_double *_Lin = (cytnx_double *)Lin->Mem;
      cytnx_uint64 *_Rin = (cytnx_uint64 *)Rin->Mem;

      if (Lin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[0] * _Rin[i];
        }
      } else if (Rin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[i] * _Rin[0];
        }
      } else {
        if (shape.size() == 0) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (unsigned long long i = 0; i < len; i++) {
            _out[i] = _Lin[i] * _Rin[i];
          }
        } else {
          /// handle non-contiguous:
          std::vector<cytnx_uint64> accu_shape(shape.size());
          std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
          cytnx_uint64 tmp1 = 1, tmp2 = 1, tmp3 = 1;
          for (cytnx_uint64 i = 0; i < shape.size(); i++) {
            accu_shape[shape.size() - 1 - i] = tmp1;
            tmp1 *= shape[shape.size() - 1 - i];

            old_accu_shapeL[shape.size() - 1 - i] = tmp2;
            tmp2 *= shape[invmapper_L[shape.size() - 1 - i]];

            old_accu_shapeR[shape.size() - 1 - i] = tmp3;
            tmp3 *= shape[invmapper_R[shape.size() - 1 - i]];
          }

// handle non-contiguous
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (cytnx_uint64 i = 0; i < len; i++) {
            std::vector<cytnx_uint64> tmpv = c2cartesian(i, accu_shape);
            _out[i] = _Lin[cartesian2c(vec_map(tmpv, invmapper_L), old_accu_shapeL)] *
                      _Rin[cartesian2c(vec_map(tmpv, invmapper_R), old_accu_shapeR)];
          }
        }
      }
    }
    void Mul_internal_dtu32(boost::intrusive_ptr<Storage_base> &out,
                            boost::intrusive_ptr<Storage_base> &Lin,
                            boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                            const std::vector<cytnx_uint64> &shape,
                            const std::vector<cytnx_uint64> &invmapper_L,
                            const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_double *_out = (cytnx_double *)out->Mem;
      cytnx_double *_Lin = (cytnx_double *)Lin->Mem;
      cytnx_uint32 *_Rin = (cytnx_uint32 *)Rin->Mem;

      if (Lin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[0] * _Rin[i];
        }
      } else if (Rin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[i] * _Rin[0];
        }
      } else {
        if (shape.size() == 0) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (unsigned long long i = 0; i < len; i++) {
            _out[i] = _Lin[i] * _Rin[i];
          }
        } else {
          /// handle non-contiguous:
          std::vector<cytnx_uint64> accu_shape(shape.size());
          std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
          cytnx_uint64 tmp1 = 1, tmp2 = 1, tmp3 = 1;
          for (cytnx_uint64 i = 0; i < shape.size(); i++) {
            accu_shape[shape.size() - 1 - i] = tmp1;
            tmp1 *= shape[shape.size() - 1 - i];

            old_accu_shapeL[shape.size() - 1 - i] = tmp2;
            tmp2 *= shape[invmapper_L[shape.size() - 1 - i]];

            old_accu_shapeR[shape.size() - 1 - i] = tmp3;
            tmp3 *= shape[invmapper_R[shape.size() - 1 - i]];
          }

// handle non-contiguous
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (cytnx_uint64 i = 0; i < len; i++) {
            std::vector<cytnx_uint64> tmpv = c2cartesian(i, accu_shape);
            _out[i] = _Lin[cartesian2c(vec_map(tmpv, invmapper_L), old_accu_shapeL)] *
                      _Rin[cartesian2c(vec_map(tmpv, invmapper_R), old_accu_shapeR)];
          }
        }
      }
    }
    void Mul_internal_dti64(boost::intrusive_ptr<Storage_base> &out,
                            boost::intrusive_ptr<Storage_base> &Lin,
                            boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                            const std::vector<cytnx_uint64> &shape,
                            const std::vector<cytnx_uint64> &invmapper_L,
                            const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_double *_out = (cytnx_double *)out->Mem;
      cytnx_double *_Lin = (cytnx_double *)Lin->Mem;
      cytnx_int64 *_Rin = (cytnx_int64 *)Rin->Mem;

      if (Lin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[0] * _Rin[i];
        }
      } else if (Rin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[i] * _Rin[0];
        }
      } else {
        if (shape.size() == 0) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (unsigned long long i = 0; i < len; i++) {
            _out[i] = _Lin[i] * _Rin[i];
          }
        } else {
          /// handle non-contiguous:
          std::vector<cytnx_uint64> accu_shape(shape.size());
          std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
          cytnx_uint64 tmp1 = 1, tmp2 = 1, tmp3 = 1;
          for (cytnx_uint64 i = 0; i < shape.size(); i++) {
            accu_shape[shape.size() - 1 - i] = tmp1;
            tmp1 *= shape[shape.size() - 1 - i];

            old_accu_shapeL[shape.size() - 1 - i] = tmp2;
            tmp2 *= shape[invmapper_L[shape.size() - 1 - i]];

            old_accu_shapeR[shape.size() - 1 - i] = tmp3;
            tmp3 *= shape[invmapper_R[shape.size() - 1 - i]];
          }

// handle non-contiguous
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (cytnx_uint64 i = 0; i < len; i++) {
            std::vector<cytnx_uint64> tmpv = c2cartesian(i, accu_shape);
            _out[i] = _Lin[cartesian2c(vec_map(tmpv, invmapper_L), old_accu_shapeL)] *
                      _Rin[cartesian2c(vec_map(tmpv, invmapper_R), old_accu_shapeR)];
          }
        }
      }
    }
    void Mul_internal_dti32(boost::intrusive_ptr<Storage_base> &out,
                            boost::intrusive_ptr<Storage_base> &Lin,
                            boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                            const std::vector<cytnx_uint64> &shape,
                            const std::vector<cytnx_uint64> &invmapper_L,
                            const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_double *_out = (cytnx_double *)out->Mem;
      cytnx_double *_Lin = (cytnx_double *)Lin->Mem;
      cytnx_int32 *_Rin = (cytnx_int32 *)Rin->Mem;

      if (Lin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[0] * _Rin[i];
        }
      } else if (Rin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[i] * _Rin[0];
        }
      } else {
        if (shape.size() == 0) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (unsigned long long i = 0; i < len; i++) {
            _out[i] = _Lin[i] * _Rin[i];
          }
        } else {
          /// handle non-contiguous:
          std::vector<cytnx_uint64> accu_shape(shape.size());
          std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
          cytnx_uint64 tmp1 = 1, tmp2 = 1, tmp3 = 1;
          for (cytnx_uint64 i = 0; i < shape.size(); i++) {
            accu_shape[shape.size() - 1 - i] = tmp1;
            tmp1 *= shape[shape.size() - 1 - i];

            old_accu_shapeL[shape.size() - 1 - i] = tmp2;
            tmp2 *= shape[invmapper_L[shape.size() - 1 - i]];

            old_accu_shapeR[shape.size() - 1 - i] = tmp3;
            tmp3 *= shape[invmapper_R[shape.size() - 1 - i]];
          }

// handle non-contiguous
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (cytnx_uint64 i = 0; i < len; i++) {
            std::vector<cytnx_uint64> tmpv = c2cartesian(i, accu_shape);
            _out[i] = _Lin[cartesian2c(vec_map(tmpv, invmapper_L), old_accu_shapeL)] *
                      _Rin[cartesian2c(vec_map(tmpv, invmapper_R), old_accu_shapeR)];
          }
        }
      }
    }
    void Mul_internal_dti16(boost::intrusive_ptr<Storage_base> &out,
                            boost::intrusive_ptr<Storage_base> &Lin,
                            boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                            const std::vector<cytnx_uint64> &shape,
                            const std::vector<cytnx_uint64> &invmapper_L,
                            const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_double *_out = (cytnx_double *)out->Mem;
      cytnx_double *_Lin = (cytnx_double *)Lin->Mem;
      cytnx_int16 *_Rin = (cytnx_int16 *)Rin->Mem;

      if (Lin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[0] * _Rin[i];
        }
      } else if (Rin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[i] * _Rin[0];
        }
      } else {
        if (shape.size() == 0) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (unsigned long long i = 0; i < len; i++) {
            _out[i] = _Lin[i] * _Rin[i];
          }
        } else {
          /// handle non-contiguous:
          std::vector<cytnx_uint64> accu_shape(shape.size());
          std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
          cytnx_uint64 tmp1 = 1, tmp2 = 1, tmp3 = 1;
          for (cytnx_uint64 i = 0; i < shape.size(); i++) {
            accu_shape[shape.size() - 1 - i] = tmp1;
            tmp1 *= shape[shape.size() - 1 - i];

            old_accu_shapeL[shape.size() - 1 - i] = tmp2;
            tmp2 *= shape[invmapper_L[shape.size() - 1 - i]];

            old_accu_shapeR[shape.size() - 1 - i] = tmp3;
            tmp3 *= shape[invmapper_R[shape.size() - 1 - i]];
          }

// handle non-contiguous
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (cytnx_uint64 i = 0; i < len; i++) {
            std::vector<cytnx_uint64> tmpv = c2cartesian(i, accu_shape);
            _out[i] = _Lin[cartesian2c(vec_map(tmpv, invmapper_L), old_accu_shapeL)] *
                      _Rin[cartesian2c(vec_map(tmpv, invmapper_R), old_accu_shapeR)];
          }
        }
      }
    }
    void Mul_internal_dtu16(boost::intrusive_ptr<Storage_base> &out,
                            boost::intrusive_ptr<Storage_base> &Lin,
                            boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                            const std::vector<cytnx_uint64> &shape,
                            const std::vector<cytnx_uint64> &invmapper_L,
                            const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_double *_out = (cytnx_double *)out->Mem;
      cytnx_double *_Lin = (cytnx_double *)Lin->Mem;
      cytnx_uint16 *_Rin = (cytnx_uint16 *)Rin->Mem;

      if (Lin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[0] * _Rin[i];
        }
      } else if (Rin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[i] * _Rin[0];
        }
      } else {
        if (shape.size() == 0) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (unsigned long long i = 0; i < len; i++) {
            _out[i] = _Lin[i] * _Rin[i];
          }
        } else {
          /// handle non-contiguous:
          std::vector<cytnx_uint64> accu_shape(shape.size());
          std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
          cytnx_uint64 tmp1 = 1, tmp2 = 1, tmp3 = 1;
          for (cytnx_uint64 i = 0; i < shape.size(); i++) {
            accu_shape[shape.size() - 1 - i] = tmp1;
            tmp1 *= shape[shape.size() - 1 - i];

            old_accu_shapeL[shape.size() - 1 - i] = tmp2;
            tmp2 *= shape[invmapper_L[shape.size() - 1 - i]];

            old_accu_shapeR[shape.size() - 1 - i] = tmp3;
            tmp3 *= shape[invmapper_R[shape.size() - 1 - i]];
          }

// handle non-contiguous
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (cytnx_uint64 i = 0; i < len; i++) {
            std::vector<cytnx_uint64> tmpv = c2cartesian(i, accu_shape);
            _out[i] = _Lin[cartesian2c(vec_map(tmpv, invmapper_L), old_accu_shapeL)] *
                      _Rin[cartesian2c(vec_map(tmpv, invmapper_R), old_accu_shapeR)];
          }
        }
      }
    }
    void Mul_internal_dtb(boost::intrusive_ptr<Storage_base> &out,
                          boost::intrusive_ptr<Storage_base> &Lin,
                          boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                          const std::vector<cytnx_uint64> &shape,
                          const std::vector<cytnx_uint64> &invmapper_L,
                          const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_double *_out = (cytnx_double *)out->Mem;
      cytnx_double *_Lin = (cytnx_double *)Lin->Mem;
      cytnx_bool *_Rin = (cytnx_bool *)Rin->Mem;

      if (Lin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[0] * double(_Rin[i]);
        }
      } else if (Rin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[i] * double(_Rin[0]);
        }
      } else {
        if (shape.size() == 0) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (unsigned long long i = 0; i < len; i++) {
            _out[i] = _Lin[i] * double(_Rin[i]);
          }
        } else {
          /// handle non-contiguous:
          std::vector<cytnx_uint64> accu_shape(shape.size());
          std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
          cytnx_uint64 tmp1 = 1, tmp2 = 1, tmp3 = 1;
          for (cytnx_uint64 i = 0; i < shape.size(); i++) {
            accu_shape[shape.size() - 1 - i] = tmp1;
            tmp1 *= shape[shape.size() - 1 - i];

            old_accu_shapeL[shape.size() - 1 - i] = tmp2;
            tmp2 *= shape[invmapper_L[shape.size() - 1 - i]];

            old_accu_shapeR[shape.size() - 1 - i] = tmp3;
            tmp3 *= shape[invmapper_R[shape.size() - 1 - i]];
          }

// handle non-contiguous
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (cytnx_uint64 i = 0; i < len; i++) {
            std::vector<cytnx_uint64> tmpv = c2cartesian(i, accu_shape);
            _out[i] = _Lin[cartesian2c(vec_map(tmpv, invmapper_L), old_accu_shapeL)] *
                      double(_Rin[cartesian2c(vec_map(tmpv, invmapper_R), old_accu_shapeR)]);
          }
        }
      }
    }
    //---------------------------
    void Mul_internal_ftcd(boost::intrusive_ptr<Storage_base> &out,
                           boost::intrusive_ptr<Storage_base> &Lin,
                           boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                           const std::vector<cytnx_uint64> &shape,
                           const std::vector<cytnx_uint64> &invmapper_L,
                           const std::vector<cytnx_uint64> &invmapper_R) {
      Mul_internal_cdtf(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void Mul_internal_ftcf(boost::intrusive_ptr<Storage_base> &out,
                           boost::intrusive_ptr<Storage_base> &Lin,
                           boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                           const std::vector<cytnx_uint64> &shape,
                           const std::vector<cytnx_uint64> &invmapper_L,
                           const std::vector<cytnx_uint64> &invmapper_R) {
      Mul_internal_cftf(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void Mul_internal_ftd(boost::intrusive_ptr<Storage_base> &out,
                          boost::intrusive_ptr<Storage_base> &Lin,
                          boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                          const std::vector<cytnx_uint64> &shape,
                          const std::vector<cytnx_uint64> &invmapper_L,
                          const std::vector<cytnx_uint64> &invmapper_R) {
      Mul_internal_dtf(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void Mul_internal_ftf(boost::intrusive_ptr<Storage_base> &out,
                          boost::intrusive_ptr<Storage_base> &Lin,
                          boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                          const std::vector<cytnx_uint64> &shape,
                          const std::vector<cytnx_uint64> &invmapper_L,
                          const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_float *_out = (cytnx_float *)out->Mem;
      cytnx_float *_Lin = (cytnx_float *)Lin->Mem;
      cytnx_float *_Rin = (cytnx_float *)Rin->Mem;

      if (Lin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[0] * _Rin[i];
        }
      } else if (Rin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[i] * _Rin[0];
        }
      } else {
        if (shape.size() == 0) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (unsigned long long i = 0; i < len; i++) {
            _out[i] = _Lin[i] * _Rin[i];
          }
        } else {
          /// handle non-contiguous:
          std::vector<cytnx_uint64> accu_shape(shape.size());
          std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
          cytnx_uint64 tmp1 = 1, tmp2 = 1, tmp3 = 1;
          for (cytnx_uint64 i = 0; i < shape.size(); i++) {
            accu_shape[shape.size() - 1 - i] = tmp1;
            tmp1 *= shape[shape.size() - 1 - i];

            old_accu_shapeL[shape.size() - 1 - i] = tmp2;
            tmp2 *= shape[invmapper_L[shape.size() - 1 - i]];

            old_accu_shapeR[shape.size() - 1 - i] = tmp3;
            tmp3 *= shape[invmapper_R[shape.size() - 1 - i]];
          }

// handle non-contiguous
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (cytnx_uint64 i = 0; i < len; i++) {
            std::vector<cytnx_uint64> tmpv = c2cartesian(i, accu_shape);
            _out[i] = _Lin[cartesian2c(vec_map(tmpv, invmapper_L), old_accu_shapeL)] *
                      _Rin[cartesian2c(vec_map(tmpv, invmapper_R), old_accu_shapeR)];
          }
        }
      }
    }
    void Mul_internal_ftu64(boost::intrusive_ptr<Storage_base> &out,
                            boost::intrusive_ptr<Storage_base> &Lin,
                            boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                            const std::vector<cytnx_uint64> &shape,
                            const std::vector<cytnx_uint64> &invmapper_L,
                            const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_float *_out = (cytnx_float *)out->Mem;
      cytnx_float *_Lin = (cytnx_float *)Lin->Mem;
      cytnx_uint64 *_Rin = (cytnx_uint64 *)Rin->Mem;

      if (Lin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[0] * _Rin[i];
        }
      } else if (Rin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[i] * _Rin[0];
        }
      } else {
        if (shape.size() == 0) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (unsigned long long i = 0; i < len; i++) {
            _out[i] = _Lin[i] * _Rin[i];
          }
        } else {
          /// handle non-contiguous:
          std::vector<cytnx_uint64> accu_shape(shape.size());
          std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
          cytnx_uint64 tmp1 = 1, tmp2 = 1, tmp3 = 1;
          for (cytnx_uint64 i = 0; i < shape.size(); i++) {
            accu_shape[shape.size() - 1 - i] = tmp1;
            tmp1 *= shape[shape.size() - 1 - i];

            old_accu_shapeL[shape.size() - 1 - i] = tmp2;
            tmp2 *= shape[invmapper_L[shape.size() - 1 - i]];

            old_accu_shapeR[shape.size() - 1 - i] = tmp3;
            tmp3 *= shape[invmapper_R[shape.size() - 1 - i]];
          }

// handle non-contiguous
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (cytnx_uint64 i = 0; i < len; i++) {
            std::vector<cytnx_uint64> tmpv = c2cartesian(i, accu_shape);
            _out[i] = _Lin[cartesian2c(vec_map(tmpv, invmapper_L), old_accu_shapeL)] *
                      _Rin[cartesian2c(vec_map(tmpv, invmapper_R), old_accu_shapeR)];
          }
        }
      }
    }
    void Mul_internal_ftu32(boost::intrusive_ptr<Storage_base> &out,
                            boost::intrusive_ptr<Storage_base> &Lin,
                            boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                            const std::vector<cytnx_uint64> &shape,
                            const std::vector<cytnx_uint64> &invmapper_L,
                            const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_float *_out = (cytnx_float *)out->Mem;
      cytnx_float *_Lin = (cytnx_float *)Lin->Mem;
      cytnx_uint32 *_Rin = (cytnx_uint32 *)Rin->Mem;

      if (Lin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[0] * _Rin[i];
        }
      } else if (Rin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[i] * _Rin[0];
        }
      } else {
        if (shape.size() == 0) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (unsigned long long i = 0; i < len; i++) {
            _out[i] = _Lin[i] * _Rin[i];
          }
        } else {
          /// handle non-contiguous:
          std::vector<cytnx_uint64> accu_shape(shape.size());
          std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
          cytnx_uint64 tmp1 = 1, tmp2 = 1, tmp3 = 1;
          for (cytnx_uint64 i = 0; i < shape.size(); i++) {
            accu_shape[shape.size() - 1 - i] = tmp1;
            tmp1 *= shape[shape.size() - 1 - i];

            old_accu_shapeL[shape.size() - 1 - i] = tmp2;
            tmp2 *= shape[invmapper_L[shape.size() - 1 - i]];

            old_accu_shapeR[shape.size() - 1 - i] = tmp3;
            tmp3 *= shape[invmapper_R[shape.size() - 1 - i]];
          }

// handle non-contiguous
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (cytnx_uint64 i = 0; i < len; i++) {
            std::vector<cytnx_uint64> tmpv = c2cartesian(i, accu_shape);
            _out[i] = _Lin[cartesian2c(vec_map(tmpv, invmapper_L), old_accu_shapeL)] *
                      _Rin[cartesian2c(vec_map(tmpv, invmapper_R), old_accu_shapeR)];
          }
        }
      }
    }
    void Mul_internal_fti64(boost::intrusive_ptr<Storage_base> &out,
                            boost::intrusive_ptr<Storage_base> &Lin,
                            boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                            const std::vector<cytnx_uint64> &shape,
                            const std::vector<cytnx_uint64> &invmapper_L,
                            const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_float *_out = (cytnx_float *)out->Mem;
      cytnx_float *_Lin = (cytnx_float *)Lin->Mem;
      cytnx_int64 *_Rin = (cytnx_int64 *)Rin->Mem;

      if (Lin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[0] * _Rin[i];
        }
      } else if (Rin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[i] * _Rin[0];
        }
      } else {
        if (shape.size() == 0) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (unsigned long long i = 0; i < len; i++) {
            _out[i] = _Lin[i] * _Rin[i];
          }
        } else {
          /// handle non-contiguous:
          std::vector<cytnx_uint64> accu_shape(shape.size());
          std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
          cytnx_uint64 tmp1 = 1, tmp2 = 1, tmp3 = 1;
          for (cytnx_uint64 i = 0; i < shape.size(); i++) {
            accu_shape[shape.size() - 1 - i] = tmp1;
            tmp1 *= shape[shape.size() - 1 - i];

            old_accu_shapeL[shape.size() - 1 - i] = tmp2;
            tmp2 *= shape[invmapper_L[shape.size() - 1 - i]];

            old_accu_shapeR[shape.size() - 1 - i] = tmp3;
            tmp3 *= shape[invmapper_R[shape.size() - 1 - i]];
          }

// handle non-contiguous
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (cytnx_uint64 i = 0; i < len; i++) {
            std::vector<cytnx_uint64> tmpv = c2cartesian(i, accu_shape);
            _out[i] = _Lin[cartesian2c(vec_map(tmpv, invmapper_L), old_accu_shapeL)] *
                      _Rin[cartesian2c(vec_map(tmpv, invmapper_R), old_accu_shapeR)];
          }
        }
      }
    }
    void Mul_internal_fti32(boost::intrusive_ptr<Storage_base> &out,
                            boost::intrusive_ptr<Storage_base> &Lin,
                            boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                            const std::vector<cytnx_uint64> &shape,
                            const std::vector<cytnx_uint64> &invmapper_L,
                            const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_float *_out = (cytnx_float *)out->Mem;
      cytnx_float *_Lin = (cytnx_float *)Lin->Mem;
      cytnx_int32 *_Rin = (cytnx_int32 *)Rin->Mem;

      if (Lin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[0] * _Rin[i];
        }
      } else if (Rin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[i] * _Rin[0];
        }
      } else {
        if (shape.size() == 0) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (unsigned long long i = 0; i < len; i++) {
            _out[i] = _Lin[i] * _Rin[i];
          }
        } else {
          /// handle non-contiguous:
          std::vector<cytnx_uint64> accu_shape(shape.size());
          std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
          cytnx_uint64 tmp1 = 1, tmp2 = 1, tmp3 = 1;
          for (cytnx_uint64 i = 0; i < shape.size(); i++) {
            accu_shape[shape.size() - 1 - i] = tmp1;
            tmp1 *= shape[shape.size() - 1 - i];

            old_accu_shapeL[shape.size() - 1 - i] = tmp2;
            tmp2 *= shape[invmapper_L[shape.size() - 1 - i]];

            old_accu_shapeR[shape.size() - 1 - i] = tmp3;
            tmp3 *= shape[invmapper_R[shape.size() - 1 - i]];
          }

// handle non-contiguous
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (cytnx_uint64 i = 0; i < len; i++) {
            std::vector<cytnx_uint64> tmpv = c2cartesian(i, accu_shape);
            _out[i] = _Lin[cartesian2c(vec_map(tmpv, invmapper_L), old_accu_shapeL)] *
                      _Rin[cartesian2c(vec_map(tmpv, invmapper_R), old_accu_shapeR)];
          }
        }
      }
    }
    void Mul_internal_fti16(boost::intrusive_ptr<Storage_base> &out,
                            boost::intrusive_ptr<Storage_base> &Lin,
                            boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                            const std::vector<cytnx_uint64> &shape,
                            const std::vector<cytnx_uint64> &invmapper_L,
                            const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_float *_out = (cytnx_float *)out->Mem;
      cytnx_float *_Lin = (cytnx_float *)Lin->Mem;
      cytnx_int16 *_Rin = (cytnx_int16 *)Rin->Mem;

      if (Lin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[0] * _Rin[i];
        }
      } else if (Rin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[i] * _Rin[0];
        }
      } else {
        if (shape.size() == 0) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (unsigned long long i = 0; i < len; i++) {
            _out[i] = _Lin[i] * _Rin[i];
          }
        } else {
          /// handle non-contiguous:
          std::vector<cytnx_uint64> accu_shape(shape.size());
          std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
          cytnx_uint64 tmp1 = 1, tmp2 = 1, tmp3 = 1;
          for (cytnx_uint64 i = 0; i < shape.size(); i++) {
            accu_shape[shape.size() - 1 - i] = tmp1;
            tmp1 *= shape[shape.size() - 1 - i];

            old_accu_shapeL[shape.size() - 1 - i] = tmp2;
            tmp2 *= shape[invmapper_L[shape.size() - 1 - i]];

            old_accu_shapeR[shape.size() - 1 - i] = tmp3;
            tmp3 *= shape[invmapper_R[shape.size() - 1 - i]];
          }

// handle non-contiguous
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (cytnx_uint64 i = 0; i < len; i++) {
            std::vector<cytnx_uint64> tmpv = c2cartesian(i, accu_shape);
            _out[i] = _Lin[cartesian2c(vec_map(tmpv, invmapper_L), old_accu_shapeL)] *
                      _Rin[cartesian2c(vec_map(tmpv, invmapper_R), old_accu_shapeR)];
          }
        }
      }
    }
    void Mul_internal_ftu16(boost::intrusive_ptr<Storage_base> &out,
                            boost::intrusive_ptr<Storage_base> &Lin,
                            boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                            const std::vector<cytnx_uint64> &shape,
                            const std::vector<cytnx_uint64> &invmapper_L,
                            const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_float *_out = (cytnx_float *)out->Mem;
      cytnx_float *_Lin = (cytnx_float *)Lin->Mem;
      cytnx_uint16 *_Rin = (cytnx_uint16 *)Rin->Mem;

      if (Lin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[0] * _Rin[i];
        }
      } else if (Rin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[i] * _Rin[0];
        }
      } else {
        if (shape.size() == 0) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (unsigned long long i = 0; i < len; i++) {
            _out[i] = _Lin[i] * _Rin[i];
          }
        } else {
          /// handle non-contiguous:
          std::vector<cytnx_uint64> accu_shape(shape.size());
          std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
          cytnx_uint64 tmp1 = 1, tmp2 = 1, tmp3 = 1;
          for (cytnx_uint64 i = 0; i < shape.size(); i++) {
            accu_shape[shape.size() - 1 - i] = tmp1;
            tmp1 *= shape[shape.size() - 1 - i];

            old_accu_shapeL[shape.size() - 1 - i] = tmp2;
            tmp2 *= shape[invmapper_L[shape.size() - 1 - i]];

            old_accu_shapeR[shape.size() - 1 - i] = tmp3;
            tmp3 *= shape[invmapper_R[shape.size() - 1 - i]];
          }

// handle non-contiguous
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (cytnx_uint64 i = 0; i < len; i++) {
            std::vector<cytnx_uint64> tmpv = c2cartesian(i, accu_shape);
            _out[i] = _Lin[cartesian2c(vec_map(tmpv, invmapper_L), old_accu_shapeL)] *
                      _Rin[cartesian2c(vec_map(tmpv, invmapper_R), old_accu_shapeR)];
          }
        }
      }
    }
    void Mul_internal_ftb(boost::intrusive_ptr<Storage_base> &out,
                          boost::intrusive_ptr<Storage_base> &Lin,
                          boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                          const std::vector<cytnx_uint64> &shape,
                          const std::vector<cytnx_uint64> &invmapper_L,
                          const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_float *_out = (cytnx_float *)out->Mem;
      cytnx_float *_Lin = (cytnx_float *)Lin->Mem;
      cytnx_bool *_Rin = (cytnx_bool *)Rin->Mem;

      if (Lin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[0] * float(_Rin[i]);
        }
      } else if (Rin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[i] * float(_Rin[0]);
        }
      } else {
        if (shape.size() == 0) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (unsigned long long i = 0; i < len; i++) {
            _out[i] = _Lin[i] * float(_Rin[i]);
          }
        } else {
          /// handle non-contiguous:
          std::vector<cytnx_uint64> accu_shape(shape.size());
          std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
          cytnx_uint64 tmp1 = 1, tmp2 = 1, tmp3 = 1;
          for (cytnx_uint64 i = 0; i < shape.size(); i++) {
            accu_shape[shape.size() - 1 - i] = tmp1;
            tmp1 *= shape[shape.size() - 1 - i];

            old_accu_shapeL[shape.size() - 1 - i] = tmp2;
            tmp2 *= shape[invmapper_L[shape.size() - 1 - i]];

            old_accu_shapeR[shape.size() - 1 - i] = tmp3;
            tmp3 *= shape[invmapper_R[shape.size() - 1 - i]];
          }

// handle non-contiguous
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (cytnx_uint64 i = 0; i < len; i++) {
            std::vector<cytnx_uint64> tmpv = c2cartesian(i, accu_shape);
            _out[i] = _Lin[cartesian2c(vec_map(tmpv, invmapper_L), old_accu_shapeL)] *
                      float(_Rin[cartesian2c(vec_map(tmpv, invmapper_R), old_accu_shapeR)]);
          }
        }
      }
    }

    //-------------------
    void Mul_internal_i64tcd(boost::intrusive_ptr<Storage_base> &out,
                             boost::intrusive_ptr<Storage_base> &Lin,
                             boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                             const std::vector<cytnx_uint64> &shape,
                             const std::vector<cytnx_uint64> &invmapper_L,
                             const std::vector<cytnx_uint64> &invmapper_R) {
      Mul_internal_cdti64(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void Mul_internal_i64tcf(boost::intrusive_ptr<Storage_base> &out,
                             boost::intrusive_ptr<Storage_base> &Lin,
                             boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                             const std::vector<cytnx_uint64> &shape,
                             const std::vector<cytnx_uint64> &invmapper_L,
                             const std::vector<cytnx_uint64> &invmapper_R) {
      Mul_internal_cfti64(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void Mul_internal_i64td(boost::intrusive_ptr<Storage_base> &out,
                            boost::intrusive_ptr<Storage_base> &Lin,
                            boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                            const std::vector<cytnx_uint64> &shape,
                            const std::vector<cytnx_uint64> &invmapper_L,
                            const std::vector<cytnx_uint64> &invmapper_R) {
      Mul_internal_dti64(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void Mul_internal_i64tf(boost::intrusive_ptr<Storage_base> &out,
                            boost::intrusive_ptr<Storage_base> &Lin,
                            boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                            const std::vector<cytnx_uint64> &shape,
                            const std::vector<cytnx_uint64> &invmapper_L,
                            const std::vector<cytnx_uint64> &invmapper_R) {
      Mul_internal_fti64(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void Mul_internal_i64ti64(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_int64 *_out = (cytnx_int64 *)out->Mem;
      cytnx_int64 *_Lin = (cytnx_int64 *)Lin->Mem;
      cytnx_int64 *_Rin = (cytnx_int64 *)Rin->Mem;

      if (Lin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[0] * _Rin[i];
        }
      } else if (Rin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[i] * _Rin[0];
        }
      } else {
        if (shape.size() == 0) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (unsigned long long i = 0; i < len; i++) {
            _out[i] = _Lin[i] * _Rin[i];
          }
        } else {
          /// handle non-contiguous:
          std::vector<cytnx_uint64> accu_shape(shape.size());
          std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
          cytnx_uint64 tmp1 = 1, tmp2 = 1, tmp3 = 1;
          for (cytnx_uint64 i = 0; i < shape.size(); i++) {
            accu_shape[shape.size() - 1 - i] = tmp1;
            tmp1 *= shape[shape.size() - 1 - i];

            old_accu_shapeL[shape.size() - 1 - i] = tmp2;
            tmp2 *= shape[invmapper_L[shape.size() - 1 - i]];

            old_accu_shapeR[shape.size() - 1 - i] = tmp3;
            tmp3 *= shape[invmapper_R[shape.size() - 1 - i]];
          }

// handle non-contiguous
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (cytnx_uint64 i = 0; i < len; i++) {
            std::vector<cytnx_uint64> tmpv = c2cartesian(i, accu_shape);
            _out[i] = _Lin[cartesian2c(vec_map(tmpv, invmapper_L), old_accu_shapeL)] *
                      _Rin[cartesian2c(vec_map(tmpv, invmapper_R), old_accu_shapeR)];
          }
        }
      }
    }
    void Mul_internal_i64tu64(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_int64 *_out = (cytnx_int64 *)out->Mem;
      cytnx_int64 *_Lin = (cytnx_int64 *)Lin->Mem;
      cytnx_uint64 *_Rin = (cytnx_uint64 *)Rin->Mem;

      if (Lin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[0] * _Rin[i];
        }
      } else if (Rin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[i] * _Rin[0];
        }
      } else {
        if (shape.size() == 0) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (unsigned long long i = 0; i < len; i++) {
            _out[i] = _Lin[i] * _Rin[i];
          }
        } else {
          /// handle non-contiguous:
          std::vector<cytnx_uint64> accu_shape(shape.size());
          std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
          cytnx_uint64 tmp1 = 1, tmp2 = 1, tmp3 = 1;
          for (cytnx_uint64 i = 0; i < shape.size(); i++) {
            accu_shape[shape.size() - 1 - i] = tmp1;
            tmp1 *= shape[shape.size() - 1 - i];

            old_accu_shapeL[shape.size() - 1 - i] = tmp2;
            tmp2 *= shape[invmapper_L[shape.size() - 1 - i]];

            old_accu_shapeR[shape.size() - 1 - i] = tmp3;
            tmp3 *= shape[invmapper_R[shape.size() - 1 - i]];
          }

// handle non-contiguous
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (cytnx_uint64 i = 0; i < len; i++) {
            std::vector<cytnx_uint64> tmpv = c2cartesian(i, accu_shape);
            _out[i] = _Lin[cartesian2c(vec_map(tmpv, invmapper_L), old_accu_shapeL)] *
                      _Rin[cartesian2c(vec_map(tmpv, invmapper_R), old_accu_shapeR)];
          }
        }
      }
    }
    void Mul_internal_i64ti32(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_int64 *_out = (cytnx_int64 *)out->Mem;
      cytnx_int64 *_Lin = (cytnx_int64 *)Lin->Mem;
      cytnx_int32 *_Rin = (cytnx_int32 *)Rin->Mem;

      if (Lin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[0] * _Rin[i];
        }
      } else if (Rin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[i] * _Rin[0];
        }
      } else {
        if (shape.size() == 0) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (unsigned long long i = 0; i < len; i++) {
            _out[i] = _Lin[i] * _Rin[i];
          }
        } else {
          /// handle non-contiguous:
          std::vector<cytnx_uint64> accu_shape(shape.size());
          std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
          cytnx_uint64 tmp1 = 1, tmp2 = 1, tmp3 = 1;
          for (cytnx_uint64 i = 0; i < shape.size(); i++) {
            accu_shape[shape.size() - 1 - i] = tmp1;
            tmp1 *= shape[shape.size() - 1 - i];

            old_accu_shapeL[shape.size() - 1 - i] = tmp2;
            tmp2 *= shape[invmapper_L[shape.size() - 1 - i]];

            old_accu_shapeR[shape.size() - 1 - i] = tmp3;
            tmp3 *= shape[invmapper_R[shape.size() - 1 - i]];
          }

// handle non-contiguous
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (cytnx_uint64 i = 0; i < len; i++) {
            std::vector<cytnx_uint64> tmpv = c2cartesian(i, accu_shape);
            _out[i] = _Lin[cartesian2c(vec_map(tmpv, invmapper_L), old_accu_shapeL)] *
                      _Rin[cartesian2c(vec_map(tmpv, invmapper_R), old_accu_shapeR)];
          }
        }
      }
    }
    void Mul_internal_i64tu32(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_int64 *_out = (cytnx_int64 *)out->Mem;
      cytnx_int64 *_Lin = (cytnx_int64 *)Lin->Mem;
      cytnx_uint32 *_Rin = (cytnx_uint32 *)Rin->Mem;

      if (Lin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[0] * _Rin[i];
        }
      } else if (Rin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[i] * _Rin[0];
        }
      } else {
        if (shape.size() == 0) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (unsigned long long i = 0; i < len; i++) {
            _out[i] = _Lin[i] * _Rin[i];
          }
        } else {
          /// handle non-contiguous:
          std::vector<cytnx_uint64> accu_shape(shape.size());
          std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
          cytnx_uint64 tmp1 = 1, tmp2 = 1, tmp3 = 1;
          for (cytnx_uint64 i = 0; i < shape.size(); i++) {
            accu_shape[shape.size() - 1 - i] = tmp1;
            tmp1 *= shape[shape.size() - 1 - i];

            old_accu_shapeL[shape.size() - 1 - i] = tmp2;
            tmp2 *= shape[invmapper_L[shape.size() - 1 - i]];

            old_accu_shapeR[shape.size() - 1 - i] = tmp3;
            tmp3 *= shape[invmapper_R[shape.size() - 1 - i]];
          }

// handle non-contiguous
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (cytnx_uint64 i = 0; i < len; i++) {
            std::vector<cytnx_uint64> tmpv = c2cartesian(i, accu_shape);
            _out[i] = _Lin[cartesian2c(vec_map(tmpv, invmapper_L), old_accu_shapeL)] *
                      _Rin[cartesian2c(vec_map(tmpv, invmapper_R), old_accu_shapeR)];
          }
        }
      }
    }
    void Mul_internal_i64ti16(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_int64 *_out = (cytnx_int64 *)out->Mem;
      cytnx_int64 *_Lin = (cytnx_int64 *)Lin->Mem;
      cytnx_int16 *_Rin = (cytnx_int16 *)Rin->Mem;

      if (Lin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[0] * _Rin[i];
        }
      } else if (Rin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[i] * _Rin[0];
        }
      } else {
        if (shape.size() == 0) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (unsigned long long i = 0; i < len; i++) {
            _out[i] = _Lin[i] * _Rin[i];
          }
        } else {
          /// handle non-contiguous:
          std::vector<cytnx_uint64> accu_shape(shape.size());
          std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
          cytnx_uint64 tmp1 = 1, tmp2 = 1, tmp3 = 1;
          for (cytnx_uint64 i = 0; i < shape.size(); i++) {
            accu_shape[shape.size() - 1 - i] = tmp1;
            tmp1 *= shape[shape.size() - 1 - i];

            old_accu_shapeL[shape.size() - 1 - i] = tmp2;
            tmp2 *= shape[invmapper_L[shape.size() - 1 - i]];

            old_accu_shapeR[shape.size() - 1 - i] = tmp3;
            tmp3 *= shape[invmapper_R[shape.size() - 1 - i]];
          }

// handle non-contiguous
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (cytnx_uint64 i = 0; i < len; i++) {
            std::vector<cytnx_uint64> tmpv = c2cartesian(i, accu_shape);
            _out[i] = _Lin[cartesian2c(vec_map(tmpv, invmapper_L), old_accu_shapeL)] *
                      _Rin[cartesian2c(vec_map(tmpv, invmapper_R), old_accu_shapeR)];
          }
        }
      }
    }
    void Mul_internal_i64tu16(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_int64 *_out = (cytnx_int64 *)out->Mem;
      cytnx_int64 *_Lin = (cytnx_int64 *)Lin->Mem;
      cytnx_uint16 *_Rin = (cytnx_uint16 *)Rin->Mem;

      if (Lin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[0] * _Rin[i];
        }
      } else if (Rin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[i] * _Rin[0];
        }
      } else {
        if (shape.size() == 0) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (unsigned long long i = 0; i < len; i++) {
            _out[i] = _Lin[i] * _Rin[i];
          }
        } else {
          /// handle non-contiguous:
          std::vector<cytnx_uint64> accu_shape(shape.size());
          std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
          cytnx_uint64 tmp1 = 1, tmp2 = 1, tmp3 = 1;
          for (cytnx_uint64 i = 0; i < shape.size(); i++) {
            accu_shape[shape.size() - 1 - i] = tmp1;
            tmp1 *= shape[shape.size() - 1 - i];

            old_accu_shapeL[shape.size() - 1 - i] = tmp2;
            tmp2 *= shape[invmapper_L[shape.size() - 1 - i]];

            old_accu_shapeR[shape.size() - 1 - i] = tmp3;
            tmp3 *= shape[invmapper_R[shape.size() - 1 - i]];
          }

// handle non-contiguous
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (cytnx_uint64 i = 0; i < len; i++) {
            std::vector<cytnx_uint64> tmpv = c2cartesian(i, accu_shape);
            _out[i] = _Lin[cartesian2c(vec_map(tmpv, invmapper_L), old_accu_shapeL)] *
                      _Rin[cartesian2c(vec_map(tmpv, invmapper_R), old_accu_shapeR)];
          }
        }
      }
    }
    void Mul_internal_i64tb(boost::intrusive_ptr<Storage_base> &out,
                            boost::intrusive_ptr<Storage_base> &Lin,
                            boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                            const std::vector<cytnx_uint64> &shape,
                            const std::vector<cytnx_uint64> &invmapper_L,
                            const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_int64 *_out = (cytnx_int64 *)out->Mem;
      cytnx_int64 *_Lin = (cytnx_int64 *)Lin->Mem;
      cytnx_bool *_Rin = (cytnx_bool *)Rin->Mem;

      if (Lin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[0] * cytnx_int64(_Rin[i]);
        }
      } else if (Rin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[i] * cytnx_int64(_Rin[0]);
        }
      } else {
        if (shape.size() == 0) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (unsigned long long i = 0; i < len; i++) {
            _out[i] = _Lin[i] * cytnx_int64(_Rin[i]);
          }
        } else {
          /// handle non-contiguous:
          std::vector<cytnx_uint64> accu_shape(shape.size());
          std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
          cytnx_uint64 tmp1 = 1, tmp2 = 1, tmp3 = 1;
          for (cytnx_uint64 i = 0; i < shape.size(); i++) {
            accu_shape[shape.size() - 1 - i] = tmp1;
            tmp1 *= shape[shape.size() - 1 - i];

            old_accu_shapeL[shape.size() - 1 - i] = tmp2;
            tmp2 *= shape[invmapper_L[shape.size() - 1 - i]];

            old_accu_shapeR[shape.size() - 1 - i] = tmp3;
            tmp3 *= shape[invmapper_R[shape.size() - 1 - i]];
          }

// handle non-contiguous
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (cytnx_uint64 i = 0; i < len; i++) {
            std::vector<cytnx_uint64> tmpv = c2cartesian(i, accu_shape);
            _out[i] = _Lin[cartesian2c(vec_map(tmpv, invmapper_L), old_accu_shapeL)] *
                      cytnx_int64(_Rin[cartesian2c(vec_map(tmpv, invmapper_R), old_accu_shapeR)]);
          }
        }
      }
    }
    //-------------------------
    void Mul_internal_u64tcd(boost::intrusive_ptr<Storage_base> &out,
                             boost::intrusive_ptr<Storage_base> &Lin,
                             boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                             const std::vector<cytnx_uint64> &shape,
                             const std::vector<cytnx_uint64> &invmapper_L,
                             const std::vector<cytnx_uint64> &invmapper_R) {
      Mul_internal_cdtu64(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void Mul_internal_u64tcf(boost::intrusive_ptr<Storage_base> &out,
                             boost::intrusive_ptr<Storage_base> &Lin,
                             boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                             const std::vector<cytnx_uint64> &shape,
                             const std::vector<cytnx_uint64> &invmapper_L,
                             const std::vector<cytnx_uint64> &invmapper_R) {
      Mul_internal_cftu64(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void Mul_internal_u64td(boost::intrusive_ptr<Storage_base> &out,
                            boost::intrusive_ptr<Storage_base> &Lin,
                            boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                            const std::vector<cytnx_uint64> &shape,
                            const std::vector<cytnx_uint64> &invmapper_L,
                            const std::vector<cytnx_uint64> &invmapper_R) {
      Mul_internal_dtu64(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void Mul_internal_u64tf(boost::intrusive_ptr<Storage_base> &out,
                            boost::intrusive_ptr<Storage_base> &Lin,
                            boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                            const std::vector<cytnx_uint64> &shape,
                            const std::vector<cytnx_uint64> &invmapper_L,
                            const std::vector<cytnx_uint64> &invmapper_R) {
      Mul_internal_ftu64(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void Mul_internal_u64ti64(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R) {
      Mul_internal_i64tu64(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void Mul_internal_u64tu64(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_uint64 *_out = (cytnx_uint64 *)out->Mem;
      cytnx_uint64 *_Lin = (cytnx_uint64 *)Lin->Mem;
      cytnx_uint64 *_Rin = (cytnx_uint64 *)Rin->Mem;

      if (Lin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[0] * _Rin[i];
        }
      } else if (Rin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[i] * _Rin[0];
        }
      } else {
        if (shape.size() == 0) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (unsigned long long i = 0; i < len; i++) {
            _out[i] = _Lin[i] * _Rin[i];
          }
        } else {
          /// handle non-contiguous:
          std::vector<cytnx_uint64> accu_shape(shape.size());
          std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
          cytnx_uint64 tmp1 = 1, tmp2 = 1, tmp3 = 1;
          for (cytnx_uint64 i = 0; i < shape.size(); i++) {
            accu_shape[shape.size() - 1 - i] = tmp1;
            tmp1 *= shape[shape.size() - 1 - i];

            old_accu_shapeL[shape.size() - 1 - i] = tmp2;
            tmp2 *= shape[invmapper_L[shape.size() - 1 - i]];

            old_accu_shapeR[shape.size() - 1 - i] = tmp3;
            tmp3 *= shape[invmapper_R[shape.size() - 1 - i]];
          }

// handle non-contiguous
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (cytnx_uint64 i = 0; i < len; i++) {
            std::vector<cytnx_uint64> tmpv = c2cartesian(i, accu_shape);
            _out[i] = _Lin[cartesian2c(vec_map(tmpv, invmapper_L), old_accu_shapeL)] *
                      _Rin[cartesian2c(vec_map(tmpv, invmapper_R), old_accu_shapeR)];
          }
        }
      }
    }
    void Mul_internal_u64ti32(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_uint64 *_out = (cytnx_uint64 *)out->Mem;
      cytnx_uint64 *_Lin = (cytnx_uint64 *)Lin->Mem;
      cytnx_int32 *_Rin = (cytnx_int32 *)Rin->Mem;

      if (Lin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[0] * _Rin[i];
        }
      } else if (Rin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[i] * _Rin[0];
        }
      } else {
        if (shape.size() == 0) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (unsigned long long i = 0; i < len; i++) {
            _out[i] = _Lin[i] * _Rin[i];
          }
        } else {
          /// handle non-contiguous:
          std::vector<cytnx_uint64> accu_shape(shape.size());
          std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
          cytnx_uint64 tmp1 = 1, tmp2 = 1, tmp3 = 1;
          for (cytnx_uint64 i = 0; i < shape.size(); i++) {
            accu_shape[shape.size() - 1 - i] = tmp1;
            tmp1 *= shape[shape.size() - 1 - i];

            old_accu_shapeL[shape.size() - 1 - i] = tmp2;
            tmp2 *= shape[invmapper_L[shape.size() - 1 - i]];

            old_accu_shapeR[shape.size() - 1 - i] = tmp3;
            tmp3 *= shape[invmapper_R[shape.size() - 1 - i]];
          }

// handle non-contiguous
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (cytnx_uint64 i = 0; i < len; i++) {
            std::vector<cytnx_uint64> tmpv = c2cartesian(i, accu_shape);
            _out[i] = _Lin[cartesian2c(vec_map(tmpv, invmapper_L), old_accu_shapeL)] *
                      _Rin[cartesian2c(vec_map(tmpv, invmapper_R), old_accu_shapeR)];
          }
        }
      }
    }
    void Mul_internal_u64tu32(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_uint64 *_out = (cytnx_uint64 *)out->Mem;
      cytnx_uint64 *_Lin = (cytnx_uint64 *)Lin->Mem;
      cytnx_uint32 *_Rin = (cytnx_uint32 *)Rin->Mem;

      if (Lin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[0] * _Rin[i];
        }
      } else if (Rin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[i] * _Rin[0];
        }
      } else {
        if (shape.size() == 0) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (unsigned long long i = 0; i < len; i++) {
            _out[i] = _Lin[i] * _Rin[i];
          }
        } else {
          /// handle non-contiguous:
          std::vector<cytnx_uint64> accu_shape(shape.size());
          std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
          cytnx_uint64 tmp1 = 1, tmp2 = 1, tmp3 = 1;
          for (cytnx_uint64 i = 0; i < shape.size(); i++) {
            accu_shape[shape.size() - 1 - i] = tmp1;
            tmp1 *= shape[shape.size() - 1 - i];

            old_accu_shapeL[shape.size() - 1 - i] = tmp2;
            tmp2 *= shape[invmapper_L[shape.size() - 1 - i]];

            old_accu_shapeR[shape.size() - 1 - i] = tmp3;
            tmp3 *= shape[invmapper_R[shape.size() - 1 - i]];
          }

// handle non-contiguous
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (cytnx_uint64 i = 0; i < len; i++) {
            std::vector<cytnx_uint64> tmpv = c2cartesian(i, accu_shape);
            _out[i] = _Lin[cartesian2c(vec_map(tmpv, invmapper_L), old_accu_shapeL)] *
                      _Rin[cartesian2c(vec_map(tmpv, invmapper_R), old_accu_shapeR)];
          }
        }
      }
    }
    void Mul_internal_u64ti16(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_uint64 *_out = (cytnx_uint64 *)out->Mem;
      cytnx_uint64 *_Lin = (cytnx_uint64 *)Lin->Mem;
      cytnx_int16 *_Rin = (cytnx_int16 *)Rin->Mem;

      if (Lin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[0] * _Rin[i];
        }
      } else if (Rin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[i] * _Rin[0];
        }
      } else {
        if (shape.size() == 0) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (unsigned long long i = 0; i < len; i++) {
            _out[i] = _Lin[i] * _Rin[i];
          }
        } else {
          /// handle non-contiguous:
          std::vector<cytnx_uint64> accu_shape(shape.size());
          std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
          cytnx_uint64 tmp1 = 1, tmp2 = 1, tmp3 = 1;
          for (cytnx_uint64 i = 0; i < shape.size(); i++) {
            accu_shape[shape.size() - 1 - i] = tmp1;
            tmp1 *= shape[shape.size() - 1 - i];

            old_accu_shapeL[shape.size() - 1 - i] = tmp2;
            tmp2 *= shape[invmapper_L[shape.size() - 1 - i]];

            old_accu_shapeR[shape.size() - 1 - i] = tmp3;
            tmp3 *= shape[invmapper_R[shape.size() - 1 - i]];
          }

// handle non-contiguous
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (cytnx_uint64 i = 0; i < len; i++) {
            std::vector<cytnx_uint64> tmpv = c2cartesian(i, accu_shape);
            _out[i] = _Lin[cartesian2c(vec_map(tmpv, invmapper_L), old_accu_shapeL)] *
                      _Rin[cartesian2c(vec_map(tmpv, invmapper_R), old_accu_shapeR)];
          }
        }
      }
    }
    void Mul_internal_u64tu16(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_uint64 *_out = (cytnx_uint64 *)out->Mem;
      cytnx_uint64 *_Lin = (cytnx_uint64 *)Lin->Mem;
      cytnx_uint16 *_Rin = (cytnx_uint16 *)Rin->Mem;

      if (Lin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[0] * _Rin[i];
        }
      } else if (Rin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[i] * _Rin[0];
        }
      } else {
        if (shape.size() == 0) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (unsigned long long i = 0; i < len; i++) {
            _out[i] = _Lin[i] * _Rin[i];
          }
        } else {
          /// handle non-contiguous:
          std::vector<cytnx_uint64> accu_shape(shape.size());
          std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
          cytnx_uint64 tmp1 = 1, tmp2 = 1, tmp3 = 1;
          for (cytnx_uint64 i = 0; i < shape.size(); i++) {
            accu_shape[shape.size() - 1 - i] = tmp1;
            tmp1 *= shape[shape.size() - 1 - i];

            old_accu_shapeL[shape.size() - 1 - i] = tmp2;
            tmp2 *= shape[invmapper_L[shape.size() - 1 - i]];

            old_accu_shapeR[shape.size() - 1 - i] = tmp3;
            tmp3 *= shape[invmapper_R[shape.size() - 1 - i]];
          }

// handle non-contiguous
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (cytnx_uint64 i = 0; i < len; i++) {
            std::vector<cytnx_uint64> tmpv = c2cartesian(i, accu_shape);
            _out[i] = _Lin[cartesian2c(vec_map(tmpv, invmapper_L), old_accu_shapeL)] *
                      _Rin[cartesian2c(vec_map(tmpv, invmapper_R), old_accu_shapeR)];
          }
        }
      }
    }
    void Mul_internal_u64tb(boost::intrusive_ptr<Storage_base> &out,
                            boost::intrusive_ptr<Storage_base> &Lin,
                            boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                            const std::vector<cytnx_uint64> &shape,
                            const std::vector<cytnx_uint64> &invmapper_L,
                            const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_uint64 *_out = (cytnx_uint64 *)out->Mem;
      cytnx_uint64 *_Lin = (cytnx_uint64 *)Lin->Mem;
      cytnx_bool *_Rin = (cytnx_bool *)Rin->Mem;

      if (Lin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[0] * cytnx_uint64(_Rin[i]);
        }
      } else if (Rin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[i] * cytnx_uint64(_Rin[0]);
        }
      } else {
        if (shape.size() == 0) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (unsigned long long i = 0; i < len; i++) {
            _out[i] = _Lin[i] * cytnx_uint64(_Rin[i]);
          }
        } else {
          /// handle non-contiguous:
          std::vector<cytnx_uint64> accu_shape(shape.size());
          std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
          cytnx_uint64 tmp1 = 1, tmp2 = 1, tmp3 = 1;
          for (cytnx_uint64 i = 0; i < shape.size(); i++) {
            accu_shape[shape.size() - 1 - i] = tmp1;
            tmp1 *= shape[shape.size() - 1 - i];

            old_accu_shapeL[shape.size() - 1 - i] = tmp2;
            tmp2 *= shape[invmapper_L[shape.size() - 1 - i]];

            old_accu_shapeR[shape.size() - 1 - i] = tmp3;
            tmp3 *= shape[invmapper_R[shape.size() - 1 - i]];
          }

// handle non-contiguous
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (cytnx_uint64 i = 0; i < len; i++) {
            std::vector<cytnx_uint64> tmpv = c2cartesian(i, accu_shape);
            _out[i] = _Lin[cartesian2c(vec_map(tmpv, invmapper_L), old_accu_shapeL)] *
                      cytnx_uint64(_Rin[cartesian2c(vec_map(tmpv, invmapper_R), old_accu_shapeR)]);
          }
        }
      }
    }
    //---------------------------
    void Mul_internal_i32tcd(boost::intrusive_ptr<Storage_base> &out,
                             boost::intrusive_ptr<Storage_base> &Lin,
                             boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                             const std::vector<cytnx_uint64> &shape,
                             const std::vector<cytnx_uint64> &invmapper_L,
                             const std::vector<cytnx_uint64> &invmapper_R) {
      Mul_internal_cdti32(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void Mul_internal_i32tcf(boost::intrusive_ptr<Storage_base> &out,
                             boost::intrusive_ptr<Storage_base> &Lin,
                             boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                             const std::vector<cytnx_uint64> &shape,
                             const std::vector<cytnx_uint64> &invmapper_L,
                             const std::vector<cytnx_uint64> &invmapper_R) {
      Mul_internal_cfti32(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void Mul_internal_i32td(boost::intrusive_ptr<Storage_base> &out,
                            boost::intrusive_ptr<Storage_base> &Lin,
                            boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                            const std::vector<cytnx_uint64> &shape,
                            const std::vector<cytnx_uint64> &invmapper_L,
                            const std::vector<cytnx_uint64> &invmapper_R) {
      Mul_internal_dti32(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void Mul_internal_i32tf(boost::intrusive_ptr<Storage_base> &out,
                            boost::intrusive_ptr<Storage_base> &Lin,
                            boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                            const std::vector<cytnx_uint64> &shape,
                            const std::vector<cytnx_uint64> &invmapper_L,
                            const std::vector<cytnx_uint64> &invmapper_R) {
      Mul_internal_fti32(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void Mul_internal_i32ti64(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R) {
      Mul_internal_i64ti32(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void Mul_internal_i32tu64(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R) {
      Mul_internal_u64ti32(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void Mul_internal_i32ti32(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_int32 *_out = (cytnx_int32 *)out->Mem;
      cytnx_int32 *_Lin = (cytnx_int32 *)Lin->Mem;
      cytnx_int32 *_Rin = (cytnx_int32 *)Rin->Mem;

      if (Lin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[0] * _Rin[i];
        }
      } else if (Rin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[i] * _Rin[0];
        }
      } else {
        if (shape.size() == 0) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (unsigned long long i = 0; i < len; i++) {
            _out[i] = _Lin[i] * _Rin[i];
          }
        } else {
          /// handle non-contiguous:
          std::vector<cytnx_uint64> accu_shape(shape.size());
          std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
          cytnx_uint64 tmp1 = 1, tmp2 = 1, tmp3 = 1;
          for (cytnx_uint64 i = 0; i < shape.size(); i++) {
            accu_shape[shape.size() - 1 - i] = tmp1;
            tmp1 *= shape[shape.size() - 1 - i];

            old_accu_shapeL[shape.size() - 1 - i] = tmp2;
            tmp2 *= shape[invmapper_L[shape.size() - 1 - i]];

            old_accu_shapeR[shape.size() - 1 - i] = tmp3;
            tmp3 *= shape[invmapper_R[shape.size() - 1 - i]];
          }

// handle non-contiguous
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (cytnx_uint64 i = 0; i < len; i++) {
            std::vector<cytnx_uint64> tmpv = c2cartesian(i, accu_shape);
            _out[i] = _Lin[cartesian2c(vec_map(tmpv, invmapper_L), old_accu_shapeL)] *
                      _Rin[cartesian2c(vec_map(tmpv, invmapper_R), old_accu_shapeR)];
          }
        }
      }
    }
    void Mul_internal_i32tu32(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_int32 *_out = (cytnx_int32 *)out->Mem;
      cytnx_int32 *_Lin = (cytnx_int32 *)Lin->Mem;
      cytnx_uint32 *_Rin = (cytnx_uint32 *)Rin->Mem;

      if (Lin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[0] * _Rin[i];
        }
      } else if (Rin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[i] * _Rin[0];
        }
      } else {
        if (shape.size() == 0) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (unsigned long long i = 0; i < len; i++) {
            _out[i] = _Lin[i] * _Rin[i];
          }
        } else {
          /// handle non-contiguous:
          std::vector<cytnx_uint64> accu_shape(shape.size());
          std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
          cytnx_uint64 tmp1 = 1, tmp2 = 1, tmp3 = 1;
          for (cytnx_uint64 i = 0; i < shape.size(); i++) {
            accu_shape[shape.size() - 1 - i] = tmp1;
            tmp1 *= shape[shape.size() - 1 - i];

            old_accu_shapeL[shape.size() - 1 - i] = tmp2;
            tmp2 *= shape[invmapper_L[shape.size() - 1 - i]];

            old_accu_shapeR[shape.size() - 1 - i] = tmp3;
            tmp3 *= shape[invmapper_R[shape.size() - 1 - i]];
          }

// handle non-contiguous
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (cytnx_uint64 i = 0; i < len; i++) {
            std::vector<cytnx_uint64> tmpv = c2cartesian(i, accu_shape);
            _out[i] = _Lin[cartesian2c(vec_map(tmpv, invmapper_L), old_accu_shapeL)] *
                      _Rin[cartesian2c(vec_map(tmpv, invmapper_R), old_accu_shapeR)];
          }
        }
      }
    }
    void Mul_internal_i32ti16(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_int32 *_out = (cytnx_int32 *)out->Mem;
      cytnx_int32 *_Lin = (cytnx_int32 *)Lin->Mem;
      cytnx_int16 *_Rin = (cytnx_int16 *)Rin->Mem;

      if (Lin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[0] * _Rin[i];
        }
      } else if (Rin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[i] * _Rin[0];
        }
      } else {
        if (shape.size() == 0) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (unsigned long long i = 0; i < len; i++) {
            _out[i] = _Lin[i] * _Rin[i];
          }
        } else {
          /// handle non-contiguous:
          std::vector<cytnx_uint64> accu_shape(shape.size());
          std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
          cytnx_uint64 tmp1 = 1, tmp2 = 1, tmp3 = 1;
          for (cytnx_uint64 i = 0; i < shape.size(); i++) {
            accu_shape[shape.size() - 1 - i] = tmp1;
            tmp1 *= shape[shape.size() - 1 - i];

            old_accu_shapeL[shape.size() - 1 - i] = tmp2;
            tmp2 *= shape[invmapper_L[shape.size() - 1 - i]];

            old_accu_shapeR[shape.size() - 1 - i] = tmp3;
            tmp3 *= shape[invmapper_R[shape.size() - 1 - i]];
          }

// handle non-contiguous
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (cytnx_uint64 i = 0; i < len; i++) {
            std::vector<cytnx_uint64> tmpv = c2cartesian(i, accu_shape);
            _out[i] = _Lin[cartesian2c(vec_map(tmpv, invmapper_L), old_accu_shapeL)] *
                      _Rin[cartesian2c(vec_map(tmpv, invmapper_R), old_accu_shapeR)];
          }
        }
      }
    }
    void Mul_internal_i32tu16(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_int32 *_out = (cytnx_int32 *)out->Mem;
      cytnx_int32 *_Lin = (cytnx_int32 *)Lin->Mem;
      cytnx_uint16 *_Rin = (cytnx_uint16 *)Rin->Mem;

      if (Lin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[0] * _Rin[i];
        }
      } else if (Rin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[i] * _Rin[0];
        }
      } else {
        if (shape.size() == 0) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (unsigned long long i = 0; i < len; i++) {
            _out[i] = _Lin[i] * _Rin[i];
          }
        } else {
          /// handle non-contiguous:
          std::vector<cytnx_uint64> accu_shape(shape.size());
          std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
          cytnx_uint64 tmp1 = 1, tmp2 = 1, tmp3 = 1;
          for (cytnx_uint64 i = 0; i < shape.size(); i++) {
            accu_shape[shape.size() - 1 - i] = tmp1;
            tmp1 *= shape[shape.size() - 1 - i];

            old_accu_shapeL[shape.size() - 1 - i] = tmp2;
            tmp2 *= shape[invmapper_L[shape.size() - 1 - i]];

            old_accu_shapeR[shape.size() - 1 - i] = tmp3;
            tmp3 *= shape[invmapper_R[shape.size() - 1 - i]];
          }

// handle non-contiguous
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (cytnx_uint64 i = 0; i < len; i++) {
            std::vector<cytnx_uint64> tmpv = c2cartesian(i, accu_shape);
            _out[i] = _Lin[cartesian2c(vec_map(tmpv, invmapper_L), old_accu_shapeL)] *
                      _Rin[cartesian2c(vec_map(tmpv, invmapper_R), old_accu_shapeR)];
          }
        }
      }
    }
    void Mul_internal_i32tb(boost::intrusive_ptr<Storage_base> &out,
                            boost::intrusive_ptr<Storage_base> &Lin,
                            boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                            const std::vector<cytnx_uint64> &shape,
                            const std::vector<cytnx_uint64> &invmapper_L,
                            const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_int32 *_out = (cytnx_int32 *)out->Mem;
      cytnx_int32 *_Lin = (cytnx_int32 *)Lin->Mem;
      cytnx_bool *_Rin = (cytnx_bool *)Rin->Mem;

      if (Lin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[0] * cytnx_int32(_Rin[i]);
        }
      } else if (Rin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[i] * cytnx_int32(_Rin[0]);
        }
      } else {
        if (shape.size() == 0) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (unsigned long long i = 0; i < len; i++) {
            _out[i] = _Lin[i] * cytnx_int32(_Rin[i]);
          }
        } else {
          /// handle non-contiguous:
          std::vector<cytnx_uint64> accu_shape(shape.size());
          std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
          cytnx_uint64 tmp1 = 1, tmp2 = 1, tmp3 = 1;
          for (cytnx_uint64 i = 0; i < shape.size(); i++) {
            accu_shape[shape.size() - 1 - i] = tmp1;
            tmp1 *= shape[shape.size() - 1 - i];

            old_accu_shapeL[shape.size() - 1 - i] = tmp2;
            tmp2 *= shape[invmapper_L[shape.size() - 1 - i]];

            old_accu_shapeR[shape.size() - 1 - i] = tmp3;
            tmp3 *= shape[invmapper_R[shape.size() - 1 - i]];
          }

// handle non-contiguous
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (cytnx_uint64 i = 0; i < len; i++) {
            std::vector<cytnx_uint64> tmpv = c2cartesian(i, accu_shape);
            _out[i] = _Lin[cartesian2c(vec_map(tmpv, invmapper_L), old_accu_shapeL)] *
                      cytnx_int32(_Rin[cartesian2c(vec_map(tmpv, invmapper_R), old_accu_shapeR)]);
          }
        }
      }
    }

    //-------------------------------
    void Mul_internal_u32tcd(boost::intrusive_ptr<Storage_base> &out,
                             boost::intrusive_ptr<Storage_base> &Lin,
                             boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                             const std::vector<cytnx_uint64> &shape,
                             const std::vector<cytnx_uint64> &invmapper_L,
                             const std::vector<cytnx_uint64> &invmapper_R) {
      Mul_internal_cdtu32(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void Mul_internal_u32tcf(boost::intrusive_ptr<Storage_base> &out,
                             boost::intrusive_ptr<Storage_base> &Lin,
                             boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                             const std::vector<cytnx_uint64> &shape,
                             const std::vector<cytnx_uint64> &invmapper_L,
                             const std::vector<cytnx_uint64> &invmapper_R) {
      Mul_internal_cftu32(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void Mul_internal_u32td(boost::intrusive_ptr<Storage_base> &out,
                            boost::intrusive_ptr<Storage_base> &Lin,
                            boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                            const std::vector<cytnx_uint64> &shape,
                            const std::vector<cytnx_uint64> &invmapper_L,
                            const std::vector<cytnx_uint64> &invmapper_R) {
      Mul_internal_dtu32(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void Mul_internal_u32tf(boost::intrusive_ptr<Storage_base> &out,
                            boost::intrusive_ptr<Storage_base> &Lin,
                            boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                            const std::vector<cytnx_uint64> &shape,
                            const std::vector<cytnx_uint64> &invmapper_L,
                            const std::vector<cytnx_uint64> &invmapper_R) {
      Mul_internal_ftu32(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void Mul_internal_u32ti64(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R) {
      Mul_internal_i64tu32(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void Mul_internal_u32tu64(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R) {
      Mul_internal_u64tu32(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void Mul_internal_u32ti32(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R) {
      Mul_internal_i32tu32(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void Mul_internal_u32tu32(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_uint32 *_out = (cytnx_uint32 *)out->Mem;
      cytnx_uint32 *_Lin = (cytnx_uint32 *)Lin->Mem;
      cytnx_uint32 *_Rin = (cytnx_uint32 *)Rin->Mem;

      if (Lin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[0] * _Rin[i];
        }
      } else if (Rin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[i] * _Rin[0];
        }
      } else {
        if (shape.size() == 0) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (unsigned long long i = 0; i < len; i++) {
            _out[i] = _Lin[i] * _Rin[i];
          }
        } else {
          /// handle non-contiguous:
          std::vector<cytnx_uint64> accu_shape(shape.size());
          std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
          cytnx_uint64 tmp1 = 1, tmp2 = 1, tmp3 = 1;
          for (cytnx_uint64 i = 0; i < shape.size(); i++) {
            accu_shape[shape.size() - 1 - i] = tmp1;
            tmp1 *= shape[shape.size() - 1 - i];

            old_accu_shapeL[shape.size() - 1 - i] = tmp2;
            tmp2 *= shape[invmapper_L[shape.size() - 1 - i]];

            old_accu_shapeR[shape.size() - 1 - i] = tmp3;
            tmp3 *= shape[invmapper_R[shape.size() - 1 - i]];
          }

// handle non-contiguous
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (cytnx_uint64 i = 0; i < len; i++) {
            std::vector<cytnx_uint64> tmpv = c2cartesian(i, accu_shape);
            _out[i] = _Lin[cartesian2c(vec_map(tmpv, invmapper_L), old_accu_shapeL)] *
                      _Rin[cartesian2c(vec_map(tmpv, invmapper_R), old_accu_shapeR)];
          }
        }
      }
    }
    void Mul_internal_u32ti16(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_uint32 *_out = (cytnx_uint32 *)out->Mem;
      cytnx_uint32 *_Lin = (cytnx_uint32 *)Lin->Mem;
      cytnx_int16 *_Rin = (cytnx_int16 *)Rin->Mem;

      if (Lin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[0] * _Rin[i];
        }
      } else if (Rin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[i] * _Rin[0];
        }
      } else {
        if (shape.size() == 0) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (unsigned long long i = 0; i < len; i++) {
            _out[i] = _Lin[i] * _Rin[i];
          }
        } else {
          /// handle non-contiguous:
          std::vector<cytnx_uint64> accu_shape(shape.size());
          std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
          cytnx_uint64 tmp1 = 1, tmp2 = 1, tmp3 = 1;
          for (cytnx_uint64 i = 0; i < shape.size(); i++) {
            accu_shape[shape.size() - 1 - i] = tmp1;
            tmp1 *= shape[shape.size() - 1 - i];

            old_accu_shapeL[shape.size() - 1 - i] = tmp2;
            tmp2 *= shape[invmapper_L[shape.size() - 1 - i]];

            old_accu_shapeR[shape.size() - 1 - i] = tmp3;
            tmp3 *= shape[invmapper_R[shape.size() - 1 - i]];
          }

// handle non-contiguous
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (cytnx_uint64 i = 0; i < len; i++) {
            std::vector<cytnx_uint64> tmpv = c2cartesian(i, accu_shape);
            _out[i] = _Lin[cartesian2c(vec_map(tmpv, invmapper_L), old_accu_shapeL)] *
                      _Rin[cartesian2c(vec_map(tmpv, invmapper_R), old_accu_shapeR)];
          }
        }
      }
    }
    void Mul_internal_u32tu16(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_uint32 *_out = (cytnx_uint32 *)out->Mem;
      cytnx_uint32 *_Lin = (cytnx_uint32 *)Lin->Mem;
      cytnx_uint16 *_Rin = (cytnx_uint16 *)Rin->Mem;

      if (Lin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[0] * _Rin[i];
        }
      } else if (Rin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[i] * _Rin[0];
        }
      } else {
        if (shape.size() == 0) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (unsigned long long i = 0; i < len; i++) {
            _out[i] = _Lin[i] * _Rin[i];
          }
        } else {
          /// handle non-contiguous:
          std::vector<cytnx_uint64> accu_shape(shape.size());
          std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
          cytnx_uint64 tmp1 = 1, tmp2 = 1, tmp3 = 1;
          for (cytnx_uint64 i = 0; i < shape.size(); i++) {
            accu_shape[shape.size() - 1 - i] = tmp1;
            tmp1 *= shape[shape.size() - 1 - i];

            old_accu_shapeL[shape.size() - 1 - i] = tmp2;
            tmp2 *= shape[invmapper_L[shape.size() - 1 - i]];

            old_accu_shapeR[shape.size() - 1 - i] = tmp3;
            tmp3 *= shape[invmapper_R[shape.size() - 1 - i]];
          }

// handle non-contiguous
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (cytnx_uint64 i = 0; i < len; i++) {
            std::vector<cytnx_uint64> tmpv = c2cartesian(i, accu_shape);
            _out[i] = _Lin[cartesian2c(vec_map(tmpv, invmapper_L), old_accu_shapeL)] *
                      _Rin[cartesian2c(vec_map(tmpv, invmapper_R), old_accu_shapeR)];
          }
        }
      }
    }
    void Mul_internal_u32tb(boost::intrusive_ptr<Storage_base> &out,
                            boost::intrusive_ptr<Storage_base> &Lin,
                            boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                            const std::vector<cytnx_uint64> &shape,
                            const std::vector<cytnx_uint64> &invmapper_L,
                            const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_uint32 *_out = (cytnx_uint32 *)out->Mem;
      cytnx_uint32 *_Lin = (cytnx_uint32 *)Lin->Mem;
      cytnx_bool *_Rin = (cytnx_bool *)Rin->Mem;

      if (Lin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[0] * cytnx_uint32(_Rin[i]);
        }
      } else if (Rin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[i] * cytnx_uint32(_Rin[0]);
        }
      } else {
        if (shape.size() == 0) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (unsigned long long i = 0; i < len; i++) {
            _out[i] = _Lin[i] * cytnx_uint32(_Rin[i]);
          }
        } else {
          /// handle non-contiguous:
          std::vector<cytnx_uint64> accu_shape(shape.size());
          std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
          cytnx_uint64 tmp1 = 1, tmp2 = 1, tmp3 = 1;
          for (cytnx_uint64 i = 0; i < shape.size(); i++) {
            accu_shape[shape.size() - 1 - i] = tmp1;
            tmp1 *= shape[shape.size() - 1 - i];

            old_accu_shapeL[shape.size() - 1 - i] = tmp2;
            tmp2 *= shape[invmapper_L[shape.size() - 1 - i]];

            old_accu_shapeR[shape.size() - 1 - i] = tmp3;
            tmp3 *= shape[invmapper_R[shape.size() - 1 - i]];
          }

// handle non-contiguous
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (cytnx_uint64 i = 0; i < len; i++) {
            std::vector<cytnx_uint64> tmpv = c2cartesian(i, accu_shape);
            _out[i] = _Lin[cartesian2c(vec_map(tmpv, invmapper_L), old_accu_shapeL)] *
                      cytnx_uint32(_Rin[cartesian2c(vec_map(tmpv, invmapper_R), old_accu_shapeR)]);
          }
        }
      }
    }
    //-------------------------------
    void Mul_internal_i16tcd(boost::intrusive_ptr<Storage_base> &out,
                             boost::intrusive_ptr<Storage_base> &Lin,
                             boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                             const std::vector<cytnx_uint64> &shape,
                             const std::vector<cytnx_uint64> &invmapper_L,
                             const std::vector<cytnx_uint64> &invmapper_R) {
      Mul_internal_cdti16(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void Mul_internal_i16tcf(boost::intrusive_ptr<Storage_base> &out,
                             boost::intrusive_ptr<Storage_base> &Lin,
                             boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                             const std::vector<cytnx_uint64> &shape,
                             const std::vector<cytnx_uint64> &invmapper_L,
                             const std::vector<cytnx_uint64> &invmapper_R) {
      Mul_internal_cfti16(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void Mul_internal_i16td(boost::intrusive_ptr<Storage_base> &out,
                            boost::intrusive_ptr<Storage_base> &Lin,
                            boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                            const std::vector<cytnx_uint64> &shape,
                            const std::vector<cytnx_uint64> &invmapper_L,
                            const std::vector<cytnx_uint64> &invmapper_R) {
      Mul_internal_dti16(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void Mul_internal_i16tf(boost::intrusive_ptr<Storage_base> &out,
                            boost::intrusive_ptr<Storage_base> &Lin,
                            boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                            const std::vector<cytnx_uint64> &shape,
                            const std::vector<cytnx_uint64> &invmapper_L,
                            const std::vector<cytnx_uint64> &invmapper_R) {
      Mul_internal_fti16(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void Mul_internal_i16ti64(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R) {
      Mul_internal_i64ti16(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void Mul_internal_i16tu64(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R) {
      Mul_internal_u64ti16(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void Mul_internal_i16ti32(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R) {
      Mul_internal_i32ti16(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void Mul_internal_i16tu32(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R) {
      Mul_internal_u32ti16(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void Mul_internal_i16ti16(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_int16 *_out = (cytnx_int16 *)out->Mem;
      cytnx_int16 *_Lin = (cytnx_int16 *)Lin->Mem;
      cytnx_int16 *_Rin = (cytnx_int16 *)Rin->Mem;

      if (Lin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[0] * _Rin[i];
        }
      } else if (Rin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[i] * _Rin[0];
        }
      } else {
        if (shape.size() == 0) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (unsigned long long i = 0; i < len; i++) {
            _out[i] = _Lin[i] * _Rin[i];
          }
        } else {
          /// handle non-contiguous:
          std::vector<cytnx_uint64> accu_shape(shape.size());
          std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
          cytnx_uint64 tmp1 = 1, tmp2 = 1, tmp3 = 1;
          for (cytnx_uint64 i = 0; i < shape.size(); i++) {
            accu_shape[shape.size() - 1 - i] = tmp1;
            tmp1 *= shape[shape.size() - 1 - i];

            old_accu_shapeL[shape.size() - 1 - i] = tmp2;
            tmp2 *= shape[invmapper_L[shape.size() - 1 - i]];

            old_accu_shapeR[shape.size() - 1 - i] = tmp3;
            tmp3 *= shape[invmapper_R[shape.size() - 1 - i]];
          }

// handle non-contiguous
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (cytnx_uint64 i = 0; i < len; i++) {
            std::vector<cytnx_uint64> tmpv = c2cartesian(i, accu_shape);
            _out[i] = _Lin[cartesian2c(vec_map(tmpv, invmapper_L), old_accu_shapeL)] *
                      _Rin[cartesian2c(vec_map(tmpv, invmapper_R), old_accu_shapeR)];
          }
        }
      }
    }
    void Mul_internal_i16tu16(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_int16 *_out = (cytnx_int16 *)out->Mem;
      cytnx_int16 *_Lin = (cytnx_int16 *)Lin->Mem;
      cytnx_uint16 *_Rin = (cytnx_uint16 *)Rin->Mem;

      if (Lin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[0] * _Rin[i];
        }
      } else if (Rin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[i] * _Rin[0];
        }
      } else {
        if (shape.size() == 0) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (unsigned long long i = 0; i < len; i++) {
            _out[i] = _Lin[i] * _Rin[i];
          }
        } else {
          /// handle non-contiguous:
          std::vector<cytnx_uint64> accu_shape(shape.size());
          std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
          cytnx_uint64 tmp1 = 1, tmp2 = 1, tmp3 = 1;
          for (cytnx_uint64 i = 0; i < shape.size(); i++) {
            accu_shape[shape.size() - 1 - i] = tmp1;
            tmp1 *= shape[shape.size() - 1 - i];

            old_accu_shapeL[shape.size() - 1 - i] = tmp2;
            tmp2 *= shape[invmapper_L[shape.size() - 1 - i]];

            old_accu_shapeR[shape.size() - 1 - i] = tmp3;
            tmp3 *= shape[invmapper_R[shape.size() - 1 - i]];
          }

// handle non-contiguous
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (cytnx_uint64 i = 0; i < len; i++) {
            std::vector<cytnx_uint64> tmpv = c2cartesian(i, accu_shape);
            _out[i] = _Lin[cartesian2c(vec_map(tmpv, invmapper_L), old_accu_shapeL)] *
                      _Rin[cartesian2c(vec_map(tmpv, invmapper_R), old_accu_shapeR)];
          }
        }
      }
    }
    void Mul_internal_i16tb(boost::intrusive_ptr<Storage_base> &out,
                            boost::intrusive_ptr<Storage_base> &Lin,
                            boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                            const std::vector<cytnx_uint64> &shape,
                            const std::vector<cytnx_uint64> &invmapper_L,
                            const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_int16 *_out = (cytnx_int16 *)out->Mem;
      cytnx_int16 *_Lin = (cytnx_int16 *)Lin->Mem;
      cytnx_bool *_Rin = (cytnx_bool *)Rin->Mem;

      if (Lin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[0] * cytnx_int16(_Rin[i]);
        }
      } else if (Rin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[i] * cytnx_int16(_Rin[0]);
        }
      } else {
        if (shape.size() == 0) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (unsigned long long i = 0; i < len; i++) {
            _out[i] = _Lin[i] * cytnx_int16(_Rin[i]);
          }
        } else {
          /// handle non-contiguous:
          std::vector<cytnx_uint64> accu_shape(shape.size());
          std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
          cytnx_uint64 tmp1 = 1, tmp2 = 1, tmp3 = 1;
          for (cytnx_uint64 i = 0; i < shape.size(); i++) {
            accu_shape[shape.size() - 1 - i] = tmp1;
            tmp1 *= shape[shape.size() - 1 - i];

            old_accu_shapeL[shape.size() - 1 - i] = tmp2;
            tmp2 *= shape[invmapper_L[shape.size() - 1 - i]];

            old_accu_shapeR[shape.size() - 1 - i] = tmp3;
            tmp3 *= shape[invmapper_R[shape.size() - 1 - i]];
          }

// handle non-contiguous
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (cytnx_uint64 i = 0; i < len; i++) {
            std::vector<cytnx_uint64> tmpv = c2cartesian(i, accu_shape);
            _out[i] = _Lin[cartesian2c(vec_map(tmpv, invmapper_L), old_accu_shapeL)] *
                      cytnx_int16(_Rin[cartesian2c(vec_map(tmpv, invmapper_R), old_accu_shapeR)]);
          }
        }
      }
    }
    //-------------------------------
    void Mul_internal_u16tcd(boost::intrusive_ptr<Storage_base> &out,
                             boost::intrusive_ptr<Storage_base> &Lin,
                             boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                             const std::vector<cytnx_uint64> &shape,
                             const std::vector<cytnx_uint64> &invmapper_L,
                             const std::vector<cytnx_uint64> &invmapper_R) {
      Mul_internal_cdtu16(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void Mul_internal_u16tcf(boost::intrusive_ptr<Storage_base> &out,
                             boost::intrusive_ptr<Storage_base> &Lin,
                             boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                             const std::vector<cytnx_uint64> &shape,
                             const std::vector<cytnx_uint64> &invmapper_L,
                             const std::vector<cytnx_uint64> &invmapper_R) {
      Mul_internal_cftu16(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void Mul_internal_u16td(boost::intrusive_ptr<Storage_base> &out,
                            boost::intrusive_ptr<Storage_base> &Lin,
                            boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                            const std::vector<cytnx_uint64> &shape,
                            const std::vector<cytnx_uint64> &invmapper_L,
                            const std::vector<cytnx_uint64> &invmapper_R) {
      Mul_internal_dtu16(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void Mul_internal_u16tf(boost::intrusive_ptr<Storage_base> &out,
                            boost::intrusive_ptr<Storage_base> &Lin,
                            boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                            const std::vector<cytnx_uint64> &shape,
                            const std::vector<cytnx_uint64> &invmapper_L,
                            const std::vector<cytnx_uint64> &invmapper_R) {
      Mul_internal_ftu16(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void Mul_internal_u16ti64(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R) {
      Mul_internal_i64tu16(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void Mul_internal_u16tu64(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R) {
      Mul_internal_u64tu16(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void Mul_internal_u16ti32(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R) {
      Mul_internal_i32tu16(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void Mul_internal_u16tu32(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R) {
      Mul_internal_u32tu16(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void Mul_internal_u16ti16(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R) {
      Mul_internal_i16tu16(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void Mul_internal_u16tu16(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_uint16 *_out = (cytnx_uint16 *)out->Mem;
      cytnx_uint16 *_Lin = (cytnx_uint16 *)Lin->Mem;
      cytnx_uint16 *_Rin = (cytnx_uint16 *)Rin->Mem;

      if (Lin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[0] * _Rin[i];
        }
      } else if (Rin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[i] * _Rin[0];
        }
      } else {
        if (shape.size() == 0) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (unsigned long long i = 0; i < len; i++) {
            _out[i] = _Lin[i] * _Rin[i];
          }
        } else {
          /// handle non-contiguous:
          std::vector<cytnx_uint64> accu_shape(shape.size());
          std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
          cytnx_uint64 tmp1 = 1, tmp2 = 1, tmp3 = 1;
          for (cytnx_uint64 i = 0; i < shape.size(); i++) {
            accu_shape[shape.size() - 1 - i] = tmp1;
            tmp1 *= shape[shape.size() - 1 - i];

            old_accu_shapeL[shape.size() - 1 - i] = tmp2;
            tmp2 *= shape[invmapper_L[shape.size() - 1 - i]];

            old_accu_shapeR[shape.size() - 1 - i] = tmp3;
            tmp3 *= shape[invmapper_R[shape.size() - 1 - i]];
          }

// handle non-contiguous
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (cytnx_uint64 i = 0; i < len; i++) {
            std::vector<cytnx_uint64> tmpv = c2cartesian(i, accu_shape);
            _out[i] = _Lin[cartesian2c(vec_map(tmpv, invmapper_L), old_accu_shapeL)] *
                      _Rin[cartesian2c(vec_map(tmpv, invmapper_R), old_accu_shapeR)];
          }
        }
      }
    }
    void Mul_internal_u16tb(boost::intrusive_ptr<Storage_base> &out,
                            boost::intrusive_ptr<Storage_base> &Lin,
                            boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                            const std::vector<cytnx_uint64> &shape,
                            const std::vector<cytnx_uint64> &invmapper_L,
                            const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_uint16 *_out = (cytnx_uint16 *)out->Mem;
      cytnx_uint16 *_Lin = (cytnx_uint16 *)Lin->Mem;
      cytnx_bool *_Rin = (cytnx_bool *)Rin->Mem;

      if (Lin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[0] * cytnx_int16(_Rin[i]);
        }
      } else if (Rin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[i] * cytnx_int16(_Rin[0]);
        }
      } else {
        if (shape.size() == 0) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (unsigned long long i = 0; i < len; i++) {
            _out[i] = _Lin[i] * cytnx_int16(_Rin[i]);
          }
        } else {
          /// handle non-contiguous:
          std::vector<cytnx_uint64> accu_shape(shape.size());
          std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
          cytnx_uint64 tmp1 = 1, tmp2 = 1, tmp3 = 1;
          for (cytnx_uint64 i = 0; i < shape.size(); i++) {
            accu_shape[shape.size() - 1 - i] = tmp1;
            tmp1 *= shape[shape.size() - 1 - i];

            old_accu_shapeL[shape.size() - 1 - i] = tmp2;
            tmp2 *= shape[invmapper_L[shape.size() - 1 - i]];

            old_accu_shapeR[shape.size() - 1 - i] = tmp3;
            tmp3 *= shape[invmapper_R[shape.size() - 1 - i]];
          }

// handle non-contiguous
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (cytnx_uint64 i = 0; i < len; i++) {
            std::vector<cytnx_uint64> tmpv = c2cartesian(i, accu_shape);
            _out[i] = _Lin[cartesian2c(vec_map(tmpv, invmapper_L), old_accu_shapeL)] *
                      cytnx_int16(_Rin[cartesian2c(vec_map(tmpv, invmapper_R), old_accu_shapeR)]);
          }
        }
      }
    }
    //-------------------------------
    void Mul_internal_btcd(boost::intrusive_ptr<Storage_base> &out,
                           boost::intrusive_ptr<Storage_base> &Lin,
                           boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                           const std::vector<cytnx_uint64> &shape,
                           const std::vector<cytnx_uint64> &invmapper_L,
                           const std::vector<cytnx_uint64> &invmapper_R) {
      Mul_internal_cdtb(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void Mul_internal_btcf(boost::intrusive_ptr<Storage_base> &out,
                           boost::intrusive_ptr<Storage_base> &Lin,
                           boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                           const std::vector<cytnx_uint64> &shape,
                           const std::vector<cytnx_uint64> &invmapper_L,
                           const std::vector<cytnx_uint64> &invmapper_R) {
      Mul_internal_cftb(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void Mul_internal_btd(boost::intrusive_ptr<Storage_base> &out,
                          boost::intrusive_ptr<Storage_base> &Lin,
                          boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                          const std::vector<cytnx_uint64> &shape,
                          const std::vector<cytnx_uint64> &invmapper_L,
                          const std::vector<cytnx_uint64> &invmapper_R) {
      Mul_internal_dtb(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void Mul_internal_btf(boost::intrusive_ptr<Storage_base> &out,
                          boost::intrusive_ptr<Storage_base> &Lin,
                          boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                          const std::vector<cytnx_uint64> &shape,
                          const std::vector<cytnx_uint64> &invmapper_L,
                          const std::vector<cytnx_uint64> &invmapper_R) {
      Mul_internal_ftb(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void Mul_internal_bti64(boost::intrusive_ptr<Storage_base> &out,
                            boost::intrusive_ptr<Storage_base> &Lin,
                            boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                            const std::vector<cytnx_uint64> &shape,
                            const std::vector<cytnx_uint64> &invmapper_L,
                            const std::vector<cytnx_uint64> &invmapper_R) {
      Mul_internal_i64tb(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void Mul_internal_btu64(boost::intrusive_ptr<Storage_base> &out,
                            boost::intrusive_ptr<Storage_base> &Lin,
                            boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                            const std::vector<cytnx_uint64> &shape,
                            const std::vector<cytnx_uint64> &invmapper_L,
                            const std::vector<cytnx_uint64> &invmapper_R) {
      Mul_internal_u64tb(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void Mul_internal_bti32(boost::intrusive_ptr<Storage_base> &out,
                            boost::intrusive_ptr<Storage_base> &Lin,
                            boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                            const std::vector<cytnx_uint64> &shape,
                            const std::vector<cytnx_uint64> &invmapper_L,
                            const std::vector<cytnx_uint64> &invmapper_R) {
      Mul_internal_i32tb(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void Mul_internal_btu32(boost::intrusive_ptr<Storage_base> &out,
                            boost::intrusive_ptr<Storage_base> &Lin,
                            boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                            const std::vector<cytnx_uint64> &shape,
                            const std::vector<cytnx_uint64> &invmapper_L,
                            const std::vector<cytnx_uint64> &invmapper_R) {
      Mul_internal_u32tb(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void Mul_internal_bti16(boost::intrusive_ptr<Storage_base> &out,
                            boost::intrusive_ptr<Storage_base> &Lin,
                            boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                            const std::vector<cytnx_uint64> &shape,
                            const std::vector<cytnx_uint64> &invmapper_L,
                            const std::vector<cytnx_uint64> &invmapper_R) {
      Mul_internal_i16tb(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void Mul_internal_btu16(boost::intrusive_ptr<Storage_base> &out,
                            boost::intrusive_ptr<Storage_base> &Lin,
                            boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                            const std::vector<cytnx_uint64> &shape,
                            const std::vector<cytnx_uint64> &invmapper_L,
                            const std::vector<cytnx_uint64> &invmapper_R) {
      Mul_internal_u16tb(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void Mul_internal_btb(boost::intrusive_ptr<Storage_base> &out,
                          boost::intrusive_ptr<Storage_base> &Lin,
                          boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                          const std::vector<cytnx_uint64> &shape,
                          const std::vector<cytnx_uint64> &invmapper_L,
                          const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_bool *_out = (cytnx_bool *)out->Mem;
      cytnx_bool *_Lin = (cytnx_bool *)Lin->Mem;
      cytnx_bool *_Rin = (cytnx_bool *)Rin->Mem;

      if (Lin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[0] * _Rin[i];
        }
      } else if (Rin->size() == 1) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (unsigned long long i = 0; i < len; i++) {
          _out[i] = _Lin[i] * _Rin[0];
        }
      } else {
        if (shape.size() == 0) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (unsigned long long i = 0; i < len; i++) {
            _out[i] = _Lin[i] * _Rin[i];
          }
        } else {
          /// handle non-contiguous:
          std::vector<cytnx_uint64> accu_shape(shape.size());
          std::vector<cytnx_uint64> old_accu_shapeL(shape.size()), old_accu_shapeR(shape.size());
          cytnx_uint64 tmp1 = 1, tmp2 = 1, tmp3 = 1;
          for (cytnx_uint64 i = 0; i < shape.size(); i++) {
            accu_shape[shape.size() - 1 - i] = tmp1;
            tmp1 *= shape[shape.size() - 1 - i];

            old_accu_shapeL[shape.size() - 1 - i] = tmp2;
            tmp2 *= shape[invmapper_L[shape.size() - 1 - i]];

            old_accu_shapeR[shape.size() - 1 - i] = tmp3;
            tmp3 *= shape[invmapper_R[shape.size() - 1 - i]];
          }

// handle non-contiguous
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
          for (cytnx_uint64 i = 0; i < len; i++) {
            std::vector<cytnx_uint64> tmpv = c2cartesian(i, accu_shape);
            _out[i] = _Lin[cartesian2c(vec_map(tmpv, invmapper_L), old_accu_shapeL)] *
                      _Rin[cartesian2c(vec_map(tmpv, invmapper_R), old_accu_shapeR)];
          }
        }
      }
    }

  }  // namespace linalg_internal
}  // namespace cytnx
