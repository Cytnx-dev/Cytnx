#include "cuMod_internal.hpp"
#include "../utils_internal_interface.hpp"
#include <cuda.h>
#ifdef UNI_OMP
  #include <omp.h>
#endif

namespace cytnx {

  namespace linalg_internal {

    //====================================================================
    // generic R+R kernel
    template <class T2, class T3>
    __global__ void cufMod_rconst_kernel(double *out, const T2 *ptr, const cytnx_uint64 Nelem,
                                         const T3 val) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          fmod(double(ptr[blockIdx.x * blockDim.x + threadIdx.x]), double(val));
      }
      __syncthreads();
    }

    template <class T2, class T3>
    __global__ void cufMod_lconst_kernel(double *out, const T2 val, const cytnx_uint64 Nelem,
                                         const T3 *ptr) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          fmod(double(val), double(ptr[blockIdx.x * blockDim.x + threadIdx.x]));
      }
      __syncthreads();
    }

    template <class T2, class T3>
    __global__ void cufMod_tn_kernel(double *out, const T2 *val, const cytnx_uint64 Nelem,
                                     const T3 *ptr) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          fmod(double(val[blockIdx.x * blockDim.x + threadIdx.x]),
               double(ptr[blockIdx.x * blockDim.x + threadIdx.x]));
      }
      __syncthreads();
    }

    //====================================================================
    // generic R+R kernel
    template <class T2, class T3>
    __global__ void cufModf_rconst_kernel(float *out, const T2 *ptr, const cytnx_uint64 Nelem,
                                          const T3 val) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          fmodf(ptr[blockIdx.x * blockDim.x + threadIdx.x], val);
      }
      __syncthreads();
    }

    template <class T2, class T3>
    __global__ void cufModf_lconst_kernel(float *out, const T2 val, const cytnx_uint64 Nelem,
                                          const T3 *ptr) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          fmodf(val, ptr[blockIdx.x * blockDim.x + threadIdx.x]);
      }
      __syncthreads();
    }

    template <class T2, class T3>
    __global__ void cufModf_tn_kernel(float *out, const T2 *val, const cytnx_uint64 Nelem,
                                      const T3 *ptr) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] = fmodf(
          val[blockIdx.x * blockDim.x + threadIdx.x], ptr[blockIdx.x * blockDim.x + threadIdx.x]);
      }
      __syncthreads();
    }

    //====================================================================
    // generic R+R kernel
    template <class T1, class T2, class T3>
    __global__ void cuMod_rconst_kernel(T1 *out, const T2 *ptr, const cytnx_uint64 Nelem,
                                        const T3 val) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          ptr[blockIdx.x * blockDim.x + threadIdx.x] % val;
      }
      __syncthreads();
    }

    template <class T1, class T2, class T3>
    __global__ void cuMod_lconst_kernel(T1 *out, const T2 val, const cytnx_uint64 Nelem,
                                        const T3 *ptr) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          val % ptr[blockIdx.x * blockDim.x + threadIdx.x];
      }
      __syncthreads();
    }

    template <class T1, class T2, class T3>
    __global__ void cuMod_tn_kernel(T1 *out, const T2 *val, const cytnx_uint64 Nelem,
                                    const T3 *ptr) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          val[blockIdx.x * blockDim.x + threadIdx.x] % ptr[blockIdx.x * blockDim.x + threadIdx.x];
      }
      __syncthreads();
    }

    //=====================================================================
    void cuMod_internal_cdtcd(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_error_msg(true, "[cuMod] Cannot mod complex numbers%s", "\n");
    }

    void cuMod_internal_cdtcf(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_error_msg(true, "[cuMod] Cannot mod complex numbers%s", "\n");
    }
    void cuMod_internal_cdtd(boost::intrusive_ptr<Storage_base> &out,
                             boost::intrusive_ptr<Storage_base> &Lin,
                             boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                             const std::vector<cytnx_uint64> &shape,
                             const std::vector<cytnx_uint64> &invmapper_L,
                             const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_error_msg(true, "[cuMod] Cannot mod complex numbers%s", "\n");
    }
    void cuMod_internal_cdtf(boost::intrusive_ptr<Storage_base> &out,
                             boost::intrusive_ptr<Storage_base> &Lin,
                             boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                             const std::vector<cytnx_uint64> &shape,
                             const std::vector<cytnx_uint64> &invmapper_L,
                             const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_error_msg(true, "[cuMod] Cannot mod complex numbers%s", "\n");
    }

    void cuMod_internal_cdtu64(boost::intrusive_ptr<Storage_base> &out,
                               boost::intrusive_ptr<Storage_base> &Lin,
                               boost::intrusive_ptr<Storage_base> &Rin,
                               const unsigned long long &len,
                               const std::vector<cytnx_uint64> &shape,
                               const std::vector<cytnx_uint64> &invmapper_L,
                               const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_error_msg(true, "[cuMod] Cannot mod complex numbers%s", "\n");
    }

    void cuMod_internal_cdtu32(boost::intrusive_ptr<Storage_base> &out,
                               boost::intrusive_ptr<Storage_base> &Lin,
                               boost::intrusive_ptr<Storage_base> &Rin,
                               const unsigned long long &len,
                               const std::vector<cytnx_uint64> &shape,
                               const std::vector<cytnx_uint64> &invmapper_L,
                               const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_error_msg(true, "[cuMod] Cannot mod complex numbers%s", "\n");
    }
    void cuMod_internal_cdti64(boost::intrusive_ptr<Storage_base> &out,
                               boost::intrusive_ptr<Storage_base> &Lin,
                               boost::intrusive_ptr<Storage_base> &Rin,
                               const unsigned long long &len,
                               const std::vector<cytnx_uint64> &shape,
                               const std::vector<cytnx_uint64> &invmapper_L,
                               const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_error_msg(true, "[cuMod] Cannot mod complex numbers%s", "\n");
    }

    void cuMod_internal_cdti32(boost::intrusive_ptr<Storage_base> &out,
                               boost::intrusive_ptr<Storage_base> &Lin,
                               boost::intrusive_ptr<Storage_base> &Rin,
                               const unsigned long long &len,
                               const std::vector<cytnx_uint64> &shape,
                               const std::vector<cytnx_uint64> &invmapper_L,
                               const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_error_msg(true, "[cuMod] Cannot mod complex numbers%s", "\n");
    }

    void cuMod_internal_cdti16(boost::intrusive_ptr<Storage_base> &out,
                               boost::intrusive_ptr<Storage_base> &Lin,
                               boost::intrusive_ptr<Storage_base> &Rin,
                               const unsigned long long &len,
                               const std::vector<cytnx_uint64> &shape,
                               const std::vector<cytnx_uint64> &invmapper_L,
                               const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_error_msg(true, "[cuMod] Cannot mod complex numbers%s", "\n");
    }

    void cuMod_internal_cdtu16(boost::intrusive_ptr<Storage_base> &out,
                               boost::intrusive_ptr<Storage_base> &Lin,
                               boost::intrusive_ptr<Storage_base> &Rin,
                               const unsigned long long &len,
                               const std::vector<cytnx_uint64> &shape,
                               const std::vector<cytnx_uint64> &invmapper_L,
                               const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_error_msg(true, "[cuMod] Cannot mod complex numbers%s", "\n");
    }

    void cuMod_internal_cdtb(boost::intrusive_ptr<Storage_base> &out,
                             boost::intrusive_ptr<Storage_base> &Lin,
                             boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                             const std::vector<cytnx_uint64> &shape,
                             const std::vector<cytnx_uint64> &invmapper_L,
                             const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_error_msg(true, "[cuMod] Cannot mod complex numbers%s", "\n");
    }

    //------------------------------------

    void cuMod_internal_cftcd(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_error_msg(true, "[cuMod] Cannot mod complex numbers%s", "\n");
    }

    void cuMod_internal_cftcf(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_error_msg(true, "[cuMod] Cannot mod complex numbers%s", "\n");
    }

    void cuMod_internal_cftd(boost::intrusive_ptr<Storage_base> &out,
                             boost::intrusive_ptr<Storage_base> &Lin,
                             boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                             const std::vector<cytnx_uint64> &shape,
                             const std::vector<cytnx_uint64> &invmapper_L,
                             const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_error_msg(true, "[cuMod] Cannot mod complex numbers%s", "\n");
    }

    void cuMod_internal_cftf(boost::intrusive_ptr<Storage_base> &out,
                             boost::intrusive_ptr<Storage_base> &Lin,
                             boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                             const std::vector<cytnx_uint64> &shape,
                             const std::vector<cytnx_uint64> &invmapper_L,
                             const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_error_msg(true, "[cuMod] Cannot mod complex numbers%s", "\n");
    }

    void cuMod_internal_cftu64(boost::intrusive_ptr<Storage_base> &out,
                               boost::intrusive_ptr<Storage_base> &Lin,
                               boost::intrusive_ptr<Storage_base> &Rin,
                               const unsigned long long &len,
                               const std::vector<cytnx_uint64> &shape,
                               const std::vector<cytnx_uint64> &invmapper_L,
                               const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_error_msg(true, "[cuMod] Cannot mod complex numbers%s", "\n");
    }

    void cuMod_internal_cftu32(boost::intrusive_ptr<Storage_base> &out,
                               boost::intrusive_ptr<Storage_base> &Lin,
                               boost::intrusive_ptr<Storage_base> &Rin,
                               const unsigned long long &len,
                               const std::vector<cytnx_uint64> &shape,
                               const std::vector<cytnx_uint64> &invmapper_L,
                               const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_error_msg(true, "[cuMod] Cannot mod complex numbers%s", "\n");
    }

    void cuMod_internal_cfti64(boost::intrusive_ptr<Storage_base> &out,
                               boost::intrusive_ptr<Storage_base> &Lin,
                               boost::intrusive_ptr<Storage_base> &Rin,
                               const unsigned long long &len,
                               const std::vector<cytnx_uint64> &shape,
                               const std::vector<cytnx_uint64> &invmapper_L,
                               const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_error_msg(true, "[cuMod] Cannot mod complex numbers%s", "\n");
    }

    void cuMod_internal_cfti32(boost::intrusive_ptr<Storage_base> &out,
                               boost::intrusive_ptr<Storage_base> &Lin,
                               boost::intrusive_ptr<Storage_base> &Rin,
                               const unsigned long long &len,
                               const std::vector<cytnx_uint64> &shape,
                               const std::vector<cytnx_uint64> &invmapper_L,
                               const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_error_msg(true, "[cuMod] Cannot mod complex numbers%s", "\n");
    }

    void cuMod_internal_cfti16(boost::intrusive_ptr<Storage_base> &out,
                               boost::intrusive_ptr<Storage_base> &Lin,
                               boost::intrusive_ptr<Storage_base> &Rin,
                               const unsigned long long &len,
                               const std::vector<cytnx_uint64> &shape,
                               const std::vector<cytnx_uint64> &invmapper_L,
                               const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_error_msg(true, "[cuMod] Cannot mod complex numbers%s", "\n");
    }

    void cuMod_internal_cftu16(boost::intrusive_ptr<Storage_base> &out,
                               boost::intrusive_ptr<Storage_base> &Lin,
                               boost::intrusive_ptr<Storage_base> &Rin,
                               const unsigned long long &len,
                               const std::vector<cytnx_uint64> &shape,
                               const std::vector<cytnx_uint64> &invmapper_L,
                               const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_error_msg(true, "[cuMod] Cannot mod complex numbers%s", "\n");
    }

    void cuMod_internal_cftb(boost::intrusive_ptr<Storage_base> &out,
                             boost::intrusive_ptr<Storage_base> &Lin,
                             boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                             const std::vector<cytnx_uint64> &shape,
                             const std::vector<cytnx_uint64> &invmapper_L,
                             const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_error_msg(true, "[cuMod] Cannot mod complex numbers%s", "\n");
    }

    //------------------------------

    void cuMod_internal_dtcd(boost::intrusive_ptr<Storage_base> &out,
                             boost::intrusive_ptr<Storage_base> &Lin,
                             boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                             const std::vector<cytnx_uint64> &shape,
                             const std::vector<cytnx_uint64> &invmapper_L,
                             const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_error_msg(true, "[cuMod] Cannot mod complex numbers%s", "\n");
    }

    void cuMod_internal_dtcf(boost::intrusive_ptr<Storage_base> &out,
                             boost::intrusive_ptr<Storage_base> &Lin,
                             boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                             const std::vector<cytnx_uint64> &shape,
                             const std::vector<cytnx_uint64> &invmapper_L,
                             const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_error_msg(true, "[cuMod] Cannot mod complex numbers%s", "\n");
    }

    void cuMod_internal_dtd(boost::intrusive_ptr<Storage_base> &out,
                            boost::intrusive_ptr<Storage_base> &Lin,
                            boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                            const std::vector<cytnx_uint64> &shape,
                            const std::vector<cytnx_uint64> &invmapper_L,
                            const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_double *_out = (cytnx_double *)out->Mem;
      cytnx_double *_Lin = (cytnx_double *)Lin->Mem;
      cytnx_double *_Rin = (cytnx_double *)Rin->Mem;

      cytnx_uint32 NBlocks = len / 512;
      if (len % 512) NBlocks += 1;

      if (Lin->size() == 1) {
        cufMod_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cufMod_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cufMod_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }
    void cuMod_internal_dtf(boost::intrusive_ptr<Storage_base> &out,
                            boost::intrusive_ptr<Storage_base> &Lin,
                            boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                            const std::vector<cytnx_uint64> &shape,
                            const std::vector<cytnx_uint64> &invmapper_L,
                            const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_double *_out = (cytnx_double *)out->Mem;
      cytnx_double *_Lin = (cytnx_double *)Lin->Mem;
      cytnx_float *_Rin = (cytnx_float *)Rin->Mem;

      cytnx_uint32 NBlocks = len / 512;
      if (len % 512) NBlocks += 1;

      if (Lin->size() == 1) {
        cufMod_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cufMod_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cufMod_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }
    void cuMod_internal_dtu64(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_double *_out = (cytnx_double *)out->Mem;
      cytnx_double *_Lin = (cytnx_double *)Lin->Mem;
      cytnx_uint64 *_Rin = (cytnx_uint64 *)Rin->Mem;

      cytnx_uint32 NBlocks = len / 512;
      if (len % 512) NBlocks += 1;

      if (Lin->size() == 1) {
        cufMod_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cufMod_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cufMod_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }
    void cuMod_internal_dtu32(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_double *_out = (cytnx_double *)out->Mem;
      cytnx_double *_Lin = (cytnx_double *)Lin->Mem;
      cytnx_uint32 *_Rin = (cytnx_uint32 *)Rin->Mem;

      cytnx_uint32 NBlocks = len / 512;
      if (len % 512) NBlocks += 1;

      if (Lin->size() == 1) {
        cufMod_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cufMod_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cufMod_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }
    void cuMod_internal_dti64(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_double *_out = (cytnx_double *)out->Mem;
      cytnx_double *_Lin = (cytnx_double *)Lin->Mem;
      cytnx_int64 *_Rin = (cytnx_int64 *)Rin->Mem;

      cytnx_uint32 NBlocks = len / 512;
      if (len % 512) NBlocks += 1;

      if (Lin->size() == 1) {
        cufMod_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cufMod_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cufMod_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }
    void cuMod_internal_dti32(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_double *_out = (cytnx_double *)out->Mem;
      cytnx_double *_Lin = (cytnx_double *)Lin->Mem;
      cytnx_int32 *_Rin = (cytnx_int32 *)Rin->Mem;

      cytnx_uint32 NBlocks = len / 512;
      if (len % 512) NBlocks += 1;

      if (Lin->size() == 1) {
        cufMod_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cufMod_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cufMod_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }
    void cuMod_internal_dti16(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_double *_out = (cytnx_double *)out->Mem;
      cytnx_double *_Lin = (cytnx_double *)Lin->Mem;
      cytnx_int16 *_Rin = (cytnx_int16 *)Rin->Mem;

      cytnx_uint32 NBlocks = len / 512;
      if (len % 512) NBlocks += 1;

      if (Lin->size() == 1) {
        cufMod_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cufMod_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cufMod_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }
    void cuMod_internal_dtu16(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_double *_out = (cytnx_double *)out->Mem;
      cytnx_double *_Lin = (cytnx_double *)Lin->Mem;
      cytnx_uint16 *_Rin = (cytnx_uint16 *)Rin->Mem;

      cytnx_uint32 NBlocks = len / 512;
      if (len % 512) NBlocks += 1;

      if (Lin->size() == 1) {
        cufMod_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cufMod_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cufMod_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }

    void cuMod_internal_dtb(boost::intrusive_ptr<Storage_base> &out,
                            boost::intrusive_ptr<Storage_base> &Lin,
                            boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                            const std::vector<cytnx_uint64> &shape,
                            const std::vector<cytnx_uint64> &invmapper_L,
                            const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_double *_out = (cytnx_double *)out->Mem;
      cytnx_double *_Lin = (cytnx_double *)Lin->Mem;
      cytnx_bool *_Rin = (cytnx_bool *)Rin->Mem;

      cytnx_uint32 NBlocks = len / 512;
      if (len % 512) NBlocks += 1;

      if (Lin->size() == 1) {
        cufMod_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cufMod_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, cytnx_double(_Rin[0]));
      } else {
        cufMod_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }

    //-----------------------------------

    void cuMod_internal_ftcd(boost::intrusive_ptr<Storage_base> &out,
                             boost::intrusive_ptr<Storage_base> &Lin,
                             boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                             const std::vector<cytnx_uint64> &shape,
                             const std::vector<cytnx_uint64> &invmapper_L,
                             const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_error_msg(true, "[cuMod] Cannot mod complex numbers%s", "\n");
    }

    void cuMod_internal_ftcf(boost::intrusive_ptr<Storage_base> &out,
                             boost::intrusive_ptr<Storage_base> &Lin,
                             boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                             const std::vector<cytnx_uint64> &shape,
                             const std::vector<cytnx_uint64> &invmapper_L,
                             const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_error_msg(true, "[cuMod] Cannot mod complex numbers%s", "\n");
    }

    void cuMod_internal_ftd(boost::intrusive_ptr<Storage_base> &out,
                            boost::intrusive_ptr<Storage_base> &Lin,
                            boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                            const std::vector<cytnx_uint64> &shape,
                            const std::vector<cytnx_uint64> &invmapper_L,
                            const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_double *_out = (cytnx_double *)out->Mem;
      cytnx_float *_Lin = (cytnx_float *)Lin->Mem;
      cytnx_double *_Rin = (cytnx_double *)Rin->Mem;

      cytnx_uint32 NBlocks = len / 512;
      if (len % 512) NBlocks += 1;

      if (Lin->size() == 1) {
        cufMod_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cufMod_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cufMod_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }
    void cuMod_internal_ftf(boost::intrusive_ptr<Storage_base> &out,
                            boost::intrusive_ptr<Storage_base> &Lin,
                            boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                            const std::vector<cytnx_uint64> &shape,
                            const std::vector<cytnx_uint64> &invmapper_L,
                            const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_float *_out = (cytnx_float *)out->Mem;
      cytnx_float *_Lin = (cytnx_float *)Lin->Mem;
      cytnx_float *_Rin = (cytnx_float *)Rin->Mem;

      cytnx_uint32 NBlocks = len / 512;
      if (len % 512) NBlocks += 1;

      if (Lin->size() == 1) {
        cufModf_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cufModf_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cufModf_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }
    void cuMod_internal_ftu64(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_float *_out = (cytnx_float *)out->Mem;
      cytnx_float *_Lin = (cytnx_float *)Lin->Mem;
      cytnx_uint64 *_Rin = (cytnx_uint64 *)Rin->Mem;

      cytnx_uint32 NBlocks = len / 512;
      if (len % 512) NBlocks += 1;

      if (Lin->size() == 1) {
        cufModf_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cufModf_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cufModf_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }
    void cuMod_internal_ftu32(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_float *_out = (cytnx_float *)out->Mem;
      cytnx_float *_Lin = (cytnx_float *)Lin->Mem;
      cytnx_uint32 *_Rin = (cytnx_uint32 *)Rin->Mem;

      cytnx_uint32 NBlocks = len / 512;
      if (len % 512) NBlocks += 1;

      if (Lin->size() == 1) {
        cufModf_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cufModf_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cufModf_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }
    void cuMod_internal_fti64(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_float *_out = (cytnx_float *)out->Mem;
      cytnx_float *_Lin = (cytnx_float *)Lin->Mem;
      cytnx_int64 *_Rin = (cytnx_int64 *)Rin->Mem;

      cytnx_uint32 NBlocks = len / 512;
      if (len % 512) NBlocks += 1;

      if (Lin->size() == 1) {
        cufModf_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cufModf_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cufModf_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }
    void cuMod_internal_fti32(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_float *_out = (cytnx_float *)out->Mem;
      cytnx_float *_Lin = (cytnx_float *)Lin->Mem;
      cytnx_int32 *_Rin = (cytnx_int32 *)Rin->Mem;

      cytnx_uint32 NBlocks = len / 512;
      if (len % 512) NBlocks += 1;

      if (Lin->size() == 1) {
        cufModf_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cufModf_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cufModf_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }
    void cuMod_internal_fti16(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_float *_out = (cytnx_float *)out->Mem;
      cytnx_float *_Lin = (cytnx_float *)Lin->Mem;
      cytnx_int16 *_Rin = (cytnx_int16 *)Rin->Mem;

      cytnx_uint32 NBlocks = len / 512;
      if (len % 512) NBlocks += 1;

      if (Lin->size() == 1) {
        cufModf_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cufModf_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cufModf_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }
    void cuMod_internal_ftu16(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_float *_out = (cytnx_float *)out->Mem;
      cytnx_float *_Lin = (cytnx_float *)Lin->Mem;
      cytnx_uint16 *_Rin = (cytnx_uint16 *)Rin->Mem;

      cytnx_uint32 NBlocks = len / 512;
      if (len % 512) NBlocks += 1;

      if (Lin->size() == 1) {
        cufModf_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cufModf_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cufModf_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }

    void cuMod_internal_ftb(boost::intrusive_ptr<Storage_base> &out,
                            boost::intrusive_ptr<Storage_base> &Lin,
                            boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                            const std::vector<cytnx_uint64> &shape,
                            const std::vector<cytnx_uint64> &invmapper_L,
                            const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_float *_out = (cytnx_float *)out->Mem;
      cytnx_float *_Lin = (cytnx_float *)Lin->Mem;
      cytnx_bool *_Rin = (cytnx_bool *)Rin->Mem;

      cytnx_uint32 NBlocks = len / 512;
      if (len % 512) NBlocks += 1;

      if (Lin->size() == 1) {
        cufModf_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cufModf_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, cytnx_float(_Rin[0]));
      } else {
        cufModf_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }
    // taag
    //----------------------------

    void cuMod_internal_i64tcd(boost::intrusive_ptr<Storage_base> &out,
                               boost::intrusive_ptr<Storage_base> &Lin,
                               boost::intrusive_ptr<Storage_base> &Rin,
                               const unsigned long long &len,
                               const std::vector<cytnx_uint64> &shape,
                               const std::vector<cytnx_uint64> &invmapper_L,
                               const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_error_msg(true, "[cuMod] Cannot mod complex numbers%s", "\n");
    }

    void cuMod_internal_i64tcf(boost::intrusive_ptr<Storage_base> &out,
                               boost::intrusive_ptr<Storage_base> &Lin,
                               boost::intrusive_ptr<Storage_base> &Rin,
                               const unsigned long long &len,
                               const std::vector<cytnx_uint64> &shape,
                               const std::vector<cytnx_uint64> &invmapper_L,
                               const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_error_msg(true, "[cuMod] Cannot mod complex numbers%s", "\n");
    }

    void cuMod_internal_i64td(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_double *_out = (cytnx_double *)out->Mem;
      cytnx_int64 *_Lin = (cytnx_int64 *)Lin->Mem;
      cytnx_double *_Rin = (cytnx_double *)Rin->Mem;

      cytnx_uint32 NBlocks = len / 512;
      if (len % 512) NBlocks += 1;

      if (Lin->size() == 1) {
        cufMod_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cufMod_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cufMod_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }
    void cuMod_internal_i64tf(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_float *_out = (cytnx_float *)out->Mem;
      cytnx_int64 *_Lin = (cytnx_int64 *)Lin->Mem;
      cytnx_float *_Rin = (cytnx_float *)Rin->Mem;

      cytnx_uint32 NBlocks = len / 512;
      if (len % 512) NBlocks += 1;

      if (Lin->size() == 1) {
        cufModf_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cufModf_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cufModf_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }
    void cuMod_internal_i64ti64(boost::intrusive_ptr<Storage_base> &out,
                                boost::intrusive_ptr<Storage_base> &Lin,
                                boost::intrusive_ptr<Storage_base> &Rin,
                                const unsigned long long &len,
                                const std::vector<cytnx_uint64> &shape,
                                const std::vector<cytnx_uint64> &invmapper_L,
                                const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_int64 *_out = (cytnx_int64 *)out->Mem;
      cytnx_int64 *_Lin = (cytnx_int64 *)Lin->Mem;
      cytnx_int64 *_Rin = (cytnx_int64 *)Rin->Mem;

      cytnx_uint32 NBlocks = len / 512;
      if (len % 512) NBlocks += 1;

      if (Lin->size() == 1) {
        cuMod_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMod_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cuMod_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }
    void cuMod_internal_i64tu64(boost::intrusive_ptr<Storage_base> &out,
                                boost::intrusive_ptr<Storage_base> &Lin,
                                boost::intrusive_ptr<Storage_base> &Rin,
                                const unsigned long long &len,
                                const std::vector<cytnx_uint64> &shape,
                                const std::vector<cytnx_uint64> &invmapper_L,
                                const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_int64 *_out = (cytnx_int64 *)out->Mem;
      cytnx_int64 *_Lin = (cytnx_int64 *)Lin->Mem;
      cytnx_uint64 *_Rin = (cytnx_uint64 *)Rin->Mem;

      cytnx_uint32 NBlocks = len / 512;
      if (len % 512) NBlocks += 1;

      if (Lin->size() == 1) {
        cuMod_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMod_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cuMod_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }
    void cuMod_internal_i64ti32(boost::intrusive_ptr<Storage_base> &out,
                                boost::intrusive_ptr<Storage_base> &Lin,
                                boost::intrusive_ptr<Storage_base> &Rin,
                                const unsigned long long &len,
                                const std::vector<cytnx_uint64> &shape,
                                const std::vector<cytnx_uint64> &invmapper_L,
                                const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_int64 *_out = (cytnx_int64 *)out->Mem;
      cytnx_int64 *_Lin = (cytnx_int64 *)Lin->Mem;
      cytnx_int32 *_Rin = (cytnx_int32 *)Rin->Mem;

      cytnx_uint32 NBlocks = len / 512;
      if (len % 512) NBlocks += 1;

      if (Lin->size() == 1) {
        cuMod_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMod_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cuMod_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }
    void cuMod_internal_i64tu32(boost::intrusive_ptr<Storage_base> &out,
                                boost::intrusive_ptr<Storage_base> &Lin,
                                boost::intrusive_ptr<Storage_base> &Rin,
                                const unsigned long long &len,
                                const std::vector<cytnx_uint64> &shape,
                                const std::vector<cytnx_uint64> &invmapper_L,
                                const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_int64 *_out = (cytnx_int64 *)out->Mem;
      cytnx_int64 *_Lin = (cytnx_int64 *)Lin->Mem;
      cytnx_uint32 *_Rin = (cytnx_uint32 *)Rin->Mem;

      cytnx_uint32 NBlocks = len / 512;
      if (len % 512) NBlocks += 1;

      if (Lin->size() == 1) {
        cuMod_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMod_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cuMod_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }
    void cuMod_internal_i64ti16(boost::intrusive_ptr<Storage_base> &out,
                                boost::intrusive_ptr<Storage_base> &Lin,
                                boost::intrusive_ptr<Storage_base> &Rin,
                                const unsigned long long &len,
                                const std::vector<cytnx_uint64> &shape,
                                const std::vector<cytnx_uint64> &invmapper_L,
                                const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_int64 *_out = (cytnx_int64 *)out->Mem;
      cytnx_int64 *_Lin = (cytnx_int64 *)Lin->Mem;
      cytnx_int16 *_Rin = (cytnx_int16 *)Rin->Mem;

      cytnx_uint32 NBlocks = len / 512;
      if (len % 512) NBlocks += 1;

      if (Lin->size() == 1) {
        cuMod_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMod_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cuMod_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }
    void cuMod_internal_i64tu16(boost::intrusive_ptr<Storage_base> &out,
                                boost::intrusive_ptr<Storage_base> &Lin,
                                boost::intrusive_ptr<Storage_base> &Rin,
                                const unsigned long long &len,
                                const std::vector<cytnx_uint64> &shape,
                                const std::vector<cytnx_uint64> &invmapper_L,
                                const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_int64 *_out = (cytnx_int64 *)out->Mem;
      cytnx_int64 *_Lin = (cytnx_int64 *)Lin->Mem;
      cytnx_uint16 *_Rin = (cytnx_uint16 *)Rin->Mem;

      cytnx_uint32 NBlocks = len / 512;
      if (len % 512) NBlocks += 1;

      if (Lin->size() == 1) {
        cuMod_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMod_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cuMod_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }

    __global__ void cuMod_lconst_kernel(cytnx_int64 *out, const cytnx_int64 val,
                                        const cytnx_uint64 Nelem, const cytnx_bool *ptr) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          val % cytnx_int64(ptr[blockIdx.x * blockDim.x + threadIdx.x]);
      }
      __syncthreads();
    }

    __global__ void cuMod_tn_kernel(cytnx_int64 *out, const cytnx_int64 *val,
                                    const cytnx_uint64 Nelem, const cytnx_bool *ptr) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          val[blockIdx.x * blockDim.x + threadIdx.x] %
          cytnx_int64(ptr[blockIdx.x * blockDim.x + threadIdx.x]);
      }
      __syncthreads();
    }
    void cuMod_internal_i64tb(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_int64 *_out = (cytnx_int64 *)out->Mem;
      cytnx_int64 *_Lin = (cytnx_int64 *)Lin->Mem;
      cytnx_bool *_Rin = (cytnx_bool *)Rin->Mem;

      cytnx_uint32 NBlocks = len / 512;
      if (len % 512) NBlocks += 1;

      if (Lin->size() == 1) {
        cuMod_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMod_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, cytnx_int64(_Rin[0]));
      } else {
        cuMod_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }

    //-----------------------------------------

    void cuMod_internal_u64tcd(boost::intrusive_ptr<Storage_base> &out,
                               boost::intrusive_ptr<Storage_base> &Lin,
                               boost::intrusive_ptr<Storage_base> &Rin,
                               const unsigned long long &len,
                               const std::vector<cytnx_uint64> &shape,
                               const std::vector<cytnx_uint64> &invmapper_L,
                               const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_error_msg(true, "[cuMod] Cannot mod complex numbers%s", "\n");
    }

    void cuMod_internal_u64tcf(boost::intrusive_ptr<Storage_base> &out,
                               boost::intrusive_ptr<Storage_base> &Lin,
                               boost::intrusive_ptr<Storage_base> &Rin,
                               const unsigned long long &len,
                               const std::vector<cytnx_uint64> &shape,
                               const std::vector<cytnx_uint64> &invmapper_L,
                               const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_error_msg(true, "[cuMod] Cannot mod complex numbers%s", "\n");
    }
    void cuMod_internal_u64td(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_double *_out = (cytnx_double *)out->Mem;
      cytnx_uint64 *_Lin = (cytnx_uint64 *)Lin->Mem;
      cytnx_double *_Rin = (cytnx_double *)Rin->Mem;

      cytnx_uint32 NBlocks = len / 512;
      if (len % 512) NBlocks += 1;

      if (Lin->size() == 1) {
        cufMod_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cufMod_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cufMod_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }
    void cuMod_internal_u64tf(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_float *_out = (cytnx_float *)out->Mem;
      cytnx_uint64 *_Lin = (cytnx_uint64 *)Lin->Mem;
      cytnx_float *_Rin = (cytnx_float *)Rin->Mem;

      cytnx_uint32 NBlocks = len / 512;
      if (len % 512) NBlocks += 1;

      if (Lin->size() == 1) {
        cufModf_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cufModf_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cufModf_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }
    void cuMod_internal_u64ti64(boost::intrusive_ptr<Storage_base> &out,
                                boost::intrusive_ptr<Storage_base> &Lin,
                                boost::intrusive_ptr<Storage_base> &Rin,
                                const unsigned long long &len,
                                const std::vector<cytnx_uint64> &shape,
                                const std::vector<cytnx_uint64> &invmapper_L,
                                const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_int64 *_out = (cytnx_int64 *)out->Mem;
      cytnx_uint64 *_Lin = (cytnx_uint64 *)Lin->Mem;
      cytnx_int64 *_Rin = (cytnx_int64 *)Rin->Mem;

      cytnx_uint32 NBlocks = len / 512;
      if (len % 512) NBlocks += 1;

      if (Lin->size() == 1) {
        cuMod_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMod_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cuMod_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }
    void cuMod_internal_u64tu64(boost::intrusive_ptr<Storage_base> &out,
                                boost::intrusive_ptr<Storage_base> &Lin,
                                boost::intrusive_ptr<Storage_base> &Rin,
                                const unsigned long long &len,
                                const std::vector<cytnx_uint64> &shape,
                                const std::vector<cytnx_uint64> &invmapper_L,
                                const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_uint64 *_out = (cytnx_uint64 *)out->Mem;
      cytnx_uint64 *_Lin = (cytnx_uint64 *)Lin->Mem;
      cytnx_uint64 *_Rin = (cytnx_uint64 *)Rin->Mem;

      cytnx_uint32 NBlocks = len / 512;
      if (len % 512) NBlocks += 1;

      if (Lin->size() == 1) {
        cuMod_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMod_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cuMod_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }
    void cuMod_internal_u64ti32(boost::intrusive_ptr<Storage_base> &out,
                                boost::intrusive_ptr<Storage_base> &Lin,
                                boost::intrusive_ptr<Storage_base> &Rin,
                                const unsigned long long &len,
                                const std::vector<cytnx_uint64> &shape,
                                const std::vector<cytnx_uint64> &invmapper_L,
                                const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_uint64 *_out = (cytnx_uint64 *)out->Mem;
      cytnx_uint64 *_Lin = (cytnx_uint64 *)Lin->Mem;
      cytnx_int32 *_Rin = (cytnx_int32 *)Rin->Mem;

      cytnx_uint32 NBlocks = len / 512;
      if (len % 512) NBlocks += 1;

      if (Lin->size() == 1) {
        cuMod_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMod_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cuMod_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }
    void cuMod_internal_u64tu32(boost::intrusive_ptr<Storage_base> &out,
                                boost::intrusive_ptr<Storage_base> &Lin,
                                boost::intrusive_ptr<Storage_base> &Rin,
                                const unsigned long long &len,
                                const std::vector<cytnx_uint64> &shape,
                                const std::vector<cytnx_uint64> &invmapper_L,
                                const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_uint64 *_out = (cytnx_uint64 *)out->Mem;
      cytnx_uint64 *_Lin = (cytnx_uint64 *)Lin->Mem;
      cytnx_uint32 *_Rin = (cytnx_uint32 *)Rin->Mem;

      cytnx_uint32 NBlocks = len / 512;
      if (len % 512) NBlocks += 1;

      if (Lin->size() == 1) {
        cuMod_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMod_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cuMod_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }
    void cuMod_internal_u64ti16(boost::intrusive_ptr<Storage_base> &out,
                                boost::intrusive_ptr<Storage_base> &Lin,
                                boost::intrusive_ptr<Storage_base> &Rin,
                                const unsigned long long &len,
                                const std::vector<cytnx_uint64> &shape,
                                const std::vector<cytnx_uint64> &invmapper_L,
                                const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_uint64 *_out = (cytnx_uint64 *)out->Mem;
      cytnx_uint64 *_Lin = (cytnx_uint64 *)Lin->Mem;
      cytnx_int16 *_Rin = (cytnx_int16 *)Rin->Mem;

      cytnx_uint32 NBlocks = len / 512;
      if (len % 512) NBlocks += 1;

      if (Lin->size() == 1) {
        cuMod_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMod_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cuMod_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }
    void cuMod_internal_u64tu16(boost::intrusive_ptr<Storage_base> &out,
                                boost::intrusive_ptr<Storage_base> &Lin,
                                boost::intrusive_ptr<Storage_base> &Rin,
                                const unsigned long long &len,
                                const std::vector<cytnx_uint64> &shape,
                                const std::vector<cytnx_uint64> &invmapper_L,
                                const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_uint64 *_out = (cytnx_uint64 *)out->Mem;
      cytnx_uint64 *_Lin = (cytnx_uint64 *)Lin->Mem;
      cytnx_uint16 *_Rin = (cytnx_uint16 *)Rin->Mem;

      cytnx_uint32 NBlocks = len / 512;
      if (len % 512) NBlocks += 1;

      if (Lin->size() == 1) {
        cuMod_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMod_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cuMod_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }

    __global__ void cuMod_lconst_kernel(cytnx_uint64 *out, const cytnx_uint64 val,
                                        const cytnx_uint64 Nelem, const cytnx_bool *ptr) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          val % cytnx_uint64(ptr[blockIdx.x * blockDim.x + threadIdx.x]);
      }
      __syncthreads();
    }

    __global__ void cuMod_tn_kernel(cytnx_uint64 *out, const cytnx_uint64 *val,
                                    const cytnx_uint64 Nelem, const cytnx_bool *ptr) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          val[blockIdx.x * blockDim.x + threadIdx.x] %
          cytnx_uint64(ptr[blockIdx.x * blockDim.x + threadIdx.x]);
      }
      __syncthreads();
    }
    void cuMod_internal_u64tb(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_uint64 *_out = (cytnx_uint64 *)out->Mem;
      cytnx_uint64 *_Lin = (cytnx_uint64 *)Lin->Mem;
      cytnx_bool *_Rin = (cytnx_bool *)Rin->Mem;

      cytnx_uint32 NBlocks = len / 512;
      if (len % 512) NBlocks += 1;

      if (Lin->size() == 1) {
        cuMod_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMod_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, cytnx_uint64(_Rin[0]));
      } else {
        cuMod_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }

    //-----------------------------------------

    void cuMod_internal_i32tcd(boost::intrusive_ptr<Storage_base> &out,
                               boost::intrusive_ptr<Storage_base> &Lin,
                               boost::intrusive_ptr<Storage_base> &Rin,
                               const unsigned long long &len,
                               const std::vector<cytnx_uint64> &shape,
                               const std::vector<cytnx_uint64> &invmapper_L,
                               const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_error_msg(true, "[cuMod] Cannot mod complex numbers%s", "\n");
    }

    void cuMod_internal_i32tcf(boost::intrusive_ptr<Storage_base> &out,
                               boost::intrusive_ptr<Storage_base> &Lin,
                               boost::intrusive_ptr<Storage_base> &Rin,
                               const unsigned long long &len,
                               const std::vector<cytnx_uint64> &shape,
                               const std::vector<cytnx_uint64> &invmapper_L,
                               const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_error_msg(true, "[cuMod] Cannot mod complex numbers%s", "\n");
    }
    void cuMod_internal_i32td(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_double *_out = (cytnx_double *)out->Mem;
      cytnx_int32 *_Lin = (cytnx_int32 *)Lin->Mem;
      cytnx_double *_Rin = (cytnx_double *)Rin->Mem;

      cytnx_uint32 NBlocks = len / 512;
      if (len % 512) NBlocks += 1;

      if (Lin->size() == 1) {
        cufMod_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cufMod_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cufMod_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }
    void cuMod_internal_i32tf(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_float *_out = (cytnx_float *)out->Mem;
      cytnx_int32 *_Lin = (cytnx_int32 *)Lin->Mem;
      cytnx_float *_Rin = (cytnx_float *)Rin->Mem;

      cytnx_uint32 NBlocks = len / 512;
      if (len % 512) NBlocks += 1;

      if (Lin->size() == 1) {
        cufModf_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cufModf_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cufModf_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }
    void cuMod_internal_i32ti64(boost::intrusive_ptr<Storage_base> &out,
                                boost::intrusive_ptr<Storage_base> &Lin,
                                boost::intrusive_ptr<Storage_base> &Rin,
                                const unsigned long long &len,
                                const std::vector<cytnx_uint64> &shape,
                                const std::vector<cytnx_uint64> &invmapper_L,
                                const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_int64 *_out = (cytnx_int64 *)out->Mem;
      cytnx_int32 *_Lin = (cytnx_int32 *)Lin->Mem;
      cytnx_int64 *_Rin = (cytnx_int64 *)Rin->Mem;

      cytnx_uint32 NBlocks = len / 512;
      if (len % 512) NBlocks += 1;

      if (Lin->size() == 1) {
        cuMod_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMod_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cuMod_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }
    void cuMod_internal_i32tu64(boost::intrusive_ptr<Storage_base> &out,
                                boost::intrusive_ptr<Storage_base> &Lin,
                                boost::intrusive_ptr<Storage_base> &Rin,
                                const unsigned long long &len,
                                const std::vector<cytnx_uint64> &shape,
                                const std::vector<cytnx_uint64> &invmapper_L,
                                const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_uint64 *_out = (cytnx_uint64 *)out->Mem;
      cytnx_int32 *_Lin = (cytnx_int32 *)Lin->Mem;
      cytnx_uint64 *_Rin = (cytnx_uint64 *)Rin->Mem;

      cytnx_uint32 NBlocks = len / 512;
      if (len % 512) NBlocks += 1;

      if (Lin->size() == 1) {
        cuMod_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMod_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cuMod_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }
    void cuMod_internal_i32ti32(boost::intrusive_ptr<Storage_base> &out,
                                boost::intrusive_ptr<Storage_base> &Lin,
                                boost::intrusive_ptr<Storage_base> &Rin,
                                const unsigned long long &len,
                                const std::vector<cytnx_uint64> &shape,
                                const std::vector<cytnx_uint64> &invmapper_L,
                                const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_int32 *_out = (cytnx_int32 *)out->Mem;
      cytnx_int32 *_Lin = (cytnx_int32 *)Lin->Mem;
      cytnx_int32 *_Rin = (cytnx_int32 *)Rin->Mem;

      cytnx_uint32 NBlocks = len / 512;
      if (len % 512) NBlocks += 1;

      if (Lin->size() == 1) {
        cuMod_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMod_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cuMod_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }
    void cuMod_internal_i32tu32(boost::intrusive_ptr<Storage_base> &out,
                                boost::intrusive_ptr<Storage_base> &Lin,
                                boost::intrusive_ptr<Storage_base> &Rin,
                                const unsigned long long &len,
                                const std::vector<cytnx_uint64> &shape,
                                const std::vector<cytnx_uint64> &invmapper_L,
                                const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_int32 *_out = (cytnx_int32 *)out->Mem;
      cytnx_int32 *_Lin = (cytnx_int32 *)Lin->Mem;
      cytnx_uint32 *_Rin = (cytnx_uint32 *)Rin->Mem;

      cytnx_uint32 NBlocks = len / 512;
      if (len % 512) NBlocks += 1;

      if (Lin->size() == 1) {
        cuMod_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMod_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cuMod_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }
    void cuMod_internal_i32ti16(boost::intrusive_ptr<Storage_base> &out,
                                boost::intrusive_ptr<Storage_base> &Lin,
                                boost::intrusive_ptr<Storage_base> &Rin,
                                const unsigned long long &len,
                                const std::vector<cytnx_uint64> &shape,
                                const std::vector<cytnx_uint64> &invmapper_L,
                                const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_int32 *_out = (cytnx_int32 *)out->Mem;
      cytnx_int32 *_Lin = (cytnx_int32 *)Lin->Mem;
      cytnx_int16 *_Rin = (cytnx_int16 *)Rin->Mem;

      cytnx_uint32 NBlocks = len / 512;
      if (len % 512) NBlocks += 1;

      if (Lin->size() == 1) {
        cuMod_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMod_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cuMod_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }
    void cuMod_internal_i32tu16(boost::intrusive_ptr<Storage_base> &out,
                                boost::intrusive_ptr<Storage_base> &Lin,
                                boost::intrusive_ptr<Storage_base> &Rin,
                                const unsigned long long &len,
                                const std::vector<cytnx_uint64> &shape,
                                const std::vector<cytnx_uint64> &invmapper_L,
                                const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_int32 *_out = (cytnx_int32 *)out->Mem;
      cytnx_int32 *_Lin = (cytnx_int32 *)Lin->Mem;
      cytnx_uint16 *_Rin = (cytnx_uint16 *)Rin->Mem;

      cytnx_uint32 NBlocks = len / 512;
      if (len % 512) NBlocks += 1;

      if (Lin->size() == 1) {
        cuMod_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMod_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cuMod_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }

    __global__ void cuMod_lconst_kernel(cytnx_int32 *out, const cytnx_int32 val,
                                        const cytnx_uint64 Nelem, const cytnx_bool *ptr) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          val % cytnx_int32(ptr[blockIdx.x * blockDim.x + threadIdx.x]);
      }
      __syncthreads();
    }

    __global__ void cuMod_tn_kernel(cytnx_int32 *out, const cytnx_int32 *val,
                                    const cytnx_uint64 Nelem, const cytnx_bool *ptr) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          val[blockIdx.x * blockDim.x + threadIdx.x] %
          cytnx_int32(ptr[blockIdx.x * blockDim.x + threadIdx.x]);
      }
      __syncthreads();
    }
    void cuMod_internal_i32tb(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_int32 *_out = (cytnx_int32 *)out->Mem;
      cytnx_int32 *_Lin = (cytnx_int32 *)Lin->Mem;
      cytnx_bool *_Rin = (cytnx_bool *)Rin->Mem;

      cytnx_uint32 NBlocks = len / 512;
      if (len % 512) NBlocks += 1;

      if (Lin->size() == 1) {
        cuMod_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMod_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, cytnx_int32(_Rin[0]));
      } else {
        cuMod_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }

    //---------------------------------------

    void cuMod_internal_u32tcd(boost::intrusive_ptr<Storage_base> &out,
                               boost::intrusive_ptr<Storage_base> &Lin,
                               boost::intrusive_ptr<Storage_base> &Rin,
                               const unsigned long long &len,
                               const std::vector<cytnx_uint64> &shape,
                               const std::vector<cytnx_uint64> &invmapper_L,
                               const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_error_msg(true, "[cuMod] Cannot mod complex numbers%s", "\n");
    }

    void cuMod_internal_u32tcf(boost::intrusive_ptr<Storage_base> &out,
                               boost::intrusive_ptr<Storage_base> &Lin,
                               boost::intrusive_ptr<Storage_base> &Rin,
                               const unsigned long long &len,
                               const std::vector<cytnx_uint64> &shape,
                               const std::vector<cytnx_uint64> &invmapper_L,
                               const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_error_msg(true, "[cuMod] Cannot mod complex numbers%s", "\n");
    }
    void cuMod_internal_u32td(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_double *_out = (cytnx_double *)out->Mem;
      cytnx_uint32 *_Lin = (cytnx_uint32 *)Lin->Mem;
      cytnx_double *_Rin = (cytnx_double *)Rin->Mem;

      cytnx_uint32 NBlocks = len / 512;
      if (len % 512) NBlocks += 1;

      if (Lin->size() == 1) {
        cufMod_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cufMod_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cufMod_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }
    void cuMod_internal_u32tf(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_float *_out = (cytnx_float *)out->Mem;
      cytnx_uint32 *_Lin = (cytnx_uint32 *)Lin->Mem;
      cytnx_float *_Rin = (cytnx_float *)Rin->Mem;

      cytnx_uint32 NBlocks = len / 512;
      if (len % 512) NBlocks += 1;

      if (Lin->size() == 1) {
        cufModf_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cufModf_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cufModf_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }
    void cuMod_internal_u32ti64(boost::intrusive_ptr<Storage_base> &out,
                                boost::intrusive_ptr<Storage_base> &Lin,
                                boost::intrusive_ptr<Storage_base> &Rin,
                                const unsigned long long &len,
                                const std::vector<cytnx_uint64> &shape,
                                const std::vector<cytnx_uint64> &invmapper_L,
                                const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_int64 *_out = (cytnx_int64 *)out->Mem;
      cytnx_uint32 *_Lin = (cytnx_uint32 *)Lin->Mem;
      cytnx_int64 *_Rin = (cytnx_int64 *)Rin->Mem;

      cytnx_uint32 NBlocks = len / 512;
      if (len % 512) NBlocks += 1;

      if (Lin->size() == 1) {
        cuMod_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMod_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cuMod_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }
    void cuMod_internal_u32tu64(boost::intrusive_ptr<Storage_base> &out,
                                boost::intrusive_ptr<Storage_base> &Lin,
                                boost::intrusive_ptr<Storage_base> &Rin,
                                const unsigned long long &len,
                                const std::vector<cytnx_uint64> &shape,
                                const std::vector<cytnx_uint64> &invmapper_L,
                                const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_uint64 *_out = (cytnx_uint64 *)out->Mem;
      cytnx_uint32 *_Lin = (cytnx_uint32 *)Lin->Mem;
      cytnx_uint64 *_Rin = (cytnx_uint64 *)Rin->Mem;

      cytnx_uint32 NBlocks = len / 512;
      if (len % 512) NBlocks += 1;

      if (Lin->size() == 1) {
        cuMod_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMod_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cuMod_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }
    void cuMod_internal_u32ti32(boost::intrusive_ptr<Storage_base> &out,
                                boost::intrusive_ptr<Storage_base> &Lin,
                                boost::intrusive_ptr<Storage_base> &Rin,
                                const unsigned long long &len,
                                const std::vector<cytnx_uint64> &shape,
                                const std::vector<cytnx_uint64> &invmapper_L,
                                const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_int32 *_out = (cytnx_int32 *)out->Mem;
      cytnx_uint32 *_Lin = (cytnx_uint32 *)Lin->Mem;
      cytnx_int32 *_Rin = (cytnx_int32 *)Rin->Mem;

      cytnx_uint32 NBlocks = len / 512;
      if (len % 512) NBlocks += 1;

      if (Lin->size() == 1) {
        cuMod_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMod_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cuMod_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }
    void cuMod_internal_u32tu32(boost::intrusive_ptr<Storage_base> &out,
                                boost::intrusive_ptr<Storage_base> &Lin,
                                boost::intrusive_ptr<Storage_base> &Rin,
                                const unsigned long long &len,
                                const std::vector<cytnx_uint64> &shape,
                                const std::vector<cytnx_uint64> &invmapper_L,
                                const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_uint32 *_out = (cytnx_uint32 *)out->Mem;
      cytnx_uint32 *_Lin = (cytnx_uint32 *)Lin->Mem;
      cytnx_uint32 *_Rin = (cytnx_uint32 *)Rin->Mem;

      cytnx_uint32 NBlocks = len / 512;
      if (len % 512) NBlocks += 1;

      if (Lin->size() == 1) {
        cuMod_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMod_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cuMod_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }
    void cuMod_internal_u32ti16(boost::intrusive_ptr<Storage_base> &out,
                                boost::intrusive_ptr<Storage_base> &Lin,
                                boost::intrusive_ptr<Storage_base> &Rin,
                                const unsigned long long &len,
                                const std::vector<cytnx_uint64> &shape,
                                const std::vector<cytnx_uint64> &invmapper_L,
                                const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_uint32 *_out = (cytnx_uint32 *)out->Mem;
      cytnx_uint32 *_Lin = (cytnx_uint32 *)Lin->Mem;
      cytnx_int16 *_Rin = (cytnx_int16 *)Rin->Mem;

      cytnx_uint32 NBlocks = len / 512;
      if (len % 512) NBlocks += 1;

      if (Lin->size() == 1) {
        cuMod_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMod_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cuMod_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }
    void cuMod_internal_u32tu16(boost::intrusive_ptr<Storage_base> &out,
                                boost::intrusive_ptr<Storage_base> &Lin,
                                boost::intrusive_ptr<Storage_base> &Rin,
                                const unsigned long long &len,
                                const std::vector<cytnx_uint64> &shape,
                                const std::vector<cytnx_uint64> &invmapper_L,
                                const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_uint32 *_out = (cytnx_uint32 *)out->Mem;
      cytnx_uint32 *_Lin = (cytnx_uint32 *)Lin->Mem;
      cytnx_uint16 *_Rin = (cytnx_uint16 *)Rin->Mem;

      cytnx_uint32 NBlocks = len / 512;
      if (len % 512) NBlocks += 1;

      if (Lin->size() == 1) {
        cuMod_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMod_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cuMod_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }

    __global__ void cuMod_lconst_kernel(cytnx_uint32 *out, const cytnx_uint32 val,
                                        const cytnx_uint64 Nelem, const cytnx_bool *ptr) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          val % cytnx_uint32(ptr[blockIdx.x * blockDim.x + threadIdx.x]);
      }
      __syncthreads();
    }

    __global__ void cuMod_tn_kernel(cytnx_uint32 *out, const cytnx_uint32 *val,
                                    const cytnx_uint64 Nelem, const cytnx_bool *ptr) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          val[blockIdx.x * blockDim.x + threadIdx.x] %
          cytnx_uint32(ptr[blockIdx.x * blockDim.x + threadIdx.x]);
      }
      __syncthreads();
    }
    void cuMod_internal_u32tb(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_uint32 *_out = (cytnx_uint32 *)out->Mem;
      cytnx_uint32 *_Lin = (cytnx_uint32 *)Lin->Mem;
      cytnx_bool *_Rin = (cytnx_bool *)Rin->Mem;

      cytnx_uint32 NBlocks = len / 512;
      if (len % 512) NBlocks += 1;

      if (Lin->size() == 1) {
        cuMod_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMod_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, cytnx_uint32(_Rin[0]));
      } else {
        cuMod_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }

    //---------------------------------------

    void cuMod_internal_i16tcd(boost::intrusive_ptr<Storage_base> &out,
                               boost::intrusive_ptr<Storage_base> &Lin,
                               boost::intrusive_ptr<Storage_base> &Rin,
                               const unsigned long long &len,
                               const std::vector<cytnx_uint64> &shape,
                               const std::vector<cytnx_uint64> &invmapper_L,
                               const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_error_msg(true, "[cuMod] Cannot mod complex numbers%s", "\n");
    }

    void cuMod_internal_i16tcf(boost::intrusive_ptr<Storage_base> &out,
                               boost::intrusive_ptr<Storage_base> &Lin,
                               boost::intrusive_ptr<Storage_base> &Rin,
                               const unsigned long long &len,
                               const std::vector<cytnx_uint64> &shape,
                               const std::vector<cytnx_uint64> &invmapper_L,
                               const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_error_msg(true, "[cuMod] Cannot mod complex numbers%s", "\n");
    }
    void cuMod_internal_i16td(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_double *_out = (cytnx_double *)out->Mem;
      cytnx_int16 *_Lin = (cytnx_int16 *)Lin->Mem;
      cytnx_double *_Rin = (cytnx_double *)Rin->Mem;

      cytnx_uint32 NBlocks = len / 512;
      if (len % 512) NBlocks += 1;

      if (Lin->size() == 1) {
        cufMod_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cufMod_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cufMod_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }
    void cuMod_internal_i16tf(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_float *_out = (cytnx_float *)out->Mem;
      cytnx_int16 *_Lin = (cytnx_int16 *)Lin->Mem;
      cytnx_float *_Rin = (cytnx_float *)Rin->Mem;

      cytnx_uint32 NBlocks = len / 512;
      if (len % 512) NBlocks += 1;

      if (Lin->size() == 1) {
        cufModf_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cufModf_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cufModf_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }
    void cuMod_internal_i16ti64(boost::intrusive_ptr<Storage_base> &out,
                                boost::intrusive_ptr<Storage_base> &Lin,
                                boost::intrusive_ptr<Storage_base> &Rin,
                                const unsigned long long &len,
                                const std::vector<cytnx_uint64> &shape,
                                const std::vector<cytnx_uint64> &invmapper_L,
                                const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_int64 *_out = (cytnx_int64 *)out->Mem;
      cytnx_int16 *_Lin = (cytnx_int16 *)Lin->Mem;
      cytnx_int64 *_Rin = (cytnx_int64 *)Rin->Mem;

      cytnx_uint32 NBlocks = len / 512;
      if (len % 512) NBlocks += 1;

      if (Lin->size() == 1) {
        cuMod_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMod_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cuMod_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }
    void cuMod_internal_i16tu64(boost::intrusive_ptr<Storage_base> &out,
                                boost::intrusive_ptr<Storage_base> &Lin,
                                boost::intrusive_ptr<Storage_base> &Rin,
                                const unsigned long long &len,
                                const std::vector<cytnx_uint64> &shape,
                                const std::vector<cytnx_uint64> &invmapper_L,
                                const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_uint64 *_out = (cytnx_uint64 *)out->Mem;
      cytnx_int16 *_Lin = (cytnx_int16 *)Lin->Mem;
      cytnx_uint64 *_Rin = (cytnx_uint64 *)Rin->Mem;

      cytnx_uint32 NBlocks = len / 512;
      if (len % 512) NBlocks += 1;

      if (Lin->size() == 1) {
        cuMod_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMod_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cuMod_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }
    void cuMod_internal_i16ti32(boost::intrusive_ptr<Storage_base> &out,
                                boost::intrusive_ptr<Storage_base> &Lin,
                                boost::intrusive_ptr<Storage_base> &Rin,
                                const unsigned long long &len,
                                const std::vector<cytnx_uint64> &shape,
                                const std::vector<cytnx_uint64> &invmapper_L,
                                const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_int32 *_out = (cytnx_int32 *)out->Mem;
      cytnx_int16 *_Lin = (cytnx_int16 *)Lin->Mem;
      cytnx_int32 *_Rin = (cytnx_int32 *)Rin->Mem;

      cytnx_uint32 NBlocks = len / 512;
      if (len % 512) NBlocks += 1;

      if (Lin->size() == 1) {
        cuMod_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMod_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cuMod_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }
    void cuMod_internal_i16tu32(boost::intrusive_ptr<Storage_base> &out,
                                boost::intrusive_ptr<Storage_base> &Lin,
                                boost::intrusive_ptr<Storage_base> &Rin,
                                const unsigned long long &len,
                                const std::vector<cytnx_uint64> &shape,
                                const std::vector<cytnx_uint64> &invmapper_L,
                                const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_uint32 *_out = (cytnx_uint32 *)out->Mem;
      cytnx_int16 *_Lin = (cytnx_int16 *)Lin->Mem;
      cytnx_uint32 *_Rin = (cytnx_uint32 *)Rin->Mem;

      cytnx_uint32 NBlocks = len / 512;
      if (len % 512) NBlocks += 1;

      if (Lin->size() == 1) {
        cuMod_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMod_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cuMod_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }
    void cuMod_internal_i16ti16(boost::intrusive_ptr<Storage_base> &out,
                                boost::intrusive_ptr<Storage_base> &Lin,
                                boost::intrusive_ptr<Storage_base> &Rin,
                                const unsigned long long &len,
                                const std::vector<cytnx_uint64> &shape,
                                const std::vector<cytnx_uint64> &invmapper_L,
                                const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_int16 *_out = (cytnx_int16 *)out->Mem;
      cytnx_int16 *_Lin = (cytnx_int16 *)Lin->Mem;
      cytnx_int16 *_Rin = (cytnx_int16 *)Rin->Mem;

      cytnx_uint32 NBlocks = len / 512;
      if (len % 512) NBlocks += 1;

      if (Lin->size() == 1) {
        cuMod_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMod_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cuMod_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }
    void cuMod_internal_i16tu16(boost::intrusive_ptr<Storage_base> &out,
                                boost::intrusive_ptr<Storage_base> &Lin,
                                boost::intrusive_ptr<Storage_base> &Rin,
                                const unsigned long long &len,
                                const std::vector<cytnx_uint64> &shape,
                                const std::vector<cytnx_uint64> &invmapper_L,
                                const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_int16 *_out = (cytnx_int16 *)out->Mem;
      cytnx_int16 *_Lin = (cytnx_int16 *)Lin->Mem;
      cytnx_uint16 *_Rin = (cytnx_uint16 *)Rin->Mem;

      cytnx_uint32 NBlocks = len / 512;
      if (len % 512) NBlocks += 1;

      if (Lin->size() == 1) {
        cuMod_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMod_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cuMod_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }

    __global__ void cuMod_lconst_kernel(cytnx_int16 *out, const cytnx_int16 val,
                                        const cytnx_uint64 Nelem, const cytnx_bool *ptr) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          val % cytnx_int16(ptr[blockIdx.x * blockDim.x + threadIdx.x]);
      }
      __syncthreads();
    }

    __global__ void cuMod_tn_kernel(cytnx_int16 *out, const cytnx_int16 *val,
                                    const cytnx_uint64 Nelem, const cytnx_bool *ptr) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          val[blockIdx.x * blockDim.x + threadIdx.x] %
          cytnx_int16(ptr[blockIdx.x * blockDim.x + threadIdx.x]);
      }
      __syncthreads();
    }
    void cuMod_internal_i16tb(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_int16 *_out = (cytnx_int16 *)out->Mem;
      cytnx_int16 *_Lin = (cytnx_int16 *)Lin->Mem;
      cytnx_bool *_Rin = (cytnx_bool *)Rin->Mem;

      cytnx_uint32 NBlocks = len / 512;
      if (len % 512) NBlocks += 1;

      if (Lin->size() == 1) {
        cuMod_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMod_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, cytnx_int16(_Rin[0]));
      } else {
        cuMod_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }

    //---------------------------------------

    void cuMod_internal_u16tcd(boost::intrusive_ptr<Storage_base> &out,
                               boost::intrusive_ptr<Storage_base> &Lin,
                               boost::intrusive_ptr<Storage_base> &Rin,
                               const unsigned long long &len,
                               const std::vector<cytnx_uint64> &shape,
                               const std::vector<cytnx_uint64> &invmapper_L,
                               const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_error_msg(true, "[cuMod] Cannot mod complex numbers%s", "\n");
    }

    void cuMod_internal_u16tcf(boost::intrusive_ptr<Storage_base> &out,
                               boost::intrusive_ptr<Storage_base> &Lin,
                               boost::intrusive_ptr<Storage_base> &Rin,
                               const unsigned long long &len,
                               const std::vector<cytnx_uint64> &shape,
                               const std::vector<cytnx_uint64> &invmapper_L,
                               const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_error_msg(true, "[cuMod] Cannot mod complex numbers%s", "\n");
    }
    void cuMod_internal_u16td(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_double *_out = (cytnx_double *)out->Mem;
      cytnx_uint16 *_Lin = (cytnx_uint16 *)Lin->Mem;
      cytnx_double *_Rin = (cytnx_double *)Rin->Mem;

      cytnx_uint32 NBlocks = len / 512;
      if (len % 512) NBlocks += 1;

      if (Lin->size() == 1) {
        cufMod_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cufMod_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cufMod_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }
    void cuMod_internal_u16tf(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_float *_out = (cytnx_float *)out->Mem;
      cytnx_uint16 *_Lin = (cytnx_uint16 *)Lin->Mem;
      cytnx_float *_Rin = (cytnx_float *)Rin->Mem;

      cytnx_uint32 NBlocks = len / 512;
      if (len % 512) NBlocks += 1;

      if (Lin->size() == 1) {
        cufModf_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cufModf_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cufModf_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }
    void cuMod_internal_u16ti64(boost::intrusive_ptr<Storage_base> &out,
                                boost::intrusive_ptr<Storage_base> &Lin,
                                boost::intrusive_ptr<Storage_base> &Rin,
                                const unsigned long long &len,
                                const std::vector<cytnx_uint64> &shape,
                                const std::vector<cytnx_uint64> &invmapper_L,
                                const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_int64 *_out = (cytnx_int64 *)out->Mem;
      cytnx_uint16 *_Lin = (cytnx_uint16 *)Lin->Mem;
      cytnx_int64 *_Rin = (cytnx_int64 *)Rin->Mem;

      cytnx_uint32 NBlocks = len / 512;
      if (len % 512) NBlocks += 1;

      if (Lin->size() == 1) {
        cuMod_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMod_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cuMod_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }
    void cuMod_internal_u16tu64(boost::intrusive_ptr<Storage_base> &out,
                                boost::intrusive_ptr<Storage_base> &Lin,
                                boost::intrusive_ptr<Storage_base> &Rin,
                                const unsigned long long &len,
                                const std::vector<cytnx_uint64> &shape,
                                const std::vector<cytnx_uint64> &invmapper_L,
                                const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_uint64 *_out = (cytnx_uint64 *)out->Mem;
      cytnx_uint16 *_Lin = (cytnx_uint16 *)Lin->Mem;
      cytnx_uint64 *_Rin = (cytnx_uint64 *)Rin->Mem;

      cytnx_uint32 NBlocks = len / 512;
      if (len % 512) NBlocks += 1;

      if (Lin->size() == 1) {
        cuMod_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMod_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cuMod_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }
    void cuMod_internal_u16ti32(boost::intrusive_ptr<Storage_base> &out,
                                boost::intrusive_ptr<Storage_base> &Lin,
                                boost::intrusive_ptr<Storage_base> &Rin,
                                const unsigned long long &len,
                                const std::vector<cytnx_uint64> &shape,
                                const std::vector<cytnx_uint64> &invmapper_L,
                                const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_int32 *_out = (cytnx_int32 *)out->Mem;
      cytnx_uint16 *_Lin = (cytnx_uint16 *)Lin->Mem;
      cytnx_int32 *_Rin = (cytnx_int32 *)Rin->Mem;

      cytnx_uint32 NBlocks = len / 512;
      if (len % 512) NBlocks += 1;

      if (Lin->size() == 1) {
        cuMod_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMod_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cuMod_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }
    void cuMod_internal_u16tu32(boost::intrusive_ptr<Storage_base> &out,
                                boost::intrusive_ptr<Storage_base> &Lin,
                                boost::intrusive_ptr<Storage_base> &Rin,
                                const unsigned long long &len,
                                const std::vector<cytnx_uint64> &shape,
                                const std::vector<cytnx_uint64> &invmapper_L,
                                const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_uint32 *_out = (cytnx_uint32 *)out->Mem;
      cytnx_uint16 *_Lin = (cytnx_uint16 *)Lin->Mem;
      cytnx_uint32 *_Rin = (cytnx_uint32 *)Rin->Mem;

      cytnx_uint32 NBlocks = len / 512;
      if (len % 512) NBlocks += 1;

      if (Lin->size() == 1) {
        cuMod_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMod_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cuMod_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }
    void cuMod_internal_u16ti16(boost::intrusive_ptr<Storage_base> &out,
                                boost::intrusive_ptr<Storage_base> &Lin,
                                boost::intrusive_ptr<Storage_base> &Rin,
                                const unsigned long long &len,
                                const std::vector<cytnx_uint64> &shape,
                                const std::vector<cytnx_uint64> &invmapper_L,
                                const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_int16 *_out = (cytnx_int16 *)out->Mem;
      cytnx_uint16 *_Lin = (cytnx_uint16 *)Lin->Mem;
      cytnx_int16 *_Rin = (cytnx_int16 *)Rin->Mem;

      cytnx_uint32 NBlocks = len / 512;
      if (len % 512) NBlocks += 1;

      if (Lin->size() == 1) {
        cuMod_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMod_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cuMod_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }
    void cuMod_internal_u16tu16(boost::intrusive_ptr<Storage_base> &out,
                                boost::intrusive_ptr<Storage_base> &Lin,
                                boost::intrusive_ptr<Storage_base> &Rin,
                                const unsigned long long &len,
                                const std::vector<cytnx_uint64> &shape,
                                const std::vector<cytnx_uint64> &invmapper_L,
                                const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_uint16 *_out = (cytnx_uint16 *)out->Mem;
      cytnx_uint16 *_Lin = (cytnx_uint16 *)Lin->Mem;
      cytnx_uint16 *_Rin = (cytnx_uint16 *)Rin->Mem;

      cytnx_uint32 NBlocks = len / 512;
      if (len % 512) NBlocks += 1;

      if (Lin->size() == 1) {
        cuMod_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMod_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cuMod_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }

    __global__ void cuMod_lconst_kernel(cytnx_uint16 *out, const cytnx_uint16 val,
                                        const cytnx_uint64 Nelem, const cytnx_bool *ptr) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          val % cytnx_uint16(ptr[blockIdx.x * blockDim.x + threadIdx.x]);
      }
      __syncthreads();
    }

    __global__ void cuMod_tn_kernel(cytnx_uint16 *out, const cytnx_uint16 *val,
                                    const cytnx_uint64 Nelem, const cytnx_bool *ptr) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          val[blockIdx.x * blockDim.x + threadIdx.x] %
          cytnx_uint16(ptr[blockIdx.x * blockDim.x + threadIdx.x]);
      }
      __syncthreads();
    }
    void cuMod_internal_u16tb(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_uint16 *_out = (cytnx_uint16 *)out->Mem;
      cytnx_uint16 *_Lin = (cytnx_uint16 *)Lin->Mem;
      cytnx_bool *_Rin = (cytnx_bool *)Rin->Mem;

      cytnx_uint32 NBlocks = len / 512;
      if (len % 512) NBlocks += 1;

      if (Lin->size() == 1) {
        cuMod_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMod_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, cytnx_uint16(_Rin[0]));
      } else {
        cuMod_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }

    //---------------------------------------

    void cuMod_internal_btcd(boost::intrusive_ptr<Storage_base> &out,
                             boost::intrusive_ptr<Storage_base> &Lin,
                             boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                             const std::vector<cytnx_uint64> &shape,
                             const std::vector<cytnx_uint64> &invmapper_L,
                             const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_error_msg(true, "[cuMod] Cannot mod complex numbers%s", "\n");
    }

    void cuMod_internal_btcf(boost::intrusive_ptr<Storage_base> &out,
                             boost::intrusive_ptr<Storage_base> &Lin,
                             boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                             const std::vector<cytnx_uint64> &shape,
                             const std::vector<cytnx_uint64> &invmapper_L,
                             const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_error_msg(true, "[cuMod] Cannot mod complex numbers%s", "\n");
    }

    void cuMod_internal_btd(boost::intrusive_ptr<Storage_base> &out,
                            boost::intrusive_ptr<Storage_base> &Lin,
                            boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                            const std::vector<cytnx_uint64> &shape,
                            const std::vector<cytnx_uint64> &invmapper_L,
                            const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_double *_out = (cytnx_double *)out->Mem;
      cytnx_bool *_Lin = (cytnx_bool *)Lin->Mem;
      cytnx_double *_Rin = (cytnx_double *)Rin->Mem;

      cytnx_uint32 NBlocks = len / 512;
      if (len % 512) NBlocks += 1;

      if (Lin->size() == 1) {
        cufMod_lconst_kernel<<<NBlocks, 512>>>(_out, cytnx_double(_Lin[0]), len, _Rin);
      } else if (Rin->size() == 1) {
        cufMod_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cufMod_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }

    void cuMod_internal_btf(boost::intrusive_ptr<Storage_base> &out,
                            boost::intrusive_ptr<Storage_base> &Lin,
                            boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                            const std::vector<cytnx_uint64> &shape,
                            const std::vector<cytnx_uint64> &invmapper_L,
                            const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_float *_out = (cytnx_float *)out->Mem;
      cytnx_bool *_Lin = (cytnx_bool *)Lin->Mem;
      cytnx_float *_Rin = (cytnx_float *)Rin->Mem;

      cytnx_uint32 NBlocks = len / 512;
      if (len % 512) NBlocks += 1;

      if (Lin->size() == 1) {
        cufModf_lconst_kernel<<<NBlocks, 512>>>(_out, cytnx_float(_Lin[0]), len, _Rin);
      } else if (Rin->size() == 1) {
        cufModf_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cufModf_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }
    __global__ void cuMod_rconst_kernel(cytnx_int64 *out, const cytnx_bool *ptr,
                                        const cytnx_uint64 Nelem, const cytnx_int64 val) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          cytnx_int64(ptr[blockIdx.x * blockDim.x + threadIdx.x]) % val;
      }
      __syncthreads();
    }

    __global__ void cuMod_tn_kernel(cytnx_int64 *out, const cytnx_bool *ptr,
                                    const cytnx_uint64 Nelem, const cytnx_int64 *val) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          cytnx_int64(ptr[blockIdx.x * blockDim.x + threadIdx.x]) %
          val[blockIdx.x * blockDim.x + threadIdx.x];
      }
      __syncthreads();
    }
    void cuMod_internal_bti64(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_int64 *_out = (cytnx_int64 *)out->Mem;
      cytnx_bool *_Lin = (cytnx_bool *)Lin->Mem;
      cytnx_int64 *_Rin = (cytnx_int64 *)Rin->Mem;

      cytnx_uint32 NBlocks = len / 512;
      if (len % 512) NBlocks += 1;

      if (Lin->size() == 1) {
        cuMod_lconst_kernel<<<NBlocks, 512>>>(_out, cytnx_int64(_Lin[0]), len, _Rin);
      } else if (Rin->size() == 1) {
        cuMod_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cuMod_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }
    __global__ void cuMod_rconst_kernel(cytnx_uint64 *out, const cytnx_bool *ptr,
                                        const cytnx_uint64 Nelem, const cytnx_uint64 val) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          cytnx_uint64(ptr[blockIdx.x * blockDim.x + threadIdx.x]) % val;
      }
      __syncthreads();
    }

    __global__ void cuMod_tn_kernel(cytnx_uint64 *out, const cytnx_bool *ptr,
                                    const cytnx_uint64 Nelem, const cytnx_uint64 *val) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          cytnx_uint64(ptr[blockIdx.x * blockDim.x + threadIdx.x]) %
          val[blockIdx.x * blockDim.x + threadIdx.x];
      }
      __syncthreads();
    }
    void cuMod_internal_btu64(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_uint64 *_out = (cytnx_uint64 *)out->Mem;
      cytnx_bool *_Lin = (cytnx_bool *)Lin->Mem;
      cytnx_uint64 *_Rin = (cytnx_uint64 *)Rin->Mem;

      cytnx_uint32 NBlocks = len / 512;
      if (len % 512) NBlocks += 1;

      if (Lin->size() == 1) {
        cuMod_lconst_kernel<<<NBlocks, 512>>>(_out, cytnx_uint64(_Lin[0]), len, _Rin);
      } else if (Rin->size() == 1) {
        cuMod_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cuMod_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }
    __global__ void cuMod_rconst_kernel(cytnx_int32 *out, const cytnx_bool *ptr,
                                        const cytnx_uint64 Nelem, const cytnx_int32 val) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          cytnx_int32(ptr[blockIdx.x * blockDim.x + threadIdx.x]) % val;
      }
      __syncthreads();
    }

    __global__ void cuMod_tn_kernel(cytnx_int32 *out, const cytnx_bool *ptr,
                                    const cytnx_uint64 Nelem, const cytnx_int32 *val) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          cytnx_int32(ptr[blockIdx.x * blockDim.x + threadIdx.x]) %
          val[blockIdx.x * blockDim.x + threadIdx.x];
      }
      __syncthreads();
    }
    void cuMod_internal_bti32(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_int32 *_out = (cytnx_int32 *)out->Mem;
      cytnx_bool *_Lin = (cytnx_bool *)Lin->Mem;
      cytnx_int32 *_Rin = (cytnx_int32 *)Rin->Mem;

      cytnx_uint32 NBlocks = len / 512;
      if (len % 512) NBlocks += 1;

      if (Lin->size() == 1) {
        cuMod_lconst_kernel<<<NBlocks, 512>>>(_out, cytnx_int32(_Lin[0]), len, _Rin);
      } else if (Rin->size() == 1) {
        cuMod_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cuMod_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }
    __global__ void cuMod_rconst_kernel(cytnx_uint32 *out, const cytnx_bool *ptr,
                                        const cytnx_uint64 Nelem, const cytnx_uint32 val) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          cytnx_uint32(ptr[blockIdx.x * blockDim.x + threadIdx.x]) % val;
      }
      __syncthreads();
    }

    __global__ void cuMod_tn_kernel(cytnx_uint32 *out, const cytnx_bool *ptr,
                                    const cytnx_uint64 Nelem, const cytnx_uint32 *val) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          cytnx_uint32(ptr[blockIdx.x * blockDim.x + threadIdx.x]) %
          val[blockIdx.x * blockDim.x + threadIdx.x];
      }
      __syncthreads();
    }
    void cuMod_internal_btu32(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_uint32 *_out = (cytnx_uint32 *)out->Mem;
      cytnx_bool *_Lin = (cytnx_bool *)Lin->Mem;
      cytnx_uint32 *_Rin = (cytnx_uint32 *)Rin->Mem;

      cytnx_uint32 NBlocks = len / 512;
      if (len % 512) NBlocks += 1;

      if (Lin->size() == 1) {
        cuMod_lconst_kernel<<<NBlocks, 512>>>(_out, cytnx_uint32(_Lin[0]), len, _Rin);
      } else if (Rin->size() == 1) {
        cuMod_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cuMod_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }
    __global__ void cuMod_rconst_kernel(cytnx_int16 *out, const cytnx_bool *ptr,
                                        const cytnx_uint64 Nelem, const cytnx_int16 val) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          cytnx_int16(ptr[blockIdx.x * blockDim.x + threadIdx.x]) % val;
      }
      __syncthreads();
    }

    __global__ void cuMod_tn_kernel(cytnx_int16 *out, const cytnx_bool *ptr,
                                    const cytnx_uint64 Nelem, const cytnx_int16 *val) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          cytnx_int16(ptr[blockIdx.x * blockDim.x + threadIdx.x]) %
          val[blockIdx.x * blockDim.x + threadIdx.x];
      }
      __syncthreads();
    }
    void cuMod_internal_bti16(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_int16 *_out = (cytnx_int16 *)out->Mem;
      cytnx_bool *_Lin = (cytnx_bool *)Lin->Mem;
      cytnx_int16 *_Rin = (cytnx_int16 *)Rin->Mem;

      cytnx_uint32 NBlocks = len / 512;
      if (len % 512) NBlocks += 1;

      if (Lin->size() == 1) {
        cuMod_lconst_kernel<<<NBlocks, 512>>>(_out, cytnx_int16(_Lin[0]), len, _Rin);
      } else if (Rin->size() == 1) {
        cuMod_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cuMod_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }
    __global__ void cuMod_rconst_kernel(cytnx_uint16 *out, const cytnx_bool *ptr,
                                        const cytnx_uint64 Nelem, const cytnx_uint16 val) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          cytnx_uint16(ptr[blockIdx.x * blockDim.x + threadIdx.x]) % val;
      }
      __syncthreads();
    }

    __global__ void cuMod_tn_kernel(cytnx_uint16 *out, const cytnx_bool *ptr,
                                    const cytnx_uint64 Nelem, const cytnx_uint16 *val) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          cytnx_uint16(ptr[blockIdx.x * blockDim.x + threadIdx.x]) %
          val[blockIdx.x * blockDim.x + threadIdx.x];
      }
      __syncthreads();
    }
    void cuMod_internal_btu16(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_uint16 *_out = (cytnx_uint16 *)out->Mem;
      cytnx_bool *_Lin = (cytnx_bool *)Lin->Mem;
      cytnx_uint16 *_Rin = (cytnx_uint16 *)Rin->Mem;

      cytnx_uint32 NBlocks = len / 512;
      if (len % 512) NBlocks += 1;

      if (Lin->size() == 1) {
        cuMod_lconst_kernel<<<NBlocks, 512>>>(_out, cytnx_uint16(_Lin[0]), len, _Rin);
      } else if (Rin->size() == 1) {
        cuMod_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cuMod_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }

    void cuMod_internal_btb(boost::intrusive_ptr<Storage_base> &out,
                            boost::intrusive_ptr<Storage_base> &Lin,
                            boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                            const std::vector<cytnx_uint64> &shape,
                            const std::vector<cytnx_uint64> &invmapper_L,
                            const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_bool *_out = (cytnx_bool *)out->Mem;
      cytnx_bool *_Lin = (cytnx_bool *)Lin->Mem;
      cytnx_bool *_Rin = (cytnx_bool *)Rin->Mem;

      cytnx_uint32 NBlocks = len / 512;
      if (len % 512) NBlocks += 1;

      if (Lin->size() == 1) {
        cuMod_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMod_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cuMod_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }

  }  // namespace linalg_internal
}  // namespace cytnx
