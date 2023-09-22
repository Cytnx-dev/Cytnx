#include "cuMul_internal.hpp"
#include "../utils_internal_interface.hpp"

#ifdef UNI_OMP
  #include <omp.h>
#endif

namespace cytnx {

  namespace linalg_internal {

    //====================================================================
    // generic R+R kernel
    template <class T1, class T2, class T3>
    __global__ void cuMul_constconst_kernel(T1 *out, const T2 ptr, const cytnx_uint64 Nelem,
                                            const T3 val) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] = ptr * val;
      }
      __syncthreads();
    }
    template <class T1, class T2, class T3>
    __global__ void cuMul_rconst_kernel(T1 *out, const T2 *ptr, const cytnx_uint64 Nelem,
                                        const T3 val) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          ptr[blockIdx.x * blockDim.x + threadIdx.x] * val;
      }
      __syncthreads();
    }

    template <class T1, class T2, class T3>
    __global__ void cuMul_lconst_kernel(T1 *out, const T2 val, const cytnx_uint64 Nelem,
                                        const T3 *ptr) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          val * ptr[blockIdx.x * blockDim.x + threadIdx.x];
      }
      __syncthreads();
    }

    template <class T1, class T2, class T3>
    __global__ void cuMul_tn_kernel(T1 *out, const T2 *val, const cytnx_uint64 Nelem,
                                    const T3 *ptr) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          val[blockIdx.x * blockDim.x + threadIdx.x] * ptr[blockIdx.x * blockDim.x + threadIdx.x];
      }
      __syncthreads();
    }

    //=====================================================================

    /// cuMul
    __global__ void cuMul_constconst_kernel(cuDoubleComplex *out, const cuDoubleComplex ptr,
                                            const cytnx_uint64 Nelem, const cuDoubleComplex val) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] = cuCmul(ptr, val);
      }
      __syncthreads();
    }
    __global__ void cuMul_rconst_kernel(cuDoubleComplex *out, const cuDoubleComplex *ptr,
                                        const cytnx_uint64 Nelem, const cuDoubleComplex val) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          cuCmul(ptr[blockIdx.x * blockDim.x + threadIdx.x], val);
      }
      __syncthreads();
    }
    __global__ void cuMul_lconst_kernel(cuDoubleComplex *out, const cuDoubleComplex val,
                                        const cytnx_uint64 Nelem, const cuDoubleComplex *ptr) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          cuCmul(ptr[blockIdx.x * blockDim.x + threadIdx.x], val);
      }
      __syncthreads();
    }
    __global__ void cuMul_tn_kernel(cuDoubleComplex *out, const cuDoubleComplex *ptr,
                                    const cytnx_uint64 Nelem, const cuDoubleComplex *val) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] = cuCmul(
          ptr[blockIdx.x * blockDim.x + threadIdx.x], val[blockIdx.x * blockDim.x + threadIdx.x]);
      }
      __syncthreads();
    }
    void cuMul_internal_cdtcd(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R) {
      cuDoubleComplex *_out = (cuDoubleComplex *)out->Mem;
      cuDoubleComplex *_Lin = (cuDoubleComplex *)Lin->Mem;
      cuDoubleComplex *_Rin = (cuDoubleComplex *)Rin->Mem;

      cytnx_uint32 NBlocks = len / 512;
      if (len % 512) NBlocks += 1;

      if (Lin->size() == 1 and Rin->size() == 1) {
        cuMul_constconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin[0]);
      } else if (Lin->size() == 1) {
        cuMul_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMul_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cuMul_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }

    __global__ void cuMul_constconst_kernel(cuDoubleComplex *out, const cuDoubleComplex ptr,
                                            const cytnx_uint64 Nelem, const cuFloatComplex val) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] = cuCmul(ptr, cuComplexFloatToDouble(val));
      }
      __syncthreads();
    }
    __global__ void cuMul_rconst_kernel(cuDoubleComplex *out, const cuDoubleComplex *ptr,
                                        const cytnx_uint64 Nelem, const cuFloatComplex val) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          cuCmul(ptr[blockIdx.x * blockDim.x + threadIdx.x], cuComplexFloatToDouble(val));
      }
      __syncthreads();
    }
    __global__ void cuMul_lconst_kernel(cuDoubleComplex *out, const cuDoubleComplex val,
                                        const cytnx_uint64 Nelem, const cuFloatComplex *ptr) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          cuCmul(val, cuComplexFloatToDouble(ptr[blockIdx.x * blockDim.x + threadIdx.x]));
      }
      __syncthreads();
    }
    __global__ void cuMul_tn_kernel(cuDoubleComplex *out, const cuDoubleComplex *ptr,
                                    const cytnx_uint64 Nelem, const cuFloatComplex *val) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          cuCmul(ptr[blockIdx.x * blockDim.x + threadIdx.x],
                 cuComplexFloatToDouble(val[blockIdx.x * blockDim.x + threadIdx.x]));
      }
      __syncthreads();
    }
    void cuMul_internal_cdtcf(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R) {
      cuDoubleComplex *_out = (cuDoubleComplex *)out->Mem;
      cuDoubleComplex *_Lin = (cuDoubleComplex *)Lin->Mem;
      cuFloatComplex *_Rin = (cuFloatComplex *)Rin->Mem;

      cytnx_uint32 NBlocks = len / 512;
      if (len % 512) NBlocks += 1;

      if (Lin->size() == 1 and Rin->size() == 1) {
        cuMul_constconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin[0]);
      } else if (Lin->size() == 1) {
        cuMul_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMul_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cuMul_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }

    __global__ void cuMul_constconst_kernel(cuDoubleComplex *out, const cuDoubleComplex ptr,
                                            const cytnx_uint64 Nelem, const cytnx_double val) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] = cuCmul(ptr, make_cuDoubleComplex(val, 0));
      }
      __syncthreads();
    }
    __global__ void cuMul_rconst_kernel(cuDoubleComplex *out, const cuDoubleComplex *ptr,
                                        const cytnx_uint64 Nelem, const cytnx_double val) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          cuCmul(ptr[blockIdx.x * blockDim.x + threadIdx.x], make_cuDoubleComplex(val, 0));
      }
      __syncthreads();
    }
    __global__ void cuMul_lconst_kernel(cuDoubleComplex *out, const cuDoubleComplex val,
                                        const cytnx_uint64 Nelem, const cytnx_double *ptr) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          cuCmul(val, make_cuDoubleComplex(ptr[blockIdx.x * blockDim.x + threadIdx.x], 0));
      }
      __syncthreads();
    }
    __global__ void cuMul_tn_kernel(cuDoubleComplex *out, const cuDoubleComplex *ptr,
                                    const cytnx_uint64 Nelem, const cytnx_double *val) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          cuCmul(ptr[blockIdx.x * blockDim.x + threadIdx.x],
                 make_cuDoubleComplex(val[blockIdx.x * blockDim.x + threadIdx.x], 0));
      }
      __syncthreads();
    }
    void cuMul_internal_cdtd(boost::intrusive_ptr<Storage_base> &out,
                             boost::intrusive_ptr<Storage_base> &Lin,
                             boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                             const std::vector<cytnx_uint64> &shape,
                             const std::vector<cytnx_uint64> &invmapper_L,
                             const std::vector<cytnx_uint64> &invmapper_R) {
      cuDoubleComplex *_out = (cuDoubleComplex *)out->Mem;
      cuDoubleComplex *_Lin = (cuDoubleComplex *)Lin->Mem;
      cytnx_double *_Rin = (cytnx_double *)Rin->Mem;

      cytnx_uint32 NBlocks = len / 512;
      if (len % 512) NBlocks += 1;

      if (Lin->size() == 1 and Rin->size() == 1) {
        cuMul_constconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin[0]);
      } else if (Lin->size() == 1) {
        cuMul_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMul_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cuMul_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }

    __global__ void cuMul_constconst_kernel(cuDoubleComplex *out, const cuDoubleComplex ptr,
                                            const cytnx_uint64 Nelem, const cytnx_float val) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] = cuCmul(ptr, make_cuDoubleComplex(val, 0));
      }
      __syncthreads();
    }
    __global__ void cuMul_rconst_kernel(cuDoubleComplex *out, const cuDoubleComplex *ptr,
                                        const cytnx_uint64 Nelem, const cytnx_float val) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          cuCmul(ptr[blockIdx.x * blockDim.x + threadIdx.x], make_cuDoubleComplex(val, 0));
      }
      __syncthreads();
    }
    __global__ void cuMul_lconst_kernel(cuDoubleComplex *out, const cuDoubleComplex val,
                                        const cytnx_uint64 Nelem, const cytnx_float *ptr) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          cuCmul(val, make_cuDoubleComplex(ptr[blockIdx.x * blockDim.x + threadIdx.x], 0));
      }
      __syncthreads();
    }
    __global__ void cuMul_tn_kernel(cuDoubleComplex *out, const cuDoubleComplex *ptr,
                                    const cytnx_uint64 Nelem, const cytnx_float *val) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          cuCmul(ptr[blockIdx.x * blockDim.x + threadIdx.x],
                 make_cuDoubleComplex(val[blockIdx.x * blockDim.x + threadIdx.x], 0));
      }
      __syncthreads();
    }
    void cuMul_internal_cdtf(boost::intrusive_ptr<Storage_base> &out,
                             boost::intrusive_ptr<Storage_base> &Lin,
                             boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                             const std::vector<cytnx_uint64> &shape,
                             const std::vector<cytnx_uint64> &invmapper_L,
                             const std::vector<cytnx_uint64> &invmapper_R) {
      cuDoubleComplex *_out = (cuDoubleComplex *)out->Mem;
      cuDoubleComplex *_Lin = (cuDoubleComplex *)Lin->Mem;
      cytnx_float *_Rin = (cytnx_float *)Rin->Mem;

      cytnx_uint32 NBlocks = len / 512;
      if (len % 512) NBlocks += 1;

      if (Lin->size() == 1 and Rin->size() == 1) {
        cuMul_constconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin[0]);
      } else if (Lin->size() == 1) {
        cuMul_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMul_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cuMul_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }

    __global__ void cuMul_constconst_kernel(cuDoubleComplex *out, const cuDoubleComplex ptr,
                                            const cytnx_uint64 Nelem, const cytnx_uint64 val) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] = cuCmul(ptr, make_cuDoubleComplex(val, 0));
      }
      __syncthreads();
    }
    __global__ void cuMul_rconst_kernel(cuDoubleComplex *out, const cuDoubleComplex *ptr,
                                        const cytnx_uint64 Nelem, const cytnx_uint64 val) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          cuCmul(ptr[blockIdx.x * blockDim.x + threadIdx.x], make_cuDoubleComplex(val, 0));
      }
      __syncthreads();
    }
    __global__ void cuMul_lconst_kernel(cuDoubleComplex *out, const cuDoubleComplex val,
                                        const cytnx_uint64 Nelem, const cytnx_uint64 *ptr) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          cuCmul(val, make_cuDoubleComplex(ptr[blockIdx.x * blockDim.x + threadIdx.x], 0));
      }
      __syncthreads();
    }
    __global__ void cuMul_tn_kernel(cuDoubleComplex *out, const cuDoubleComplex *ptr,
                                    const cytnx_uint64 Nelem, const cytnx_uint64 *val) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          cuCmul(ptr[blockIdx.x * blockDim.x + threadIdx.x],
                 make_cuDoubleComplex(val[blockIdx.x * blockDim.x + threadIdx.x], 0));
      }
      __syncthreads();
    }

    void cuMul_internal_cdtu64(boost::intrusive_ptr<Storage_base> &out,
                               boost::intrusive_ptr<Storage_base> &Lin,
                               boost::intrusive_ptr<Storage_base> &Rin,
                               const unsigned long long &len,
                               const std::vector<cytnx_uint64> &shape,
                               const std::vector<cytnx_uint64> &invmapper_L,
                               const std::vector<cytnx_uint64> &invmapper_R) {
      cuDoubleComplex *_out = (cuDoubleComplex *)out->Mem;
      cuDoubleComplex *_Lin = (cuDoubleComplex *)Lin->Mem;
      cytnx_uint64 *_Rin = (cytnx_uint64 *)Rin->Mem;

      cytnx_uint32 NBlocks = len / 512;
      if (len % 512) NBlocks += 1;

      if (Lin->size() == 1 and Rin->size() == 1) {
        cuMul_constconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin[0]);
      } else if (Lin->size() == 1) {
        cuMul_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMul_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cuMul_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }

    __global__ void cuMul_constconst_kernel(cuDoubleComplex *out, const cuDoubleComplex ptr,
                                            const cytnx_uint64 Nelem, const cytnx_uint32 val) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] = cuCmul(ptr, make_cuDoubleComplex(val, 0));
      }
      __syncthreads();
    }
    __global__ void cuMul_rconst_kernel(cuDoubleComplex *out, const cuDoubleComplex *ptr,
                                        const cytnx_uint64 Nelem, const cytnx_uint32 val) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          cuCmul(ptr[blockIdx.x * blockDim.x + threadIdx.x], make_cuDoubleComplex(val, 0));
      }
      __syncthreads();
    }
    __global__ void cuMul_lconst_kernel(cuDoubleComplex *out, const cuDoubleComplex val,
                                        const cytnx_uint64 Nelem, const cytnx_uint32 *ptr) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          cuCmul(val, make_cuDoubleComplex(ptr[blockIdx.x * blockDim.x + threadIdx.x], 0));
      }
      __syncthreads();
    }
    __global__ void cuMul_tn_kernel(cuDoubleComplex *out, const cuDoubleComplex *ptr,
                                    const cytnx_uint64 Nelem, const cytnx_uint32 *val) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          cuCmul(ptr[blockIdx.x * blockDim.x + threadIdx.x],
                 make_cuDoubleComplex(val[blockIdx.x * blockDim.x + threadIdx.x], 0));
      }
      __syncthreads();
    }
    void cuMul_internal_cdtu32(boost::intrusive_ptr<Storage_base> &out,
                               boost::intrusive_ptr<Storage_base> &Lin,
                               boost::intrusive_ptr<Storage_base> &Rin,
                               const unsigned long long &len,
                               const std::vector<cytnx_uint64> &shape,
                               const std::vector<cytnx_uint64> &invmapper_L,
                               const std::vector<cytnx_uint64> &invmapper_R) {
      cuDoubleComplex *_out = (cuDoubleComplex *)out->Mem;
      cuDoubleComplex *_Lin = (cuDoubleComplex *)Lin->Mem;
      cytnx_uint32 *_Rin = (cytnx_uint32 *)Rin->Mem;

      cytnx_uint32 NBlocks = len / 512;
      if (len % 512) NBlocks += 1;

      if (Lin->size() == 1 and Rin->size() == 1) {
        cuMul_constconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin[0]);
      } else if (Lin->size() == 1) {
        cuMul_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMul_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cuMul_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }

    __global__ void cuMul_constconst_kernel(cuDoubleComplex *out, const cuDoubleComplex ptr,
                                            const cytnx_uint64 Nelem, const cytnx_int64 val) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] = cuCmul(ptr, make_cuDoubleComplex(val, 0));
      }
      __syncthreads();
    }
    __global__ void cuMul_rconst_kernel(cuDoubleComplex *out, const cuDoubleComplex *ptr,
                                        const cytnx_uint64 Nelem, const cytnx_int64 val) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          cuCmul(ptr[blockIdx.x * blockDim.x + threadIdx.x], make_cuDoubleComplex(val, 0));
      }
      __syncthreads();
    }
    __global__ void cuMul_lconst_kernel(cuDoubleComplex *out, const cuDoubleComplex val,
                                        const cytnx_uint64 Nelem, const cytnx_int64 *ptr) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          cuCmul(val, make_cuDoubleComplex(ptr[blockIdx.x * blockDim.x + threadIdx.x], 0));
      }
      __syncthreads();
    }
    __global__ void cuMul_tn_kernel(cuDoubleComplex *out, const cuDoubleComplex *ptr,
                                    const cytnx_uint64 Nelem, const cytnx_int64 *val) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          cuCmul(ptr[blockIdx.x * blockDim.x + threadIdx.x],
                 make_cuDoubleComplex(val[blockIdx.x * blockDim.x + threadIdx.x], 0));
      }
      __syncthreads();
    }
    void cuMul_internal_cdti64(boost::intrusive_ptr<Storage_base> &out,
                               boost::intrusive_ptr<Storage_base> &Lin,
                               boost::intrusive_ptr<Storage_base> &Rin,
                               const unsigned long long &len,
                               const std::vector<cytnx_uint64> &shape,
                               const std::vector<cytnx_uint64> &invmapper_L,
                               const std::vector<cytnx_uint64> &invmapper_R) {
      cuDoubleComplex *_out = (cuDoubleComplex *)out->Mem;
      cuDoubleComplex *_Lin = (cuDoubleComplex *)Lin->Mem;
      cytnx_int64 *_Rin = (cytnx_int64 *)Rin->Mem;

      cytnx_uint32 NBlocks = len / 512;
      if (len % 512) NBlocks += 1;

      if (Lin->size() == 1 and Rin->size() == 1) {
        cuMul_constconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin[0]);
      } else if (Lin->size() == 1) {
        cuMul_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMul_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cuMul_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }

    __global__ void cuMul_constconst_kernel(cuDoubleComplex *out, const cuDoubleComplex ptr,
                                            const cytnx_uint64 Nelem, const cytnx_int32 val) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] = cuCmul(ptr, make_cuDoubleComplex(val, 0));
      }
      __syncthreads();
    }
    __global__ void cuMul_rconst_kernel(cuDoubleComplex *out, const cuDoubleComplex *ptr,
                                        const cytnx_uint64 Nelem, const cytnx_int32 val) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          cuCmul(ptr[blockIdx.x * blockDim.x + threadIdx.x], make_cuDoubleComplex(val, 0));
      }
      __syncthreads();
    }
    __global__ void cuMul_lconst_kernel(cuDoubleComplex *out, const cuDoubleComplex val,
                                        const cytnx_uint64 Nelem, const cytnx_int32 *ptr) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          cuCmul(val, make_cuDoubleComplex(ptr[blockIdx.x * blockDim.x + threadIdx.x], 0));
      }
      __syncthreads();
    }
    __global__ void cuMul_tn_kernel(cuDoubleComplex *out, const cuDoubleComplex *ptr,
                                    const cytnx_uint64 Nelem, const cytnx_int32 *val) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          cuCmul(ptr[blockIdx.x * blockDim.x + threadIdx.x],
                 make_cuDoubleComplex(val[blockIdx.x * blockDim.x + threadIdx.x], 0));
      }
      __syncthreads();
    }
    void cuMul_internal_cdti32(boost::intrusive_ptr<Storage_base> &out,
                               boost::intrusive_ptr<Storage_base> &Lin,
                               boost::intrusive_ptr<Storage_base> &Rin,
                               const unsigned long long &len,
                               const std::vector<cytnx_uint64> &shape,
                               const std::vector<cytnx_uint64> &invmapper_L,
                               const std::vector<cytnx_uint64> &invmapper_R) {
      cuDoubleComplex *_out = (cuDoubleComplex *)out->Mem;
      cuDoubleComplex *_Lin = (cuDoubleComplex *)Lin->Mem;
      cytnx_int32 *_Rin = (cytnx_int32 *)Rin->Mem;

      cytnx_uint32 NBlocks = len / 512;
      if (len % 512) NBlocks += 1;

      if (Lin->size() == 1 and Rin->size() == 1) {
        cuMul_constconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin[0]);
      } else if (Lin->size() == 1) {
        cuMul_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMul_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cuMul_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }

    __global__ void cuMul_constconst_kernel(cuDoubleComplex *out, const cuDoubleComplex ptr,
                                            const cytnx_uint64 Nelem, const cytnx_int16 val) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] = cuCmul(ptr, make_cuDoubleComplex(val, 0));
      }
      __syncthreads();
    }
    __global__ void cuMul_rconst_kernel(cuDoubleComplex *out, const cuDoubleComplex *ptr,
                                        const cytnx_uint64 Nelem, const cytnx_int16 val) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          cuCmul(ptr[blockIdx.x * blockDim.x + threadIdx.x], make_cuDoubleComplex(val, 0));
      }
      __syncthreads();
    }
    __global__ void cuMul_lconst_kernel(cuDoubleComplex *out, const cuDoubleComplex val,
                                        const cytnx_uint64 Nelem, const cytnx_int16 *ptr) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          cuCmul(val, make_cuDoubleComplex(ptr[blockIdx.x * blockDim.x + threadIdx.x], 0));
      }
      __syncthreads();
    }
    __global__ void cuMul_tn_kernel(cuDoubleComplex *out, const cuDoubleComplex *ptr,
                                    const cytnx_uint64 Nelem, const cytnx_int16 *val) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          cuCmul(ptr[blockIdx.x * blockDim.x + threadIdx.x],
                 make_cuDoubleComplex(val[blockIdx.x * blockDim.x + threadIdx.x], 0));
      }
      __syncthreads();
    }
    void cuMul_internal_cdti16(boost::intrusive_ptr<Storage_base> &out,
                               boost::intrusive_ptr<Storage_base> &Lin,
                               boost::intrusive_ptr<Storage_base> &Rin,
                               const unsigned long long &len,
                               const std::vector<cytnx_uint64> &shape,
                               const std::vector<cytnx_uint64> &invmapper_L,
                               const std::vector<cytnx_uint64> &invmapper_R) {
      cuDoubleComplex *_out = (cuDoubleComplex *)out->Mem;
      cuDoubleComplex *_Lin = (cuDoubleComplex *)Lin->Mem;
      cytnx_int16 *_Rin = (cytnx_int16 *)Rin->Mem;

      cytnx_uint32 NBlocks = len / 512;
      if (len % 512) NBlocks += 1;

      if (Lin->size() == 1 and Rin->size() == 1) {
        cuMul_constconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin[0]);
      } else if (Lin->size() == 1) {
        cuMul_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMul_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cuMul_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }

    __global__ void cuMul_constconst_kernel(cuDoubleComplex *out, const cuDoubleComplex ptr,
                                            const cytnx_uint64 Nelem, const cytnx_uint16 val) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] = cuCmul(ptr, make_cuDoubleComplex(val, 0));
      }
      __syncthreads();
    }
    __global__ void cuMul_rconst_kernel(cuDoubleComplex *out, const cuDoubleComplex *ptr,
                                        const cytnx_uint64 Nelem, const cytnx_uint16 val) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          cuCmul(ptr[blockIdx.x * blockDim.x + threadIdx.x], make_cuDoubleComplex(val, 0));
      }
      __syncthreads();
    }
    __global__ void cuMul_lconst_kernel(cuDoubleComplex *out, const cuDoubleComplex val,
                                        const cytnx_uint64 Nelem, const cytnx_uint16 *ptr) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          cuCmul(val, make_cuDoubleComplex(ptr[blockIdx.x * blockDim.x + threadIdx.x], 0));
      }
      __syncthreads();
    }
    __global__ void cuMul_tn_kernel(cuDoubleComplex *out, const cuDoubleComplex *ptr,
                                    const cytnx_uint64 Nelem, const cytnx_uint16 *val) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          cuCmul(ptr[blockIdx.x * blockDim.x + threadIdx.x],
                 make_cuDoubleComplex(val[blockIdx.x * blockDim.x + threadIdx.x], 0));
      }
      __syncthreads();
    }
    void cuMul_internal_cdtu16(boost::intrusive_ptr<Storage_base> &out,
                               boost::intrusive_ptr<Storage_base> &Lin,
                               boost::intrusive_ptr<Storage_base> &Rin,
                               const unsigned long long &len,
                               const std::vector<cytnx_uint64> &shape,
                               const std::vector<cytnx_uint64> &invmapper_L,
                               const std::vector<cytnx_uint64> &invmapper_R) {
      cuDoubleComplex *_out = (cuDoubleComplex *)out->Mem;
      cuDoubleComplex *_Lin = (cuDoubleComplex *)Lin->Mem;
      cytnx_uint16 *_Rin = (cytnx_uint16 *)Rin->Mem;

      cytnx_uint32 NBlocks = len / 512;
      if (len % 512) NBlocks += 1;

      if (Lin->size() == 1 and Rin->size() == 1) {
        cuMul_constconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin[0]);
      } else if (Lin->size() == 1) {
        cuMul_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMul_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cuMul_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }

    __global__ void cuMul_constconst_kernel(cuDoubleComplex *out, const cuDoubleComplex ptr,
                                            const cytnx_uint64 Nelem, const cytnx_bool val) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] = cuCmul(ptr, make_cuDoubleComplex(val, 0));
      }
      __syncthreads();
    }
    __global__ void cuMul_rconst_kernel(cuDoubleComplex *out, const cuDoubleComplex *ptr,
                                        const cytnx_uint64 Nelem, const cytnx_bool val) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          cuCmul(ptr[blockIdx.x * blockDim.x + threadIdx.x], make_cuDoubleComplex(val, 0));
      }
      __syncthreads();
    }
    __global__ void cuMul_lconst_kernel(cuDoubleComplex *out, const cuDoubleComplex val,
                                        const cytnx_uint64 Nelem, const cytnx_bool *ptr) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          cuCmul(val, make_cuDoubleComplex(ptr[blockIdx.x * blockDim.x + threadIdx.x], 0));
      }
      __syncthreads();
    }
    __global__ void cuMul_tn_kernel(cuDoubleComplex *out, const cuDoubleComplex *ptr,
                                    const cytnx_uint64 Nelem, const cytnx_bool *val) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          cuCmul(ptr[blockIdx.x * blockDim.x + threadIdx.x],
                 make_cuDoubleComplex(val[blockIdx.x * blockDim.x + threadIdx.x], 0));
      }
      __syncthreads();
    }
    void cuMul_internal_cdtb(boost::intrusive_ptr<Storage_base> &out,
                             boost::intrusive_ptr<Storage_base> &Lin,
                             boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                             const std::vector<cytnx_uint64> &shape,
                             const std::vector<cytnx_uint64> &invmapper_L,
                             const std::vector<cytnx_uint64> &invmapper_R) {
      cuDoubleComplex *_out = (cuDoubleComplex *)out->Mem;
      cuDoubleComplex *_Lin = (cuDoubleComplex *)Lin->Mem;
      cytnx_bool *_Rin = (cytnx_bool *)Rin->Mem;

      cytnx_uint32 NBlocks = len / 512;
      if (len % 512) NBlocks += 1;

      if (Lin->size() == 1 and Rin->size() == 1) {
        cuMul_constconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin[0]);
      } else if (Lin->size() == 1) {
        cuMul_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMul_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cuMul_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }

    //----------------------------
    void cuMul_internal_cftcd(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R) {
      cuMul_internal_cdtcf(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }

    __global__ void cuMul_constconst_kernel(cuFloatComplex *out, const cuFloatComplex ptr,
                                            const cytnx_uint64 Nelem, const cuFloatComplex val) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] = cuCmulf(ptr, val);
      }
      __syncthreads();
    }
    __global__ void cuMul_rconst_kernel(cuFloatComplex *out, const cuFloatComplex *ptr,
                                        const cytnx_uint64 Nelem, const cuFloatComplex val) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          cuCmulf(ptr[blockIdx.x * blockDim.x + threadIdx.x], val);
      }
      __syncthreads();
    }
    __global__ void cuMul_lconst_kernel(cuFloatComplex *out, const cuFloatComplex val,
                                        const cytnx_uint64 Nelem, const cuFloatComplex *ptr) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          cuCmulf(val, ptr[blockIdx.x * blockDim.x + threadIdx.x]);
      }
      __syncthreads();
    }
    __global__ void cuMul_tn_kernel(cuFloatComplex *out, const cuFloatComplex *ptr,
                                    const cytnx_uint64 Nelem, const cuFloatComplex *val) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] = cuCmulf(
          ptr[blockIdx.x * blockDim.x + threadIdx.x], val[blockIdx.x * blockDim.x + threadIdx.x]);
      }
      __syncthreads();
    }
    void cuMul_internal_cftcf(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R) {
      cuFloatComplex *_out = (cuFloatComplex *)out->Mem;
      cuFloatComplex *_Lin = (cuFloatComplex *)Lin->Mem;
      cuFloatComplex *_Rin = (cuFloatComplex *)Rin->Mem;

      cytnx_uint32 NBlocks = len / 512;
      if (len % 512) NBlocks += 1;

      if (Lin->size() == 1 and Rin->size() == 1) {
        cuMul_constconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin[0]);
      } else if (Lin->size() == 1) {
        cuMul_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMul_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cuMul_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }

    __global__ void cuMul_constconst_kernel(cuFloatComplex *out, const cuFloatComplex ptr,
                                            const cytnx_uint64 Nelem, const cytnx_double val) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] = cuCmulf(ptr, make_cuFloatComplex(val, 0));
      }
      __syncthreads();
    }
    __global__ void cuMul_rconst_kernel(cuFloatComplex *out, const cuFloatComplex *ptr,
                                        const cytnx_uint64 Nelem, const cytnx_double val) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          cuCmulf(ptr[blockIdx.x * blockDim.x + threadIdx.x], make_cuFloatComplex(val, 0));
      }
      __syncthreads();
    }
    __global__ void cuMul_lconst_kernel(cuFloatComplex *out, const cuFloatComplex val,
                                        const cytnx_uint64 Nelem, const cytnx_double *ptr) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          cuCmulf(val, make_cuFloatComplex(ptr[blockIdx.x * blockDim.x + threadIdx.x], 0));
      }
      __syncthreads();
    }
    __global__ void cuMul_tn_kernel(cuFloatComplex *out, const cuFloatComplex *ptr,
                                    const cytnx_uint64 Nelem, const cytnx_double *val) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          cuCmulf(ptr[blockIdx.x * blockDim.x + threadIdx.x],
                  make_cuFloatComplex(val[blockIdx.x * blockDim.x + threadIdx.x], 0));
      }
      __syncthreads();
    }
    void cuMul_internal_cftd(boost::intrusive_ptr<Storage_base> &out,
                             boost::intrusive_ptr<Storage_base> &Lin,
                             boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                             const std::vector<cytnx_uint64> &shape,
                             const std::vector<cytnx_uint64> &invmapper_L,
                             const std::vector<cytnx_uint64> &invmapper_R) {
      cuFloatComplex *_out = (cuFloatComplex *)out->Mem;
      cuFloatComplex *_Lin = (cuFloatComplex *)Lin->Mem;
      cytnx_double *_Rin = (cytnx_double *)Rin->Mem;

      cytnx_uint32 NBlocks = len / 512;
      if (len % 512) NBlocks += 1;

      if (Lin->size() == 1 and Rin->size() == 1) {
        cuMul_constconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin[0]);
      } else if (Lin->size() == 1) {
        cuMul_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMul_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cuMul_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }

    __global__ void cuMul_constconst_kernel(cuFloatComplex *out, const cuFloatComplex ptr,
                                            const cytnx_uint64 Nelem, const cytnx_float val) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] = cuCmulf(ptr, make_cuFloatComplex(val, 0));
      }
      __syncthreads();
    }
    __global__ void cuMul_rconst_kernel(cuFloatComplex *out, const cuFloatComplex *ptr,
                                        const cytnx_uint64 Nelem, const cytnx_float val) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          cuCmulf(ptr[blockIdx.x * blockDim.x + threadIdx.x], make_cuFloatComplex(val, 0));
      }
      __syncthreads();
    }
    __global__ void cuMul_lconst_kernel(cuFloatComplex *out, const cuFloatComplex val,
                                        const cytnx_uint64 Nelem, const cytnx_float *ptr) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          cuCmulf(val, make_cuFloatComplex(ptr[blockIdx.x * blockDim.x + threadIdx.x], 0));
      }
      __syncthreads();
    }
    __global__ void cuMul_tn_kernel(cuFloatComplex *out, const cuFloatComplex *ptr,
                                    const cytnx_uint64 Nelem, const cytnx_float *val) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          cuCmulf(ptr[blockIdx.x * blockDim.x + threadIdx.x],
                  make_cuFloatComplex(val[blockIdx.x * blockDim.x + threadIdx.x], 0));
      }
      __syncthreads();
    }
    void cuMul_internal_cftf(boost::intrusive_ptr<Storage_base> &out,
                             boost::intrusive_ptr<Storage_base> &Lin,
                             boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                             const std::vector<cytnx_uint64> &shape,
                             const std::vector<cytnx_uint64> &invmapper_L,
                             const std::vector<cytnx_uint64> &invmapper_R) {
      cuFloatComplex *_out = (cuFloatComplex *)out->Mem;
      cuFloatComplex *_Lin = (cuFloatComplex *)Lin->Mem;
      cytnx_float *_Rin = (cytnx_float *)Rin->Mem;

      cytnx_uint32 NBlocks = len / 512;
      if (len % 512) NBlocks += 1;

      if (Lin->size() == 1 and Rin->size() == 1) {
        cuMul_constconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin[0]);
      } else if (Lin->size() == 1) {
        cuMul_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMul_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cuMul_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }

    __global__ void cuMul_constconst_kernel(cuFloatComplex *out, const cuFloatComplex ptr,
                                            const cytnx_uint64 Nelem, const cytnx_uint64 val) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] = cuCmulf(ptr, make_cuFloatComplex(val, 0));
      }
      __syncthreads();
    }
    __global__ void cuMul_rconst_kernel(cuFloatComplex *out, const cuFloatComplex *ptr,
                                        const cytnx_uint64 Nelem, const cytnx_uint64 val) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          cuCmulf(ptr[blockIdx.x * blockDim.x + threadIdx.x], make_cuFloatComplex(val, 0));
      }
      __syncthreads();
    }
    __global__ void cuMul_lconst_kernel(cuFloatComplex *out, const cuFloatComplex val,
                                        const cytnx_uint64 Nelem, const cytnx_uint64 *ptr) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          cuCmulf(val, make_cuFloatComplex(ptr[blockIdx.x * blockDim.x + threadIdx.x], 0));
      }
      __syncthreads();
    }
    __global__ void cuMul_tn_kernel(cuFloatComplex *out, const cuFloatComplex *ptr,
                                    const cytnx_uint64 Nelem, const cytnx_uint64 *val) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          cuCmulf(ptr[blockIdx.x * blockDim.x + threadIdx.x],
                  make_cuFloatComplex(val[blockIdx.x * blockDim.x + threadIdx.x], 0));
      }
      __syncthreads();
    }
    void cuMul_internal_cftu64(boost::intrusive_ptr<Storage_base> &out,
                               boost::intrusive_ptr<Storage_base> &Lin,
                               boost::intrusive_ptr<Storage_base> &Rin,
                               const unsigned long long &len,
                               const std::vector<cytnx_uint64> &shape,
                               const std::vector<cytnx_uint64> &invmapper_L,
                               const std::vector<cytnx_uint64> &invmapper_R) {
      cuFloatComplex *_out = (cuFloatComplex *)out->Mem;
      cuFloatComplex *_Lin = (cuFloatComplex *)Lin->Mem;
      cytnx_uint64 *_Rin = (cytnx_uint64 *)Rin->Mem;

      cytnx_uint32 NBlocks = len / 512;
      if (len % 512) NBlocks += 1;

      if (Lin->size() == 1 and Rin->size() == 1) {
        cuMul_constconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin[0]);
      } else if (Lin->size() == 1) {
        cuMul_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMul_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cuMul_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }

    __global__ void cuMul_constconst_kernel(cuFloatComplex *out, const cuFloatComplex ptr,
                                            const cytnx_uint64 Nelem, const cytnx_uint32 val) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] = cuCmulf(ptr, make_cuFloatComplex(val, 0));
      }
      __syncthreads();
    }
    __global__ void cuMul_rconst_kernel(cuFloatComplex *out, const cuFloatComplex *ptr,
                                        const cytnx_uint64 Nelem, const cytnx_uint32 val) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          cuCmulf(ptr[blockIdx.x * blockDim.x + threadIdx.x], make_cuFloatComplex(val, 0));
      }
      __syncthreads();
    }
    __global__ void cuMul_lconst_kernel(cuFloatComplex *out, const cuFloatComplex val,
                                        const cytnx_uint64 Nelem, const cytnx_uint32 *ptr) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          cuCmulf(val, make_cuFloatComplex(ptr[blockIdx.x * blockDim.x + threadIdx.x], 0));
      }
      __syncthreads();
    }
    __global__ void cuMul_tn_kernel(cuFloatComplex *out, const cuFloatComplex *ptr,
                                    const cytnx_uint64 Nelem, const cytnx_uint32 *val) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          cuCmulf(ptr[blockIdx.x * blockDim.x + threadIdx.x],
                  make_cuFloatComplex(val[blockIdx.x * blockDim.x + threadIdx.x], 0));
      }
      __syncthreads();
    }
    void cuMul_internal_cftu32(boost::intrusive_ptr<Storage_base> &out,
                               boost::intrusive_ptr<Storage_base> &Lin,
                               boost::intrusive_ptr<Storage_base> &Rin,
                               const unsigned long long &len,
                               const std::vector<cytnx_uint64> &shape,
                               const std::vector<cytnx_uint64> &invmapper_L,
                               const std::vector<cytnx_uint64> &invmapper_R) {
      cuFloatComplex *_out = (cuFloatComplex *)out->Mem;
      cuFloatComplex *_Lin = (cuFloatComplex *)Lin->Mem;
      cytnx_uint32 *_Rin = (cytnx_uint32 *)Rin->Mem;

      cytnx_uint32 NBlocks = len / 512;
      if (len % 512) NBlocks += 1;

      if (Lin->size() == 1 and Rin->size() == 1) {
        cuMul_constconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin[0]);
      } else if (Lin->size() == 1) {
        cuMul_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMul_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cuMul_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }

    __global__ void cuMul_constconst_kernel(cuFloatComplex *out, const cuFloatComplex ptr,
                                            const cytnx_uint64 Nelem, const cytnx_int64 val) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] = cuCmulf(ptr, make_cuFloatComplex(val, 0));
      }
      __syncthreads();
    }
    __global__ void cuMul_rconst_kernel(cuFloatComplex *out, const cuFloatComplex *ptr,
                                        const cytnx_uint64 Nelem, const cytnx_int64 val) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          cuCmulf(ptr[blockIdx.x * blockDim.x + threadIdx.x], make_cuFloatComplex(val, 0));
      }
      __syncthreads();
    }
    __global__ void cuMul_lconst_kernel(cuFloatComplex *out, const cuFloatComplex val,
                                        const cytnx_uint64 Nelem, const cytnx_int64 *ptr) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          cuCmulf(val, make_cuFloatComplex(ptr[blockIdx.x * blockDim.x + threadIdx.x], 0));
      }
      __syncthreads();
    }
    __global__ void cuMul_tn_kernel(cuFloatComplex *out, const cuFloatComplex *ptr,
                                    const cytnx_uint64 Nelem, const cytnx_int64 *val) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          cuCmulf(ptr[blockIdx.x * blockDim.x + threadIdx.x],
                  make_cuFloatComplex(val[blockIdx.x * blockDim.x + threadIdx.x], 0));
      }
      __syncthreads();
    }
    void cuMul_internal_cfti64(boost::intrusive_ptr<Storage_base> &out,
                               boost::intrusive_ptr<Storage_base> &Lin,
                               boost::intrusive_ptr<Storage_base> &Rin,
                               const unsigned long long &len,
                               const std::vector<cytnx_uint64> &shape,
                               const std::vector<cytnx_uint64> &invmapper_L,
                               const std::vector<cytnx_uint64> &invmapper_R) {
      cuFloatComplex *_out = (cuFloatComplex *)out->Mem;
      cuFloatComplex *_Lin = (cuFloatComplex *)Lin->Mem;
      cytnx_int64 *_Rin = (cytnx_int64 *)Rin->Mem;

      cytnx_uint32 NBlocks = len / 512;
      if (len % 512) NBlocks += 1;

      if (Lin->size() == 1 and Rin->size() == 1) {
        cuMul_constconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin[0]);
      } else if (Lin->size() == 1) {
        cuMul_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMul_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cuMul_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }

    __global__ void cuMul_constconst_kernel(cuFloatComplex *out, const cuFloatComplex ptr,
                                            const cytnx_uint64 Nelem, const cytnx_int32 val) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] = cuCmulf(ptr, make_cuFloatComplex(val, 0));
      }
      __syncthreads();
    }
    __global__ void cuMul_rconst_kernel(cuFloatComplex *out, const cuFloatComplex *ptr,
                                        const cytnx_uint64 Nelem, const cytnx_int32 val) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          cuCmulf(ptr[blockIdx.x * blockDim.x + threadIdx.x], make_cuFloatComplex(val, 0));
      }
      __syncthreads();
    }
    __global__ void cuMul_lconst_kernel(cuFloatComplex *out, const cuFloatComplex val,
                                        const cytnx_uint64 Nelem, const cytnx_int32 *ptr) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          cuCmulf(val, make_cuFloatComplex(ptr[blockIdx.x * blockDim.x + threadIdx.x], 0));
      }
      __syncthreads();
    }
    __global__ void cuMul_tn_kernel(cuFloatComplex *out, const cuFloatComplex *ptr,
                                    const cytnx_uint64 Nelem, const cytnx_int32 *val) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          cuCmulf(ptr[blockIdx.x * blockDim.x + threadIdx.x],
                  make_cuFloatComplex(val[blockIdx.x * blockDim.x + threadIdx.x], 0));
      }
      __syncthreads();
    }
    void cuMul_internal_cfti32(boost::intrusive_ptr<Storage_base> &out,
                               boost::intrusive_ptr<Storage_base> &Lin,
                               boost::intrusive_ptr<Storage_base> &Rin,
                               const unsigned long long &len,
                               const std::vector<cytnx_uint64> &shape,
                               const std::vector<cytnx_uint64> &invmapper_L,
                               const std::vector<cytnx_uint64> &invmapper_R) {
      cuFloatComplex *_out = (cuFloatComplex *)out->Mem;
      cuFloatComplex *_Lin = (cuFloatComplex *)Lin->Mem;
      cytnx_int32 *_Rin = (cytnx_int32 *)Rin->Mem;

      cytnx_uint32 NBlocks = len / 512;
      if (len % 512) NBlocks += 1;

      if (Lin->size() == 1 and Rin->size() == 1) {
        cuMul_constconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin[0]);
      } else if (Lin->size() == 1) {
        cuMul_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMul_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cuMul_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }

    __global__ void cuMul_constconst_kernel(cuFloatComplex *out, const cuFloatComplex ptr,
                                            const cytnx_uint64 Nelem, const cytnx_int16 val) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] = cuCmulf(ptr, make_cuFloatComplex(val, 0));
      }
      __syncthreads();
    }
    __global__ void cuMul_rconst_kernel(cuFloatComplex *out, const cuFloatComplex *ptr,
                                        const cytnx_uint64 Nelem, const cytnx_int16 val) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          cuCmulf(ptr[blockIdx.x * blockDim.x + threadIdx.x], make_cuFloatComplex(val, 0));
      }
      __syncthreads();
    }
    __global__ void cuMul_lconst_kernel(cuFloatComplex *out, const cuFloatComplex val,
                                        const cytnx_uint64 Nelem, const cytnx_int16 *ptr) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          cuCmulf(val, make_cuFloatComplex(ptr[blockIdx.x * blockDim.x + threadIdx.x], 0));
      }
      __syncthreads();
    }
    __global__ void cuMul_tn_kernel(cuFloatComplex *out, const cuFloatComplex *ptr,
                                    const cytnx_uint64 Nelem, const cytnx_int16 *val) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          cuCmulf(ptr[blockIdx.x * blockDim.x + threadIdx.x],
                  make_cuFloatComplex(val[blockIdx.x * blockDim.x + threadIdx.x], 0));
      }
      __syncthreads();
    }
    void cuMul_internal_cfti16(boost::intrusive_ptr<Storage_base> &out,
                               boost::intrusive_ptr<Storage_base> &Lin,
                               boost::intrusive_ptr<Storage_base> &Rin,
                               const unsigned long long &len,
                               const std::vector<cytnx_uint64> &shape,
                               const std::vector<cytnx_uint64> &invmapper_L,
                               const std::vector<cytnx_uint64> &invmapper_R) {
      cuFloatComplex *_out = (cuFloatComplex *)out->Mem;
      cuFloatComplex *_Lin = (cuFloatComplex *)Lin->Mem;
      cytnx_int16 *_Rin = (cytnx_int16 *)Rin->Mem;

      cytnx_uint32 NBlocks = len / 512;
      if (len % 512) NBlocks += 1;

      if (Lin->size() == 1 and Rin->size() == 1) {
        cuMul_constconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin[0]);
      } else if (Lin->size() == 1) {
        cuMul_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMul_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cuMul_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }

    __global__ void cuMul_constconst_kernel(cuFloatComplex *out, const cuFloatComplex ptr,
                                            const cytnx_uint64 Nelem, const cytnx_uint16 val) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] = cuCmulf(ptr, make_cuFloatComplex(val, 0));
      }
      __syncthreads();
    }
    __global__ void cuMul_rconst_kernel(cuFloatComplex *out, const cuFloatComplex *ptr,
                                        const cytnx_uint64 Nelem, const cytnx_uint16 val) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          cuCmulf(ptr[blockIdx.x * blockDim.x + threadIdx.x], make_cuFloatComplex(val, 0));
      }
      __syncthreads();
    }
    __global__ void cuMul_lconst_kernel(cuFloatComplex *out, const cuFloatComplex val,
                                        const cytnx_uint64 Nelem, const cytnx_uint16 *ptr) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          cuCmulf(val, make_cuFloatComplex(ptr[blockIdx.x * blockDim.x + threadIdx.x], 0));
      }
      __syncthreads();
    }
    __global__ void cuMul_tn_kernel(cuFloatComplex *out, const cuFloatComplex *ptr,
                                    const cytnx_uint64 Nelem, const cytnx_uint16 *val) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          cuCmulf(ptr[blockIdx.x * blockDim.x + threadIdx.x],
                  make_cuFloatComplex(val[blockIdx.x * blockDim.x + threadIdx.x], 0));
      }
      __syncthreads();
    }
    void cuMul_internal_cftu16(boost::intrusive_ptr<Storage_base> &out,
                               boost::intrusive_ptr<Storage_base> &Lin,
                               boost::intrusive_ptr<Storage_base> &Rin,
                               const unsigned long long &len,
                               const std::vector<cytnx_uint64> &shape,
                               const std::vector<cytnx_uint64> &invmapper_L,
                               const std::vector<cytnx_uint64> &invmapper_R) {
      cuFloatComplex *_out = (cuFloatComplex *)out->Mem;
      cuFloatComplex *_Lin = (cuFloatComplex *)Lin->Mem;
      cytnx_uint16 *_Rin = (cytnx_uint16 *)Rin->Mem;

      cytnx_uint32 NBlocks = len / 512;
      if (len % 512) NBlocks += 1;

      if (Lin->size() == 1 and Rin->size() == 1) {
        cuMul_constconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin[0]);
      } else if (Lin->size() == 1) {
        cuMul_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMul_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cuMul_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }

    __global__ void cuMul_constconst_kernel(cuFloatComplex *out, const cuFloatComplex ptr,
                                            const cytnx_uint64 Nelem, const cytnx_bool val) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] = cuCmulf(ptr, make_cuFloatComplex(val, 0));
      }
      __syncthreads();
    }
    __global__ void cuMul_rconst_kernel(cuFloatComplex *out, const cuFloatComplex *ptr,
                                        const cytnx_uint64 Nelem, const cytnx_bool val) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          cuCmulf(ptr[blockIdx.x * blockDim.x + threadIdx.x], make_cuFloatComplex(val, 0));
      }
      __syncthreads();
    }
    __global__ void cuMul_lconst_kernel(cuFloatComplex *out, const cuFloatComplex val,
                                        const cytnx_uint64 Nelem, const cytnx_bool *ptr) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          cuCmulf(val, make_cuFloatComplex(ptr[blockIdx.x * blockDim.x + threadIdx.x], 0));
      }
      __syncthreads();
    }
    __global__ void cuMul_tn_kernel(cuFloatComplex *out, const cuFloatComplex *ptr,
                                    const cytnx_uint64 Nelem, const cytnx_bool *val) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          cuCmulf(ptr[blockIdx.x * blockDim.x + threadIdx.x],
                  make_cuFloatComplex(val[blockIdx.x * blockDim.x + threadIdx.x], 0));
      }
      __syncthreads();
    }
    void cuMul_internal_cftb(boost::intrusive_ptr<Storage_base> &out,
                             boost::intrusive_ptr<Storage_base> &Lin,
                             boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                             const std::vector<cytnx_uint64> &shape,
                             const std::vector<cytnx_uint64> &invmapper_L,
                             const std::vector<cytnx_uint64> &invmapper_R) {
      cuFloatComplex *_out = (cuFloatComplex *)out->Mem;
      cuFloatComplex *_Lin = (cuFloatComplex *)Lin->Mem;
      cytnx_bool *_Rin = (cytnx_bool *)Rin->Mem;

      cytnx_uint32 NBlocks = len / 512;
      if (len % 512) NBlocks += 1;

      if (Lin->size() == 1 and Rin->size() == 1) {
        cuMul_constconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin[0]);
      } else if (Lin->size() == 1) {
        cuMul_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMul_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cuMul_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }

    //-------------------------
    void cuMul_internal_dtcd(boost::intrusive_ptr<Storage_base> &out,
                             boost::intrusive_ptr<Storage_base> &Lin,
                             boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                             const std::vector<cytnx_uint64> &shape,
                             const std::vector<cytnx_uint64> &invmapper_L,
                             const std::vector<cytnx_uint64> &invmapper_R) {
      cuMul_internal_cdtd(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void cuMul_internal_dtcf(boost::intrusive_ptr<Storage_base> &out,
                             boost::intrusive_ptr<Storage_base> &Lin,
                             boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                             const std::vector<cytnx_uint64> &shape,
                             const std::vector<cytnx_uint64> &invmapper_L,
                             const std::vector<cytnx_uint64> &invmapper_R) {
      cuMul_internal_cftd(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }

    void cuMul_internal_dtd(boost::intrusive_ptr<Storage_base> &out,
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

      if (Lin->size() == 1 and Rin->size() == 1) {
        cuMul_constconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin[0]);
      } else if (Lin->size() == 1) {
        cuMul_lconst_kernel<<<NBlocks, 512>>>(_out, *_Lin, len, _Rin);
      } else if (Rin->size() == 1) {
        cuMul_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cuMul_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }
    void cuMul_internal_dtf(boost::intrusive_ptr<Storage_base> &out,
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

      if (Lin->size() == 1 and Rin->size() == 1) {
        cuMul_constconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin[0]);
      } else if (Lin->size() == 1) {
        cuMul_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMul_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cuMul_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }
    void cuMul_internal_dtu64(boost::intrusive_ptr<Storage_base> &out,
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

      if (Lin->size() == 1 and Rin->size() == 1) {
        cuMul_constconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin[0]);
      } else if (Lin->size() == 1) {
        cuMul_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMul_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cuMul_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }
    void cuMul_internal_dtu32(boost::intrusive_ptr<Storage_base> &out,
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

      if (Lin->size() == 1 and Rin->size() == 1) {
        cuMul_constconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin[0]);
      } else if (Lin->size() == 1) {
        cuMul_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMul_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cuMul_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }
    void cuMul_internal_dti64(boost::intrusive_ptr<Storage_base> &out,
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

      if (Lin->size() == 1 and Rin->size() == 1) {
        cuMul_constconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin[0]);
      } else if (Lin->size() == 1) {
        cuMul_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMul_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cuMul_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }
    void cuMul_internal_dti32(boost::intrusive_ptr<Storage_base> &out,
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

      if (Lin->size() == 1 and Rin->size() == 1) {
        cuMul_constconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin[0]);
      } else if (Lin->size() == 1) {
        cuMul_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMul_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cuMul_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }
    void cuMul_internal_dti16(boost::intrusive_ptr<Storage_base> &out,
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

      if (Lin->size() == 1 and Rin->size() == 1) {
        cuMul_constconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin[0]);
      } else if (Lin->size() == 1) {
        cuMul_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMul_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cuMul_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }
    void cuMul_internal_dtu16(boost::intrusive_ptr<Storage_base> &out,
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

      if (Lin->size() == 1 and Rin->size() == 1) {
        cuMul_constconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin[0]);
      } else if (Lin->size() == 1) {
        cuMul_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMul_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cuMul_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }

    __global__ void cuMul_constconst_kernel(cytnx_double *out, const cytnx_double ptr,
                                            const cytnx_uint64 Nelem, const cytnx_bool val) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] = ptr * cytnx_double(val);
      }
      __syncthreads();
    }
    __global__ void cuMul_lconst_kernel(cytnx_double *out, const cytnx_double val,
                                        const cytnx_uint64 Nelem, const cytnx_bool *ptr) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          val * cytnx_double(ptr[blockIdx.x * blockDim.x + threadIdx.x]);
      }
      __syncthreads();
    }

    __global__ void cuMul_tn_kernel(cytnx_double *out, const cytnx_double *val,
                                    const cytnx_uint64 Nelem, const cytnx_bool *ptr) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          val[blockIdx.x * blockDim.x + threadIdx.x] *
          cytnx_double(ptr[blockIdx.x * blockDim.x + threadIdx.x]);
      }
      __syncthreads();
    }
    void cuMul_internal_dtb(boost::intrusive_ptr<Storage_base> &out,
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

      if (Lin->size() == 1 and Rin->size() == 1) {
        cuMul_constconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin[0]);
      } else if (Lin->size() == 1) {
        cuMul_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMul_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, double(_Rin[0]));
      } else {
        cuMul_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }

    //----------------------------
    void cuMul_internal_ftcd(boost::intrusive_ptr<Storage_base> &out,
                             boost::intrusive_ptr<Storage_base> &Lin,
                             boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                             const std::vector<cytnx_uint64> &shape,
                             const std::vector<cytnx_uint64> &invmapper_L,
                             const std::vector<cytnx_uint64> &invmapper_R) {
      cuMul_internal_cdtf(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void cuMul_internal_ftcf(boost::intrusive_ptr<Storage_base> &out,
                             boost::intrusive_ptr<Storage_base> &Lin,
                             boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                             const std::vector<cytnx_uint64> &shape,
                             const std::vector<cytnx_uint64> &invmapper_L,
                             const std::vector<cytnx_uint64> &invmapper_R) {
      cuMul_internal_cftf(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void cuMul_internal_ftd(boost::intrusive_ptr<Storage_base> &out,
                            boost::intrusive_ptr<Storage_base> &Lin,
                            boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                            const std::vector<cytnx_uint64> &shape,
                            const std::vector<cytnx_uint64> &invmapper_L,
                            const std::vector<cytnx_uint64> &invmapper_R) {
      cuMul_internal_dtf(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void cuMul_internal_ftf(boost::intrusive_ptr<Storage_base> &out,
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

      if (Lin->size() == 1 and Rin->size() == 1) {
        cuMul_constconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin[0]);
      } else if (Lin->size() == 1) {
        cuMul_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMul_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cuMul_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }
    void cuMul_internal_ftu64(boost::intrusive_ptr<Storage_base> &out,
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

      if (Lin->size() == 1 and Rin->size() == 1) {
        cuMul_constconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin[0]);
      } else if (Lin->size() == 1) {
        cuMul_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMul_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cuMul_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }
    void cuMul_internal_ftu32(boost::intrusive_ptr<Storage_base> &out,
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

      if (Lin->size() == 1 and Rin->size() == 1) {
        cuMul_constconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin[0]);
      } else if (Lin->size() == 1) {
        cuMul_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMul_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cuMul_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }
    void cuMul_internal_fti64(boost::intrusive_ptr<Storage_base> &out,
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

      if (Lin->size() == 1 and Rin->size() == 1) {
        cuMul_constconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin[0]);
      } else if (Lin->size() == 1) {
        cuMul_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMul_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cuMul_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }
    void cuMul_internal_fti32(boost::intrusive_ptr<Storage_base> &out,
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

      if (Lin->size() == 1 and Rin->size() == 1) {
        cuMul_constconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin[0]);
      } else if (Lin->size() == 1) {
        cuMul_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMul_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cuMul_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }
    void cuMul_internal_fti16(boost::intrusive_ptr<Storage_base> &out,
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

      if (Lin->size() == 1 and Rin->size() == 1) {
        cuMul_constconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin[0]);
      } else if (Lin->size() == 1) {
        cuMul_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMul_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cuMul_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }
    void cuMul_internal_ftu16(boost::intrusive_ptr<Storage_base> &out,
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

      if (Lin->size() == 1 and Rin->size() == 1) {
        cuMul_constconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin[0]);
      } else if (Lin->size() == 1) {
        cuMul_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMul_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cuMul_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }

    __global__ void cuMul_constconst_kernel(cytnx_float *out, const cytnx_float ptr,
                                            const cytnx_uint64 Nelem, const cytnx_bool val) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] = ptr * cytnx_float(val);
      }
      __syncthreads();
    }
    __global__ void cuMul_lconst_kernel(cytnx_float *out, const cytnx_float val,
                                        const cytnx_uint64 Nelem, const cytnx_bool *ptr) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          val * cytnx_float(ptr[blockIdx.x * blockDim.x + threadIdx.x]);
      }
      __syncthreads();
    }

    __global__ void cuMul_tn_kernel(cytnx_float *out, const cytnx_float *val,
                                    const cytnx_uint64 Nelem, const cytnx_bool *ptr) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          val[blockIdx.x * blockDim.x + threadIdx.x] *
          cytnx_float(ptr[blockIdx.x * blockDim.x + threadIdx.x]);
      }
      __syncthreads();
    }
    void cuMul_internal_ftb(boost::intrusive_ptr<Storage_base> &out,
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

      if (Lin->size() == 1 and Rin->size() == 1) {
        cuMul_constconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin[0]);
      } else if (Lin->size() == 1) {
        cuMul_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMul_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, float(_Rin[0]));
      } else {
        cuMul_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }
    //-----------------------------
    void cuMul_internal_i64tcd(boost::intrusive_ptr<Storage_base> &out,
                               boost::intrusive_ptr<Storage_base> &Lin,
                               boost::intrusive_ptr<Storage_base> &Rin,
                               const unsigned long long &len,
                               const std::vector<cytnx_uint64> &shape,
                               const std::vector<cytnx_uint64> &invmapper_L,
                               const std::vector<cytnx_uint64> &invmapper_R) {
      cuMul_internal_cdti64(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void cuMul_internal_i64tcf(boost::intrusive_ptr<Storage_base> &out,
                               boost::intrusive_ptr<Storage_base> &Lin,
                               boost::intrusive_ptr<Storage_base> &Rin,
                               const unsigned long long &len,
                               const std::vector<cytnx_uint64> &shape,
                               const std::vector<cytnx_uint64> &invmapper_L,
                               const std::vector<cytnx_uint64> &invmapper_R) {
      cuMul_internal_cfti64(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void cuMul_internal_i64td(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R) {
      cuMul_internal_dti64(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void cuMul_internal_i64tf(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R) {
      cuMul_internal_fti64(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void cuMul_internal_i64ti64(boost::intrusive_ptr<Storage_base> &out,
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

      if (Lin->size() == 1 and Rin->size() == 1) {
        cuMul_constconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin[0]);
      } else if (Lin->size() == 1) {
        cuMul_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMul_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cuMul_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }
    void cuMul_internal_i64tu64(boost::intrusive_ptr<Storage_base> &out,
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

      if (Lin->size() == 1 and Rin->size() == 1) {
        cuMul_constconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin[0]);
      } else if (Lin->size() == 1) {
        cuMul_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMul_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cuMul_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }
    void cuMul_internal_i64ti32(boost::intrusive_ptr<Storage_base> &out,
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

      if (Lin->size() == 1 and Rin->size() == 1) {
        cuMul_constconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin[0]);
      } else if (Lin->size() == 1) {
        cuMul_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMul_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cuMul_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }
    void cuMul_internal_i64tu32(boost::intrusive_ptr<Storage_base> &out,
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

      if (Lin->size() == 1 and Rin->size() == 1) {
        cuMul_constconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin[0]);
      } else if (Lin->size() == 1) {
        cuMul_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMul_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cuMul_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }
    void cuMul_internal_i64ti16(boost::intrusive_ptr<Storage_base> &out,
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

      if (Lin->size() == 1 and Rin->size() == 1) {
        cuMul_constconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin[0]);
      } else if (Lin->size() == 1) {
        cuMul_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMul_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cuMul_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }
    void cuMul_internal_i64tu16(boost::intrusive_ptr<Storage_base> &out,
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

      if (Lin->size() == 1 and Rin->size() == 1) {
        cuMul_constconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin[0]);
      } else if (Lin->size() == 1) {
        cuMul_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMul_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cuMul_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }

    __global__ void cuMul_constconst_kernel(cytnx_int64 *out, const cytnx_int64 ptr,
                                            const cytnx_uint64 Nelem, const cytnx_bool val) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] = ptr * cytnx_int64(val);
      }
      __syncthreads();
    }
    __global__ void cuMul_lconst_kernel(cytnx_int64 *out, const cytnx_int64 val,
                                        const cytnx_uint64 Nelem, const cytnx_bool *ptr) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          val * cytnx_int64(ptr[blockIdx.x * blockDim.x + threadIdx.x]);
      }
      __syncthreads();
    }

    __global__ void cuMul_tn_kernel(cytnx_int64 *out, const cytnx_int64 *val,
                                    const cytnx_uint64 Nelem, const cytnx_bool *ptr) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          val[blockIdx.x * blockDim.x + threadIdx.x] *
          cytnx_int64(ptr[blockIdx.x * blockDim.x + threadIdx.x]);
      }
      __syncthreads();
    }
    void cuMul_internal_i64tb(boost::intrusive_ptr<Storage_base> &out,
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

      if (Lin->size() == 1 and Rin->size() == 1) {
        cuMul_constconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin[0]);
      } else if (Lin->size() == 1) {
        cuMul_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMul_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, cytnx_int64(_Rin[0]));
      } else {
        cuMul_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }

    //------------------------------
    void cuMul_internal_u64tcd(boost::intrusive_ptr<Storage_base> &out,
                               boost::intrusive_ptr<Storage_base> &Lin,
                               boost::intrusive_ptr<Storage_base> &Rin,
                               const unsigned long long &len,
                               const std::vector<cytnx_uint64> &shape,
                               const std::vector<cytnx_uint64> &invmapper_L,
                               const std::vector<cytnx_uint64> &invmapper_R) {
      cuMul_internal_cdtu64(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void cuMul_internal_u64tcf(boost::intrusive_ptr<Storage_base> &out,
                               boost::intrusive_ptr<Storage_base> &Lin,
                               boost::intrusive_ptr<Storage_base> &Rin,
                               const unsigned long long &len,
                               const std::vector<cytnx_uint64> &shape,
                               const std::vector<cytnx_uint64> &invmapper_L,
                               const std::vector<cytnx_uint64> &invmapper_R) {
      cuMul_internal_cftu64(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void cuMul_internal_u64td(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R) {
      cuMul_internal_dtu64(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void cuMul_internal_u64tf(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R) {
      cuMul_internal_ftu64(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void cuMul_internal_u64ti64(boost::intrusive_ptr<Storage_base> &out,
                                boost::intrusive_ptr<Storage_base> &Lin,
                                boost::intrusive_ptr<Storage_base> &Rin,
                                const unsigned long long &len,
                                const std::vector<cytnx_uint64> &shape,
                                const std::vector<cytnx_uint64> &invmapper_L,
                                const std::vector<cytnx_uint64> &invmapper_R) {
      cuMul_internal_i64tu64(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void cuMul_internal_u64tu64(boost::intrusive_ptr<Storage_base> &out,
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

      if (Lin->size() == 1 and Rin->size() == 1) {
        cuMul_constconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin[0]);
      } else if (Lin->size() == 1) {
        cuMul_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMul_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cuMul_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }
    void cuMul_internal_u64ti32(boost::intrusive_ptr<Storage_base> &out,
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

      if (Lin->size() == 1 and Rin->size() == 1) {
        cuMul_constconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin[0]);
      } else if (Lin->size() == 1) {
        cuMul_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMul_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cuMul_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }
    void cuMul_internal_u64tu32(boost::intrusive_ptr<Storage_base> &out,
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

      if (Lin->size() == 1 and Rin->size() == 1) {
        cuMul_constconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin[0]);
      } else if (Lin->size() == 1) {
        cuMul_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMul_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cuMul_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }
    void cuMul_internal_u64ti16(boost::intrusive_ptr<Storage_base> &out,
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

      if (Lin->size() == 1 and Rin->size() == 1) {
        cuMul_constconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin[0]);
      } else if (Lin->size() == 1) {
        cuMul_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMul_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cuMul_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }
    void cuMul_internal_u64tu16(boost::intrusive_ptr<Storage_base> &out,
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

      if (Lin->size() == 1 and Rin->size() == 1) {
        cuMul_constconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin[0]);
      } else if (Lin->size() == 1) {
        cuMul_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMul_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cuMul_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }

    __global__ void cuMul_constconst_kernel(cytnx_uint64 *out, const cytnx_uint64 ptr,
                                            const cytnx_uint64 Nelem, const cytnx_bool val) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] = ptr * cytnx_uint64(val);
      }
      __syncthreads();
    }
    __global__ void cuMul_lconst_kernel(cytnx_uint64 *out, const cytnx_uint64 val,
                                        const cytnx_uint64 Nelem, const cytnx_bool *ptr) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          val * cytnx_uint64(ptr[blockIdx.x * blockDim.x + threadIdx.x]);
      }
      __syncthreads();
    }

    __global__ void cuMul_tn_kernel(cytnx_uint64 *out, const cytnx_uint64 *val,
                                    const cytnx_uint64 Nelem, const cytnx_bool *ptr) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          val[blockIdx.x * blockDim.x + threadIdx.x] *
          cytnx_uint64(ptr[blockIdx.x * blockDim.x + threadIdx.x]);
      }
      __syncthreads();
    }
    void cuMul_internal_u64tb(boost::intrusive_ptr<Storage_base> &out,
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

      if (Lin->size() == 1 and Rin->size() == 1) {
        cuMul_constconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin[0]);
      } else if (Lin->size() == 1) {
        cuMul_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMul_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, cytnx_uint64(_Rin[0]));
      } else {
        cuMul_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }

    //-----------------------------------------
    void cuMul_internal_i32tcd(boost::intrusive_ptr<Storage_base> &out,
                               boost::intrusive_ptr<Storage_base> &Lin,
                               boost::intrusive_ptr<Storage_base> &Rin,
                               const unsigned long long &len,
                               const std::vector<cytnx_uint64> &shape,
                               const std::vector<cytnx_uint64> &invmapper_L,
                               const std::vector<cytnx_uint64> &invmapper_R) {
      cuMul_internal_cdti32(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void cuMul_internal_i32tcf(boost::intrusive_ptr<Storage_base> &out,
                               boost::intrusive_ptr<Storage_base> &Lin,
                               boost::intrusive_ptr<Storage_base> &Rin,
                               const unsigned long long &len,
                               const std::vector<cytnx_uint64> &shape,
                               const std::vector<cytnx_uint64> &invmapper_L,
                               const std::vector<cytnx_uint64> &invmapper_R) {
      cuMul_internal_cfti32(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void cuMul_internal_i32td(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R) {
      cuMul_internal_dti32(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void cuMul_internal_i32tf(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R) {
      cuMul_internal_fti32(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void cuMul_internal_i32ti64(boost::intrusive_ptr<Storage_base> &out,
                                boost::intrusive_ptr<Storage_base> &Lin,
                                boost::intrusive_ptr<Storage_base> &Rin,
                                const unsigned long long &len,
                                const std::vector<cytnx_uint64> &shape,
                                const std::vector<cytnx_uint64> &invmapper_L,
                                const std::vector<cytnx_uint64> &invmapper_R) {
      cuMul_internal_i64ti32(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void cuMul_internal_i32tu64(boost::intrusive_ptr<Storage_base> &out,
                                boost::intrusive_ptr<Storage_base> &Lin,
                                boost::intrusive_ptr<Storage_base> &Rin,
                                const unsigned long long &len,
                                const std::vector<cytnx_uint64> &shape,
                                const std::vector<cytnx_uint64> &invmapper_L,
                                const std::vector<cytnx_uint64> &invmapper_R) {
      cuMul_internal_u64ti32(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void cuMul_internal_i32ti32(boost::intrusive_ptr<Storage_base> &out,
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

      if (Lin->size() == 1 and Rin->size() == 1) {
        cuMul_constconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin[0]);
      } else if (Lin->size() == 1) {
        cuMul_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMul_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cuMul_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }
    void cuMul_internal_i32tu32(boost::intrusive_ptr<Storage_base> &out,
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

      if (Lin->size() == 1 and Rin->size() == 1) {
        cuMul_constconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin[0]);
      } else if (Lin->size() == 1) {
        cuMul_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMul_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cuMul_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }
    void cuMul_internal_i32ti16(boost::intrusive_ptr<Storage_base> &out,
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

      if (Lin->size() == 1 and Rin->size() == 1) {
        cuMul_constconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin[0]);
      } else if (Lin->size() == 1) {
        cuMul_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMul_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cuMul_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }
    void cuMul_internal_i32tu16(boost::intrusive_ptr<Storage_base> &out,
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

      if (Lin->size() == 1 and Rin->size() == 1) {
        cuMul_constconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin[0]);
      } else if (Lin->size() == 1) {
        cuMul_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMul_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cuMul_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }

    __global__ void cuMul_constconst_kernel(cytnx_int32 *out, const cytnx_int32 ptr,
                                            const cytnx_uint64 Nelem, const cytnx_bool val) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] = ptr * cytnx_int32(val);
      }
      __syncthreads();
    }
    __global__ void cuMul_lconst_kernel(cytnx_int32 *out, const cytnx_int32 val,
                                        const cytnx_uint64 Nelem, const cytnx_bool *ptr) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          val * cytnx_int32(ptr[blockIdx.x * blockDim.x + threadIdx.x]);
      }
      __syncthreads();
    }

    __global__ void cuMul_tn_kernel(cytnx_int32 *out, const cytnx_int32 *val,
                                    const cytnx_uint64 Nelem, const cytnx_bool *ptr) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          val[blockIdx.x * blockDim.x + threadIdx.x] *
          cytnx_int32(ptr[blockIdx.x * blockDim.x + threadIdx.x]);
      }
      __syncthreads();
    }
    void cuMul_internal_i32tb(boost::intrusive_ptr<Storage_base> &out,
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

      if (Lin->size() == 1 and Rin->size() == 1) {
        cuMul_constconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin[0]);
      } else if (Lin->size() == 1) {
        cuMul_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMul_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, cytnx_int32(_Rin[0]));
      } else {
        cuMul_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }

    //---------------------------------

    void cuMul_internal_u32tcd(boost::intrusive_ptr<Storage_base> &out,
                               boost::intrusive_ptr<Storage_base> &Lin,
                               boost::intrusive_ptr<Storage_base> &Rin,
                               const unsigned long long &len,
                               const std::vector<cytnx_uint64> &shape,
                               const std::vector<cytnx_uint64> &invmapper_L,
                               const std::vector<cytnx_uint64> &invmapper_R) {
      cuMul_internal_cdtu32(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void cuMul_internal_u32tcf(boost::intrusive_ptr<Storage_base> &out,
                               boost::intrusive_ptr<Storage_base> &Lin,
                               boost::intrusive_ptr<Storage_base> &Rin,
                               const unsigned long long &len,
                               const std::vector<cytnx_uint64> &shape,
                               const std::vector<cytnx_uint64> &invmapper_L,
                               const std::vector<cytnx_uint64> &invmapper_R) {
      cuMul_internal_cftu32(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void cuMul_internal_u32td(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R) {
      cuMul_internal_dtu32(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void cuMul_internal_u32tf(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R) {
      cuMul_internal_ftu32(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void cuMul_internal_u32ti64(boost::intrusive_ptr<Storage_base> &out,
                                boost::intrusive_ptr<Storage_base> &Lin,
                                boost::intrusive_ptr<Storage_base> &Rin,
                                const unsigned long long &len,
                                const std::vector<cytnx_uint64> &shape,
                                const std::vector<cytnx_uint64> &invmapper_L,
                                const std::vector<cytnx_uint64> &invmapper_R) {
      cuMul_internal_i64tu32(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void cuMul_internal_u32tu64(boost::intrusive_ptr<Storage_base> &out,
                                boost::intrusive_ptr<Storage_base> &Lin,
                                boost::intrusive_ptr<Storage_base> &Rin,
                                const unsigned long long &len,
                                const std::vector<cytnx_uint64> &shape,
                                const std::vector<cytnx_uint64> &invmapper_L,
                                const std::vector<cytnx_uint64> &invmapper_R) {
      cuMul_internal_u64tu32(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void cuMul_internal_u32ti32(boost::intrusive_ptr<Storage_base> &out,
                                boost::intrusive_ptr<Storage_base> &Lin,
                                boost::intrusive_ptr<Storage_base> &Rin,
                                const unsigned long long &len,
                                const std::vector<cytnx_uint64> &shape,
                                const std::vector<cytnx_uint64> &invmapper_L,
                                const std::vector<cytnx_uint64> &invmapper_R) {
      cuMul_internal_i32tu32(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void cuMul_internal_u32tu32(boost::intrusive_ptr<Storage_base> &out,
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

      if (Lin->size() == 1 and Rin->size() == 1) {
        cuMul_constconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin[0]);
      } else if (Lin->size() == 1) {
        cuMul_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMul_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cuMul_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }
    void cuMul_internal_u32ti16(boost::intrusive_ptr<Storage_base> &out,
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

      if (Lin->size() == 1 and Rin->size() == 1) {
        cuMul_constconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin[0]);
      } else if (Lin->size() == 1) {
        cuMul_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMul_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cuMul_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }
    void cuMul_internal_u32tu16(boost::intrusive_ptr<Storage_base> &out,
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

      if (Lin->size() == 1 and Rin->size() == 1) {
        cuMul_constconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin[0]);
      } else if (Lin->size() == 1) {
        cuMul_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMul_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cuMul_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }

    __global__ void cuMul_constconst_kernel(cytnx_uint32 *out, const cytnx_uint32 ptr,
                                            const cytnx_uint64 Nelem, const cytnx_bool val) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] = ptr * cytnx_uint32(val);
      }
      __syncthreads();
    }
    __global__ void cuMul_lconst_kernel(cytnx_uint32 *out, const cytnx_uint32 val,
                                        const cytnx_uint64 Nelem, const cytnx_bool *ptr) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          val * cytnx_uint32(ptr[blockIdx.x * blockDim.x + threadIdx.x]);
      }
      __syncthreads();
    }

    __global__ void cuMul_tn_kernel(cytnx_uint32 *out, const cytnx_uint32 *val,
                                    const cytnx_uint64 Nelem, const cytnx_bool *ptr) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          val[blockIdx.x * blockDim.x + threadIdx.x] *
          cytnx_uint32(ptr[blockIdx.x * blockDim.x + threadIdx.x]);
      }
      __syncthreads();
    }
    void cuMul_internal_u32tb(boost::intrusive_ptr<Storage_base> &out,
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

      if (Lin->size() == 1 and Rin->size() == 1) {
        cuMul_constconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin[0]);
      } else if (Lin->size() == 1) {
        cuMul_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMul_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, cytnx_uint32(_Rin[0]));
      } else {
        cuMul_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }
    //---------------------------------

    void cuMul_internal_i16tcd(boost::intrusive_ptr<Storage_base> &out,
                               boost::intrusive_ptr<Storage_base> &Lin,
                               boost::intrusive_ptr<Storage_base> &Rin,
                               const unsigned long long &len,
                               const std::vector<cytnx_uint64> &shape,
                               const std::vector<cytnx_uint64> &invmapper_L,
                               const std::vector<cytnx_uint64> &invmapper_R) {
      cuMul_internal_cdti16(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void cuMul_internal_i16tcf(boost::intrusive_ptr<Storage_base> &out,
                               boost::intrusive_ptr<Storage_base> &Lin,
                               boost::intrusive_ptr<Storage_base> &Rin,
                               const unsigned long long &len,
                               const std::vector<cytnx_uint64> &shape,
                               const std::vector<cytnx_uint64> &invmapper_L,
                               const std::vector<cytnx_uint64> &invmapper_R) {
      cuMul_internal_cfti16(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void cuMul_internal_i16td(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R) {
      cuMul_internal_dti16(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void cuMul_internal_i16tf(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R) {
      cuMul_internal_fti16(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void cuMul_internal_i16ti64(boost::intrusive_ptr<Storage_base> &out,
                                boost::intrusive_ptr<Storage_base> &Lin,
                                boost::intrusive_ptr<Storage_base> &Rin,
                                const unsigned long long &len,
                                const std::vector<cytnx_uint64> &shape,
                                const std::vector<cytnx_uint64> &invmapper_L,
                                const std::vector<cytnx_uint64> &invmapper_R) {
      cuMul_internal_i64ti16(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void cuMul_internal_i16tu64(boost::intrusive_ptr<Storage_base> &out,
                                boost::intrusive_ptr<Storage_base> &Lin,
                                boost::intrusive_ptr<Storage_base> &Rin,
                                const unsigned long long &len,
                                const std::vector<cytnx_uint64> &shape,
                                const std::vector<cytnx_uint64> &invmapper_L,
                                const std::vector<cytnx_uint64> &invmapper_R) {
      cuMul_internal_u64ti16(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void cuMul_internal_i16ti32(boost::intrusive_ptr<Storage_base> &out,
                                boost::intrusive_ptr<Storage_base> &Lin,
                                boost::intrusive_ptr<Storage_base> &Rin,
                                const unsigned long long &len,
                                const std::vector<cytnx_uint64> &shape,
                                const std::vector<cytnx_uint64> &invmapper_L,
                                const std::vector<cytnx_uint64> &invmapper_R) {
      cuMul_internal_i32ti16(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void cuMul_internal_i16tu32(boost::intrusive_ptr<Storage_base> &out,
                                boost::intrusive_ptr<Storage_base> &Lin,
                                boost::intrusive_ptr<Storage_base> &Rin,
                                const unsigned long long &len,
                                const std::vector<cytnx_uint64> &shape,
                                const std::vector<cytnx_uint64> &invmapper_L,
                                const std::vector<cytnx_uint64> &invmapper_R) {
      cuMul_internal_u32ti16(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void cuMul_internal_i16ti16(boost::intrusive_ptr<Storage_base> &out,
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

      if (Lin->size() == 1 and Rin->size() == 1) {
        cuMul_constconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin[0]);
      } else if (Lin->size() == 1) {
        cuMul_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMul_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cuMul_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }
    void cuMul_internal_i16tu16(boost::intrusive_ptr<Storage_base> &out,
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

      if (Lin->size() == 1 and Rin->size() == 1) {
        cuMul_constconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin[0]);
      } else if (Lin->size() == 1) {
        cuMul_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMul_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cuMul_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }

    __global__ void cuMul_constconst_kernel(cytnx_int16 *out, const cytnx_int16 ptr,
                                            const cytnx_uint64 Nelem, const cytnx_bool val) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] = ptr * cytnx_int16(val);
      }
      __syncthreads();
    }
    __global__ void cuMul_lconst_kernel(cytnx_int16 *out, const cytnx_int16 val,
                                        const cytnx_uint64 Nelem, const cytnx_bool *ptr) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          val * cytnx_int16(ptr[blockIdx.x * blockDim.x + threadIdx.x]);
      }
      __syncthreads();
    }

    __global__ void cuMul_tn_kernel(cytnx_int16 *out, const cytnx_int16 *val,
                                    const cytnx_uint64 Nelem, const cytnx_bool *ptr) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          val[blockIdx.x * blockDim.x + threadIdx.x] *
          cytnx_int16(ptr[blockIdx.x * blockDim.x + threadIdx.x]);
      }
      __syncthreads();
    }
    void cuMul_internal_i16tb(boost::intrusive_ptr<Storage_base> &out,
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

      if (Lin->size() == 1 and Rin->size() == 1) {
        cuMul_constconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin[0]);
      } else if (Lin->size() == 1) {
        cuMul_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMul_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, cytnx_int16(_Rin[0]));
      } else {
        cuMul_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }
    //---------------------------------

    void cuMul_internal_u16tcd(boost::intrusive_ptr<Storage_base> &out,
                               boost::intrusive_ptr<Storage_base> &Lin,
                               boost::intrusive_ptr<Storage_base> &Rin,
                               const unsigned long long &len,
                               const std::vector<cytnx_uint64> &shape,
                               const std::vector<cytnx_uint64> &invmapper_L,
                               const std::vector<cytnx_uint64> &invmapper_R) {
      cuMul_internal_cdtu16(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void cuMul_internal_u16tcf(boost::intrusive_ptr<Storage_base> &out,
                               boost::intrusive_ptr<Storage_base> &Lin,
                               boost::intrusive_ptr<Storage_base> &Rin,
                               const unsigned long long &len,
                               const std::vector<cytnx_uint64> &shape,
                               const std::vector<cytnx_uint64> &invmapper_L,
                               const std::vector<cytnx_uint64> &invmapper_R) {
      cuMul_internal_cftu16(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void cuMul_internal_u16td(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R) {
      cuMul_internal_dtu16(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void cuMul_internal_u16tf(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R) {
      cuMul_internal_ftu16(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void cuMul_internal_u16ti64(boost::intrusive_ptr<Storage_base> &out,
                                boost::intrusive_ptr<Storage_base> &Lin,
                                boost::intrusive_ptr<Storage_base> &Rin,
                                const unsigned long long &len,
                                const std::vector<cytnx_uint64> &shape,
                                const std::vector<cytnx_uint64> &invmapper_L,
                                const std::vector<cytnx_uint64> &invmapper_R) {
      cuMul_internal_i64tu16(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void cuMul_internal_u16tu64(boost::intrusive_ptr<Storage_base> &out,
                                boost::intrusive_ptr<Storage_base> &Lin,
                                boost::intrusive_ptr<Storage_base> &Rin,
                                const unsigned long long &len,
                                const std::vector<cytnx_uint64> &shape,
                                const std::vector<cytnx_uint64> &invmapper_L,
                                const std::vector<cytnx_uint64> &invmapper_R) {
      cuMul_internal_u64tu16(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void cuMul_internal_u16ti32(boost::intrusive_ptr<Storage_base> &out,
                                boost::intrusive_ptr<Storage_base> &Lin,
                                boost::intrusive_ptr<Storage_base> &Rin,
                                const unsigned long long &len,
                                const std::vector<cytnx_uint64> &shape,
                                const std::vector<cytnx_uint64> &invmapper_L,
                                const std::vector<cytnx_uint64> &invmapper_R) {
      cuMul_internal_i32tu16(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void cuMul_internal_u16tu32(boost::intrusive_ptr<Storage_base> &out,
                                boost::intrusive_ptr<Storage_base> &Lin,
                                boost::intrusive_ptr<Storage_base> &Rin,
                                const unsigned long long &len,
                                const std::vector<cytnx_uint64> &shape,
                                const std::vector<cytnx_uint64> &invmapper_L,
                                const std::vector<cytnx_uint64> &invmapper_R) {
      cuMul_internal_u32tu16(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void cuMul_internal_u16ti16(boost::intrusive_ptr<Storage_base> &out,
                                boost::intrusive_ptr<Storage_base> &Lin,
                                boost::intrusive_ptr<Storage_base> &Rin,
                                const unsigned long long &len,
                                const std::vector<cytnx_uint64> &shape,
                                const std::vector<cytnx_uint64> &invmapper_L,
                                const std::vector<cytnx_uint64> &invmapper_R) {
      cuMul_internal_i16tu16(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void cuMul_internal_u16tu16(boost::intrusive_ptr<Storage_base> &out,
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

      if (Lin->size() == 1 and Rin->size() == 1) {
        cuMul_constconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin[0]);
      } else if (Lin->size() == 1) {
        cuMul_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMul_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cuMul_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }

    __global__ void cuMul_constconst_kernel(cytnx_uint16 *out, const cytnx_uint16 ptr,
                                            const cytnx_uint64 Nelem, const cytnx_bool val) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] = ptr * cytnx_uint16(val);
      }
      __syncthreads();
    }
    __global__ void cuMul_lconst_kernel(cytnx_uint16 *out, const cytnx_uint16 val,
                                        const cytnx_uint64 Nelem, const cytnx_bool *ptr) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          val * cytnx_uint16(ptr[blockIdx.x * blockDim.x + threadIdx.x]);
      }
      __syncthreads();
    }

    __global__ void cuMul_tn_kernel(cytnx_uint16 *out, const cytnx_uint16 *val,
                                    const cytnx_uint64 Nelem, const cytnx_bool *ptr) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          val[blockIdx.x * blockDim.x + threadIdx.x] *
          cytnx_uint16(ptr[blockIdx.x * blockDim.x + threadIdx.x]);
      }
      __syncthreads();
    }
    void cuMul_internal_u16tb(boost::intrusive_ptr<Storage_base> &out,
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

      if (Lin->size() == 1 and Rin->size() == 1) {
        cuMul_constconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin[0]);
      } else if (Lin->size() == 1) {
        cuMul_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMul_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, cytnx_int16(_Rin[0]));
      } else {
        cuMul_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }
    //---------------------------------

    void cuMul_internal_btcd(boost::intrusive_ptr<Storage_base> &out,
                             boost::intrusive_ptr<Storage_base> &Lin,
                             boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                             const std::vector<cytnx_uint64> &shape,
                             const std::vector<cytnx_uint64> &invmapper_L,
                             const std::vector<cytnx_uint64> &invmapper_R) {
      cuMul_internal_cdtb(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void cuMul_internal_btcf(boost::intrusive_ptr<Storage_base> &out,
                             boost::intrusive_ptr<Storage_base> &Lin,
                             boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                             const std::vector<cytnx_uint64> &shape,
                             const std::vector<cytnx_uint64> &invmapper_L,
                             const std::vector<cytnx_uint64> &invmapper_R) {
      cuMul_internal_cftb(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void cuMul_internal_btd(boost::intrusive_ptr<Storage_base> &out,
                            boost::intrusive_ptr<Storage_base> &Lin,
                            boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                            const std::vector<cytnx_uint64> &shape,
                            const std::vector<cytnx_uint64> &invmapper_L,
                            const std::vector<cytnx_uint64> &invmapper_R) {
      cuMul_internal_dtb(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void cuMul_internal_btf(boost::intrusive_ptr<Storage_base> &out,
                            boost::intrusive_ptr<Storage_base> &Lin,
                            boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                            const std::vector<cytnx_uint64> &shape,
                            const std::vector<cytnx_uint64> &invmapper_L,
                            const std::vector<cytnx_uint64> &invmapper_R) {
      cuMul_internal_ftb(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void cuMul_internal_bti64(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R) {
      cuMul_internal_i64tb(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void cuMul_internal_btu64(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R) {
      cuMul_internal_u64tb(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void cuMul_internal_bti32(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R) {
      cuMul_internal_i32tb(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void cuMul_internal_btu32(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R) {
      cuMul_internal_u32tb(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void cuMul_internal_bti16(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R) {
      cuMul_internal_i16tb(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }
    void cuMul_internal_btu16(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R) {
      cuMul_internal_u16tb(out, Rin, Lin, len, shape, invmapper_R, invmapper_L);
    }

    void cuMul_internal_btb(boost::intrusive_ptr<Storage_base> &out,
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

      if (Lin->size() == 1 and Rin->size() == 1) {
        cuMul_constconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin[0]);
      } else if (Lin->size() == 1) {
        cuMul_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
      } else if (Rin->size() == 1) {
        cuMul_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
      } else {
        cuMul_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
      }
    }

  }  // namespace linalg_internal
}  // namespace cytnx
