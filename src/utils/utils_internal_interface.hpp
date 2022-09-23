#ifndef _H_utils_internal_
#define _H_utils_internal_

#include "utils_internal_cpu/Cast_cpu.hpp"
#include "utils_internal_cpu/Movemem_cpu.hpp"
#include "utils_internal_cpu/Alloc_cpu.hpp"
#include "utils_internal_cpu/SetZeros_cpu.hpp"
#include "utils_internal_cpu/Range_cpu.hpp"
#include "utils_internal_cpu/Fill_cpu.hpp"
#include "utils_internal_cpu/SetArange_cpu.hpp"
#include "utils_internal_cpu/GetElems_cpu.hpp"
#include "utils_internal_cpu/GetElems_contiguous_cpu.hpp"
#include "utils_internal_cpu/SetElems_cpu.hpp"
#include "utils_internal_cpu/SetElems_contiguous_cpu.hpp"
#include "utils_internal_cpu/Complexmem_cpu.hpp"

#include "utils_internal_cpu/blocks_mvelems_cpu.hpp"

#ifdef UNI_GPU
  #include "utils_internal_gpu/cuCast_gpu.hpp"
  #include "utils_internal_gpu/cuAlloc_gpu.hpp"
  #include "utils_internal_gpu/cuMovemem_gpu.hpp"
  #include "utils_internal_gpu/cuReduce_gpu.hpp"
  #include "utils_internal_gpu/cuSetZeros_gpu.hpp"
  #include "utils_internal_gpu/cuFill_gpu.hpp"
  #include "utils_internal_gpu/cuSetArange_gpu.hpp"
  #include "utils_internal_gpu/cuSetElems_gpu.hpp"
  #include "utils_internal_gpu/cuGetElems_gpu.hpp"
  #include "utils_internal_gpu/cuComplexmem_gpu.hpp"
#endif

#include "Type.hpp"
#include "utils/complex_arithmetic.hpp"
#ifdef UNI_GPU
  #include "utils/cucomplex_arithmetic.hpp"
#endif

#include "Storage.hpp"
namespace cytnx {
  namespace utils_internal {

    bool _fx_compare_vec_inc(const std::vector<cytnx_int64> &v1,
                             const std::vector<cytnx_int64> &v2);
    bool _fx_compare_vec_dec(const std::vector<cytnx_int64> &v1,
                             const std::vector<cytnx_int64> &v2);
    typedef void (*ElemCast_io)(const boost::intrusive_ptr<Storage_base> &,
                                boost::intrusive_ptr<Storage_base> &, const unsigned long long &,
                                const int &);
    typedef void (*SetArange_io)(boost::intrusive_ptr<Storage_base> &, const double &,
                                 const double &, const double &, const cytnx_uint64 &);
    typedef void (*GetElems_io)(void *, void *, const std::vector<cytnx_uint64> &,
                                const std::vector<cytnx_uint64> &,
                                const std::vector<std::vector<cytnx_uint64>> &,
                                const cytnx_uint64 &);
    typedef void (*GetElems_conti_io)(void *, void *, const std::vector<cytnx_uint64> &,
                                      const std::vector<cytnx_uint64> &,
                                      const std::vector<std::vector<cytnx_uint64>> &,
                                      const cytnx_uint64 &, const cytnx_uint64 &);
    typedef void (*SetElems_io)(void *, void *, const std::vector<cytnx_uint64> &,
                                const std::vector<cytnx_uint64> &,
                                const std::vector<std::vector<cytnx_uint64>> &,
                                const cytnx_uint64 &, const bool &);
    typedef void (*SetElems_conti_io)(void *, void *, const std::vector<cytnx_uint64> &,
                                      const std::vector<cytnx_uint64> &,
                                      const std::vector<std::vector<cytnx_uint64>> &,
                                      const cytnx_uint64 &, const cytnx_uint64 &, const bool &);
    // typedef void (*blocks_mvelems_io)(vector<Storage> &, const vector<Storage> &,
    //     const vector<vector<cytnx_uint64>> &, const vector<vector<cytnx_uint64>> &,
    //     const vector<cytnx_uint64> &,
    //     const vector<vector<cytnx_uint64>>  &, const vector<vector<cytnx_uint64>>  &,
    //     map<cytnx_uint64, pair<cytnx_uint64,cytnx_uint64>>  &, map<cytnx_uint64,
    //     pair<cytnx_uint64,cytnx_uint64>> &, const vector<cytnx_uint64> &, const
    //     vector<cytnx_uint64> &, const cytnx_uint64 &,  const cytnx_uint64 &);

    typedef void (*blocks_mvelems_io)(vector<Tensor> &, const vector<Tensor> &,
                                      const vector<cytnx_uint64> &,
                                      const vector<vector<cytnx_uint64>> &,
                                      const vector<vector<cytnx_uint64>> &,
                                      map<cytnx_uint64, pair<cytnx_uint64, cytnx_uint64>> &,
                                      map<cytnx_uint64, pair<cytnx_uint64, cytnx_uint64>> &,
                                      const vector<cytnx_uint64> &, const vector<cytnx_uint64> &,
                                      const cytnx_uint64 &, const cytnx_uint64 &);

    class utils_internal_interface {
     public:
      // Cast
      std::vector<std::vector<ElemCast_io>> ElemCast;
      std::vector<SetArange_io> SetArange_ii;
      std::vector<GetElems_io> GetElems_ii;
      std::vector<GetElems_conti_io> GetElems_conti_ii;
      std::vector<std::vector<SetElems_conti_io>> SetElems_conti_ii;
      std::vector<std::vector<SetElems_io>> SetElems_ii;

      std::vector<blocks_mvelems_io> blocks_mvelems_ii;

#ifdef UNI_GPU
      std::vector<std::vector<ElemCast_io>> cuElemCast;
      std::vector<SetArange_io> cuSetArange_ii;
      std::vector<GetElems_io> cuGetElems_ii;
      std::vector<std::vector<SetElems_io>> cuSetElems_ii;
#endif

      utils_internal_interface();
    };
    extern utils_internal_interface uii;

  }  // namespace utils_internal
}  // namespace cytnx

#endif
