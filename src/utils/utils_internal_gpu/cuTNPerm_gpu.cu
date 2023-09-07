#include "cuTNPerm_gpu.hpp"
#include "cuAlloc_gpu.hpp"
#include "backend/Storage.hpp"
#include <algorithm>
#ifdef UNI_OMP
  #include <omp.h>
#endif

#ifdef UNI_GPU
  #ifdef UNI_CUTENSOR
    #include "cutensor.h"
  #endif
#endif

using namespace std;

namespace cytnx {
  namespace utils_internal {

#ifdef UNI_GPU
    boost::intrusive_ptr<Storage_base> cuTNPerm_gpu_cd(boost::intrusive_ptr<Storage_base> &in,
                                                       const std::vector<cytnx_uint64> &old_shape,
                                                       const std::vector<cytnx_uint64> &mapper,
                                                       const std::vector<cytnx_uint64> &invmapper,
                                                       const bool is_inplace) {
  #ifdef UNI_DEBUG
      cytnx_error_msg(in->dtype != Type.ComplexDouble,
                      "[DEBUG][internal error] in.dtype_str is [%s] but call cuTNPerm_gpu with "
                      "type ComplexDouble",
                      in->dtype_str().c_str());
      cytnx_error_msg(in->device == Device.cpu, "%s",
                      "[DEBUG][internal error] in.device is on cpu but all cuda function.");
  #endif

      std::vector<cytnx_uint64> newshape(old_shape.size());
      for (cytnx_uint64 i = 0; i < old_shape.size(); i++) newshape[i] = old_shape[mapper[i]];

      std::vector<cytnx_uint64> shifter_old(old_shape.size());
      std::vector<cytnx_uint64> shifter_new(old_shape.size());

      cytnx_uint64 accu_old = 1, accu_new = 1;
      for (cytnx_int64 i = old_shape.size() - 1; i >= 0; i--) {
        shifter_old[i] = accu_old;
        shifter_new[i] = accu_new;
        accu_old *= old_shape[i];
        accu_new *= newshape[i];
      }
      std::vector<cytnx_uint64> old_inds(old_shape.size());

      std::vector<cytnx_uint64> permuted_shifter_new(old_shape.size());
      for (unsigned int i = 0; i < old_shape.size(); i++)
        permuted_shifter_new[i] = shifter_new[invmapper[i]];

      /// allocate a GPU for psn-vec/so-vec/tmp des-vec
      cytnx_uint64 *dshifter_old, *dperm_shifter_new;
      cuDoubleComplex *dtmp;
      cytnx_uint64 Nelem = accu_old;

      cudaSetDevice(in->device);  // ensure the following allocation on the same device as src.
      checkCudaErrors(
        cudaMalloc((void **)&dshifter_old, sizeof(cytnx_uint64) * shifter_old.size()));
      checkCudaErrors(cudaMalloc((void **)&dperm_shifter_new,
                                 sizeof(cytnx_uint64) * permuted_shifter_new.size()));
      dtmp = (cuDoubleComplex *)cuMalloc_gpu(sizeof(cuDoubleComplex) * in->cap);

      /// copy psn-vec/so-vec to device
      checkCudaErrors(cudaMemcpy(dperm_shifter_new, &permuted_shifter_new[0],
                                 sizeof(cytnx_uint64) * permuted_shifter_new.size(),
                                 cudaMemcpyHostToDevice));
      checkCudaErrors(cudaMemcpy(dshifter_old, &shifter_old[0],
                                 sizeof(cytnx_uint64) * shifter_old.size(),
                                 cudaMemcpyHostToDevice));

      /// calculate how many blocks, and shared mem size, thpb fixed at 256 (need fine tune)
      cytnx_uint64 NBlocks = Nelem / 256;
      if (Nelem % 256) {
        NBlocks += 1;
      }
      cuMovemem_kernel<<<NBlocks, 256, shifter_old.size() * 2 * sizeof(cytnx_uint64)>>>(
        dtmp, (cuDoubleComplex *)in->Mem, dshifter_old, dperm_shifter_new, old_shape.size(), Nelem);

      /// house keeping:
      checkCudaErrors(cudaFree(dshifter_old));
      checkCudaErrors(cudaFree(dperm_shifter_new));

      boost::intrusive_ptr<Storage_base> out(new ComplexDoubleStorage());
      if (is_inplace) {
        /// cpy back:
        checkCudaErrors(
          cudaMemcpy(in->Mem, dtmp, sizeof(cytnx_complex128) * Nelem, cudaMemcpyDeviceToDevice));
        cudaFree(dtmp);
        return out;

      } else {
        out->_Init_byptr(dtmp, Nelem, in->device, true, in->cap);
        return out;
      }
    }

    boost::intrusive_ptr<Storage_base> cuTNPerm_gpu_cf(boost::intrusive_ptr<Storage_base> &in,
                                                       const std::vector<cytnx_uint64> &old_shape,
                                                       const std::vector<cytnx_uint64> &mapper,
                                                       const std::vector<cytnx_uint64> &invmapper,
                                                       const bool is_inplace) {
  #ifdef UNI_DEBUG
      cytnx_error_msg(
        in->dtype != Type.ComplexFloat,
        "[DEBUG][internal error] in.dtype_str is [%s] but call cuTNPerm_gpu with type ComplexFloat",
        in->dtype_str().c_str());
      cytnx_error_msg(in->device == Device.cpu, "%s",
                      "[DEBUG][internal error] in.device is on cpu but all cuda function.");
  #endif

      cuFloatComplex *dtmp;
      dtmp = (cuFloatComplex *)cuMalloc_gpu(sizeof(cuFloatComplex) * in->cap);
      cytnx_uint64 Nelem = in->len;

  #ifdef UNI_CUTT
      std::vector<int> perm(mapper.begin(), mapper.end());
      std::vector<int> size(old_shape.begin(), old_shape.end());
      std::reverse(size.begin(), size.end());  // matching API CUTT
      reverse_perm(perm.begin(), perm.end(), perm.size());  // matching API CUTT

      cuttHandle plan;
      cuttPlan(&plan, perm.size(), size.data(), perm.data(), sizeof(double), 0);
      cuttExecute(plan, in->Mem, dtmp);

      cuttDestroy(plan);

  #else
      std::vector<cytnx_uint64> newshape(old_shape.size());
      for (cytnx_uint64 i = 0; i < old_shape.size(); i++) newshape[i] = old_shape[mapper[i]];

      std::vector<cytnx_uint64> shifter_old(old_shape.size());
      std::vector<cytnx_uint64> shifter_new(old_shape.size());

      cytnx_uint64 accu_old = 1, accu_new = 1;
      for (cytnx_int64 i = old_shape.size() - 1; i >= 0; i--) {
        shifter_old[i] = accu_old;
        shifter_new[i] = accu_new;
        accu_old *= old_shape[i];
        accu_new *= newshape[i];
      }
      std::vector<cytnx_uint64> old_inds(old_shape.size());

      std::vector<cytnx_uint64> permuted_shifter_new(old_shape.size());
      for (unsigned int i = 0; i < old_shape.size(); i++)
        permuted_shifter_new[i] = shifter_new[invmapper[i]];

      /// allocate a GPU for psn-vec/so-vec/tmp des-vec
      cytnx_uint64 *dshifter_old, *dperm_shifter_new;

      cudaSetDevice(in->device);  // ensure the following allocation on the same device as src.
      checkCudaErrors(
        cudaMalloc((void **)&dshifter_old, sizeof(cytnx_uint64) * shifter_old.size()));
      checkCudaErrors(cudaMalloc((void **)&dperm_shifter_new,
                                 sizeof(cytnx_uint64) * permuted_shifter_new.size()));

      /// copy psn-vec/so-vec to device
      checkCudaErrors(cudaMemcpy(dperm_shifter_new, &permuted_shifter_new[0],
                                 sizeof(cytnx_uint64) * permuted_shifter_new.size(),
                                 cudaMemcpyHostToDevice));
      checkCudaErrors(cudaMemcpy(dshifter_old, &shifter_old[0],
                                 sizeof(cytnx_uint64) * shifter_old.size(),
                                 cudaMemcpyHostToDevice));

      /// calculate how many blocks, and shared mem size, thpb fixed at 256 (need fine tune)
      cytnx_uint64 NBlocks = Nelem / 256;
      if (Nelem % 256) {
        NBlocks += 1;
      }
      cuMovemem_kernel<<<NBlocks, 256, shifter_old.size() * 2 * sizeof(cytnx_uint64)>>>(
        dtmp, (cuFloatComplex *)in->Mem, dshifter_old, dperm_shifter_new, old_shape.size(), Nelem);

      /// house keeping:
      checkCudaErrors(cudaFree(dshifter_old));
      checkCudaErrors(cudaFree(dperm_shifter_new));
  #endif

      boost::intrusive_ptr<Storage_base> out(new ComplexFloatStorage());
      if (is_inplace) {
        /// cpy back:
        checkCudaErrors(
          cudaMemcpy(in->Mem, dtmp, sizeof(cytnx_complex64) * Nelem, cudaMemcpyDeviceToDevice));
        cudaFree(dtmp);
        return out;

      } else {
        out->_Init_byptr(dtmp, Nelem, in->device, true, in->cap);
        return out;
      }
    }

    boost::intrusive_ptr<Storage_base> cuTNPerm_gpu_d(boost::intrusive_ptr<Storage_base> &in,
                                                      const std::vector<cytnx_uint64> &old_shape,
                                                      const std::vector<cytnx_uint64> &mapper,
                                                      const std::vector<cytnx_uint64> &invmapper,
                                                      const bool is_inplace) {
  #ifdef UNI_DEBUG
      cytnx_error_msg(
        in->dtype != Type.Double,
        "[DEBUG][internal error] in.dtype_str is [%s] but call cuTNPerm_gpu with type Double",
        in->dtype_str().c_str());
      cytnx_error_msg(in->device == Device.cpu, "%s",
                      "[DEBUG][internal error] in.device is on cpu but all cuda function.");
  #endif

      double *dtmp;
      dtmp = (double *)cuMalloc_gpu(sizeof(double) * in->cap);
      cytnx_uint64 Nelem = in->len;

  #ifdef UNI_CUTT
      std::vector<int> perm(mapper.begin(), mapper.end());
      std::vector<int> size(old_shape.begin(), old_shape.end());
      std::reverse(size.begin(), size.end());  // matching API CUTT
      reverse_perm(perm.begin(), perm.end(), perm.size());  // matching API CUTT

      cuttHandle plan;
      cuttPlan(&plan, perm.size(), size.data(), perm.data(), sizeof(double), 0);
      cuttExecute(plan, in->Mem, dtmp);

      cuttDestroy(plan);

  #else
      std::vector<cytnx_uint64> newshape(old_shape.size());
      for (cytnx_uint64 i = 0; i < old_shape.size(); i++) newshape[i] = old_shape[mapper[i]];

      std::vector<cytnx_uint64> shifter_old(old_shape.size());
      std::vector<cytnx_uint64> shifter_new(old_shape.size());

      cytnx_uint64 accu_old = 1, accu_new = 1;
      for (cytnx_int64 i = old_shape.size() - 1; i >= 0; i--) {
        shifter_old[i] = accu_old;
        shifter_new[i] = accu_new;
        accu_old *= old_shape[i];
        accu_new *= newshape[i];
      }
      std::vector<cytnx_uint64> old_inds(old_shape.size());

      std::vector<cytnx_uint64> permuted_shifter_new(old_shape.size());
      for (unsigned int i = 0; i < old_shape.size(); i++)
        permuted_shifter_new[i] = shifter_new[invmapper[i]];

      /// allocate a GPU for psn-vec/so-vec/tmp des-vec
      cytnx_uint64 *dshifter_old, *dperm_shifter_new;

      cudaSetDevice(in->device);  // ensure the following allocation on the same device as src.
      checkCudaErrors(
        cudaMalloc((void **)&dshifter_old, sizeof(cytnx_uint64) * shifter_old.size()));
      checkCudaErrors(cudaMalloc((void **)&dperm_shifter_new,
                                 sizeof(cytnx_uint64) * permuted_shifter_new.size()));

      /// copy psn-vec/so-vec to device
      checkCudaErrors(cudaMemcpy(dperm_shifter_new, &permuted_shifter_new[0],
                                 sizeof(cytnx_uint64) * permuted_shifter_new.size(),
                                 cudaMemcpyHostToDevice));
      checkCudaErrors(cudaMemcpy(dshifter_old, &shifter_old[0],
                                 sizeof(cytnx_uint64) * shifter_old.size(),
                                 cudaMemcpyHostToDevice));

      /// calculate how many blocks, and shared mem size, thpb fixed at 256 (need fine tune)
      cytnx_uint64 NBlocks = Nelem / 256;
      if (Nelem % 256) {
        NBlocks += 1;
      }
      cuMovemem_kernel<<<NBlocks, 256, shifter_old.size() * 2 * sizeof(cytnx_uint64)>>>(
        dtmp, (double *)in->Mem, dshifter_old, dperm_shifter_new, old_shape.size(), Nelem);

      /// house keeping:
      checkCudaErrors(cudaFree(dshifter_old));
      checkCudaErrors(cudaFree(dperm_shifter_new));

  #endif

      boost::intrusive_ptr<Storage_base> out(new DoubleStorage());
      if (is_inplace) {
        /// cpy back:
        checkCudaErrors(
          cudaMemcpy(in->Mem, dtmp, sizeof(double) * Nelem, cudaMemcpyDeviceToDevice));
        cudaFree(dtmp);
        return out;

      } else {
        out->_Init_byptr(dtmp, Nelem, in->device, true, in->cap);
        return out;
      }
    }

    boost::intrusive_ptr<Storage_base> cuTNPerm_gpu_f(boost::intrusive_ptr<Storage_base> &in,
                                                      const std::vector<cytnx_uint64> &old_shape,
                                                      const std::vector<cytnx_uint64> &mapper,
                                                      const std::vector<cytnx_uint64> &invmapper,
                                                      const bool is_inplace) {
  #ifdef UNI_DEBUG
      cytnx_error_msg(
        in->dtype != Type.Float,
        "[DEBUG][internal error] in.dtype_str is [%s] but call cuTNPerm_gpu with type Float",
        in->dtype_str().c_str());
      cytnx_error_msg(in->device == Device.cpu, "%s",
                      "[DEBUG][internal error] in.device is on cpu but all cuda function.");
  #endif

      float *dtmp;
      dtmp = (float *)cuMalloc_gpu(sizeof(float) * in->cap);
      cytnx_uint64 Nelem = in->len;

  #ifdef UNI_CUTT

      std::vector<int> perm(mapper.begin(), mapper.end());
      std::vector<int> size(old_shape.begin(), old_shape.end());
      std::reverse(size.begin(), size.end());  // matching API CUTT
      reverse_perm(perm.begin(), perm.end(), perm.size());  // matching API CUTT

      cuttHandle plan;
      cuttPlan(&plan, perm.size(), size.data(), perm.data(), sizeof(float), 0);
      cuttExecute(plan, in->Mem, dtmp);
      cuttDestroy(plan);

  #else
      std::vector<cytnx_uint64> newshape(old_shape.size());
      for (cytnx_uint64 i = 0; i < old_shape.size(); i++) newshape[i] = old_shape[mapper[i]];

      std::vector<cytnx_uint64> shifter_old(old_shape.size());
      std::vector<cytnx_uint64> shifter_new(old_shape.size());

      cytnx_uint64 accu_old = 1, accu_new = 1;
      for (cytnx_int64 i = old_shape.size() - 1; i >= 0; i--) {
        shifter_old[i] = accu_old;
        shifter_new[i] = accu_new;
        accu_old *= old_shape[i];
        accu_new *= newshape[i];
      }
      std::vector<cytnx_uint64> old_inds(old_shape.size());

      std::vector<cytnx_uint64> permuted_shifter_new(old_shape.size());
      for (unsigned int i = 0; i < old_shape.size(); i++)
        permuted_shifter_new[i] = shifter_new[invmapper[i]];

      /// allocate a GPU for psn-vec/so-vec/tmp des-vec
      cytnx_uint64 *dshifter_old, *dperm_shifter_new;

      cudaSetDevice(in->device);  // ensure the following allocation on the same device as src.
      checkCudaErrors(
        cudaMalloc((void **)&dshifter_old, sizeof(cytnx_uint64) * shifter_old.size()));
      checkCudaErrors(cudaMalloc((void **)&dperm_shifter_new,
                                 sizeof(cytnx_uint64) * permuted_shifter_new.size()));

      /// copy psn-vec/so-vec to device
      checkCudaErrors(cudaMemcpy(dperm_shifter_new, &permuted_shifter_new[0],
                                 sizeof(cytnx_uint64) * permuted_shifter_new.size(),
                                 cudaMemcpyHostToDevice));
      checkCudaErrors(cudaMemcpy(dshifter_old, &shifter_old[0],
                                 sizeof(cytnx_uint64) * shifter_old.size(),
                                 cudaMemcpyHostToDevice));

      /// calculate how many blocks, and shared mem size, thpb fixed at 256 (need fine tune)
      cytnx_uint64 NBlocks = Nelem / 256;
      if (Nelem % 256) {
        NBlocks += 1;
      }
      cuMovemem_kernel<<<NBlocks, 256, shifter_old.size() * 2 * sizeof(cytnx_uint64)>>>(
        dtmp, (float *)in->Mem, dshifter_old, dperm_shifter_new, old_shape.size(), Nelem);

      /// house keeping:
      checkCudaErrors(cudaFree(dshifter_old));
      checkCudaErrors(cudaFree(dperm_shifter_new));
  #endif

      boost::intrusive_ptr<Storage_base> out(new FloatStorage());
      if (is_inplace) {
        /// cpy back:
        checkCudaErrors(cudaMemcpy(in->Mem, dtmp, sizeof(float) * Nelem, cudaMemcpyDeviceToDevice));
        cudaFree(dtmp);
        return out;

      } else {
        out->_Init_byptr(dtmp, Nelem, in->device, true, in->cap);
        return out;
      }
    }

    boost::intrusive_ptr<Storage_base> cuTNPerm_gpu_i64(boost::intrusive_ptr<Storage_base> &in,
                                                        const std::vector<cytnx_uint64> &old_shape,
                                                        const std::vector<cytnx_uint64> &mapper,
                                                        const std::vector<cytnx_uint64> &invmapper,
                                                        const bool is_inplace) {
  #ifdef UNI_DEBUG
      cytnx_error_msg(
        in->dtype != Type.Int64,
        "[DEBUG][internal error] in.dtype_str is [%s] but call cuTNPerm_gpu with type Int64",
        in->dtype_str().c_str());
      cytnx_error_msg(in->device == Device.cpu, "%s",
                      "[DEBUG][internal error] in.device is on cpu but all cuda function.");
  #endif

      std::vector<cytnx_uint64> newshape(old_shape.size());
      for (cytnx_uint64 i = 0; i < old_shape.size(); i++) newshape[i] = old_shape[mapper[i]];

      std::vector<cytnx_uint64> shifter_old(old_shape.size());
      std::vector<cytnx_uint64> shifter_new(old_shape.size());

      cytnx_uint64 accu_old = 1, accu_new = 1;
      for (cytnx_int64 i = old_shape.size() - 1; i >= 0; i--) {
        shifter_old[i] = accu_old;
        shifter_new[i] = accu_new;
        accu_old *= old_shape[i];
        accu_new *= newshape[i];
      }
      std::vector<cytnx_uint64> old_inds(old_shape.size());

      std::vector<cytnx_uint64> permuted_shifter_new(old_shape.size());
      for (unsigned int i = 0; i < old_shape.size(); i++)
        permuted_shifter_new[i] = shifter_new[invmapper[i]];

      /// allocate a GPU for psn-vec/so-vec/tmp des-vec
      cytnx_uint64 *dshifter_old, *dperm_shifter_new;
      cytnx_int64 *dtmp;
      cytnx_uint64 Nelem = accu_old;

      cudaSetDevice(in->device);  // ensure the following allocation on the same device as src.
      checkCudaErrors(
        cudaMalloc((void **)&dshifter_old, sizeof(cytnx_uint64) * shifter_old.size()));
      checkCudaErrors(cudaMalloc((void **)&dperm_shifter_new,
                                 sizeof(cytnx_uint64) * permuted_shifter_new.size()));
      dtmp = (cytnx_int64 *)cuMalloc_gpu(sizeof(cytnx_int64) * in->cap);

      /// copy psn-vec/so-vec to device
      checkCudaErrors(cudaMemcpy(dperm_shifter_new, &permuted_shifter_new[0],
                                 sizeof(cytnx_uint64) * permuted_shifter_new.size(),
                                 cudaMemcpyHostToDevice));
      checkCudaErrors(cudaMemcpy(dshifter_old, &shifter_old[0],
                                 sizeof(cytnx_uint64) * shifter_old.size(),
                                 cudaMemcpyHostToDevice));

      /// calculate how many blocks, and shared mem size, thpb fixed at 256 (need fine tune)
      cytnx_uint64 NBlocks = Nelem / 256;
      if (Nelem % 256) {
        NBlocks += 1;
      }
      cuMovemem_kernel<<<NBlocks, 256, shifter_old.size() * 2 * sizeof(cytnx_uint64)>>>(
        dtmp, (cytnx_int64 *)in->Mem, dshifter_old, dperm_shifter_new, old_shape.size(), Nelem);

      /// house keeping:
      checkCudaErrors(cudaFree(dshifter_old));
      checkCudaErrors(cudaFree(dperm_shifter_new));

      boost::intrusive_ptr<Storage_base> out(new Int64Storage());
      if (is_inplace) {
        /// cpy back:
        checkCudaErrors(
          cudaMemcpy(in->Mem, dtmp, sizeof(cytnx_int64) * Nelem, cudaMemcpyDeviceToDevice));
        cudaFree(dtmp);
        return out;

      } else {
        out->_Init_byptr(dtmp, Nelem, in->device, true, in->cap);
        return out;
      }
    }

    boost::intrusive_ptr<Storage_base> cuTNPerm_gpu_u64(boost::intrusive_ptr<Storage_base> &in,
                                                        const std::vector<cytnx_uint64> &old_shape,
                                                        const std::vector<cytnx_uint64> &mapper,
                                                        const std::vector<cytnx_uint64> &invmapper,
                                                        const bool is_inplace) {
  #ifdef UNI_DEBUG
      cytnx_error_msg(
        in->dtype != Type.Uint64,
        "[DEBUG][internal error] in.dtype_str is [%s] but call cuTNPerm_gpu with type Uint64",
        in->dtype_str().c_str());
      cytnx_error_msg(in->device == Device.cpu, "%s",
                      "[DEBUG][internal error] in.device is on cpu but all cuda function.");
  #endif

      std::vector<cytnx_uint64> newshape(old_shape.size());
      for (cytnx_uint64 i = 0; i < old_shape.size(); i++) newshape[i] = old_shape[mapper[i]];

      std::vector<cytnx_uint64> shifter_old(old_shape.size());
      std::vector<cytnx_uint64> shifter_new(old_shape.size());

      cytnx_uint64 accu_old = 1, accu_new = 1;
      for (cytnx_int64 i = old_shape.size() - 1; i >= 0; i--) {
        shifter_old[i] = accu_old;
        shifter_new[i] = accu_new;
        accu_old *= old_shape[i];
        accu_new *= newshape[i];
      }
      std::vector<cytnx_uint64> old_inds(old_shape.size());

      std::vector<cytnx_uint64> permuted_shifter_new(old_shape.size());
      for (unsigned int i = 0; i < old_shape.size(); i++)
        permuted_shifter_new[i] = shifter_new[invmapper[i]];

      /// allocate a GPU for psn-vec/so-vec/tmp des-vec
      cytnx_uint64 *dshifter_old, *dperm_shifter_new;
      cytnx_uint64 *dtmp;
      cytnx_uint64 Nelem = accu_old;

      cudaSetDevice(in->device);  // ensure the following allocation on the same device as src.
      checkCudaErrors(
        cudaMalloc((void **)&dshifter_old, sizeof(cytnx_uint64) * shifter_old.size()));
      checkCudaErrors(cudaMalloc((void **)&dperm_shifter_new,
                                 sizeof(cytnx_uint64) * permuted_shifter_new.size()));
      dtmp = (cytnx_uint64 *)cuMalloc_gpu(sizeof(cytnx_uint64) * in->cap);

      /// copy psn-vec/so-vec to device
      checkCudaErrors(cudaMemcpy(dperm_shifter_new, &permuted_shifter_new[0],
                                 sizeof(cytnx_uint64) * permuted_shifter_new.size(),
                                 cudaMemcpyHostToDevice));
      checkCudaErrors(cudaMemcpy(dshifter_old, &shifter_old[0],
                                 sizeof(cytnx_uint64) * shifter_old.size(),
                                 cudaMemcpyHostToDevice));

      /// calculate how many blocks, and shared mem size, thpb fixed at 256 (need fine tune)
      cytnx_uint64 NBlocks = Nelem / 256;
      if (Nelem % 256) {
        NBlocks += 1;
      }
      cuMovemem_kernel<<<NBlocks, 256, shifter_old.size() * 2 * sizeof(cytnx_uint64)>>>(
        dtmp, (cytnx_uint64 *)in->Mem, dshifter_old, dperm_shifter_new, old_shape.size(), Nelem);

      /// house keeping:
      checkCudaErrors(cudaFree(dshifter_old));
      checkCudaErrors(cudaFree(dperm_shifter_new));

      boost::intrusive_ptr<Storage_base> out(new Uint64Storage());
      if (is_inplace) {
        /// cpy back:
        checkCudaErrors(
          cudaMemcpy(in->Mem, dtmp, sizeof(cytnx_uint64) * Nelem, cudaMemcpyDeviceToDevice));
        cudaFree(dtmp);
        return out;

      } else {
        out->_Init_byptr(dtmp, Nelem, in->device, true, in->cap);
        return out;
      }
    }

    boost::intrusive_ptr<Storage_base> cuTNPerm_gpu_i32(boost::intrusive_ptr<Storage_base> &in,
                                                        const std::vector<cytnx_uint64> &old_shape,
                                                        const std::vector<cytnx_uint64> &mapper,
                                                        const std::vector<cytnx_uint64> &invmapper,
                                                        const bool is_inplace) {
  #ifdef UNI_DEBUG
      cytnx_error_msg(
        in->dtype != Type.Int32,
        "[DEBUG][internal error] in.dtype_str is [%s] but call cuTNPerm_gpu with type Int32",
        in->dtype_str().c_str());
      cytnx_error_msg(in->device == Device.cpu, "%s",
                      "[DEBUG][internal error] in.device is on cpu but all cuda function.");
  #endif

      std::vector<cytnx_uint64> newshape(old_shape.size());
      for (cytnx_uint64 i = 0; i < old_shape.size(); i++) newshape[i] = old_shape[mapper[i]];

      std::vector<cytnx_uint64> shifter_old(old_shape.size());
      std::vector<cytnx_uint64> shifter_new(old_shape.size());

      cytnx_uint64 accu_old = 1, accu_new = 1;
      for (cytnx_int64 i = old_shape.size() - 1; i >= 0; i--) {
        shifter_old[i] = accu_old;
        shifter_new[i] = accu_new;
        accu_old *= old_shape[i];
        accu_new *= newshape[i];
      }
      std::vector<cytnx_uint64> old_inds(old_shape.size());

      std::vector<cytnx_uint64> permuted_shifter_new(old_shape.size());
      for (unsigned int i = 0; i < old_shape.size(); i++)
        permuted_shifter_new[i] = shifter_new[invmapper[i]];

      /// allocate a GPU for psn-vec/so-vec/tmp des-vec
      cytnx_uint64 *dshifter_old, *dperm_shifter_new;
      cytnx_int32 *dtmp;
      cytnx_uint64 Nelem = accu_old;

      cudaSetDevice(in->device);  // ensure the following allocation on the same device as src.
      checkCudaErrors(
        cudaMalloc((void **)&dshifter_old, sizeof(cytnx_uint64) * shifter_old.size()));
      checkCudaErrors(cudaMalloc((void **)&dperm_shifter_new,
                                 sizeof(cytnx_uint64) * permuted_shifter_new.size()));
      dtmp = (cytnx_int32 *)cuMalloc_gpu(sizeof(cytnx_int32) * in->cap);

      /// copy psn-vec/so-vec to device
      checkCudaErrors(cudaMemcpy(dperm_shifter_new, &permuted_shifter_new[0],
                                 sizeof(cytnx_uint64) * permuted_shifter_new.size(),
                                 cudaMemcpyHostToDevice));
      checkCudaErrors(cudaMemcpy(dshifter_old, &shifter_old[0],
                                 sizeof(cytnx_uint64) * shifter_old.size(),
                                 cudaMemcpyHostToDevice));

      /// calculate how many blocks, and shared mem size, thpb fixed at 256 (need fine tune)
      cytnx_uint64 NBlocks = Nelem / 256;
      if (Nelem % 256) {
        NBlocks += 1;
      }
      cuMovemem_kernel<<<NBlocks, 256, shifter_old.size() * 2 * sizeof(cytnx_uint64)>>>(
        dtmp, (cytnx_int32 *)in->Mem, dshifter_old, dperm_shifter_new, old_shape.size(), Nelem);

      /// house keeping:
      checkCudaErrors(cudaFree(dshifter_old));
      checkCudaErrors(cudaFree(dperm_shifter_new));

      boost::intrusive_ptr<Storage_base> out(new Int32Storage());
      if (is_inplace) {
        /// cpy back:
        checkCudaErrors(
          cudaMemcpy(in->Mem, dtmp, sizeof(cytnx_int32) * Nelem, cudaMemcpyDeviceToDevice));
        cudaFree(dtmp);
        return out;

      } else {
        out->_Init_byptr(dtmp, Nelem, in->device, true, in->cap);
        return out;
      }
    }

    boost::intrusive_ptr<Storage_base> cuTNPerm_gpu_u32(boost::intrusive_ptr<Storage_base> &in,
                                                        const std::vector<cytnx_uint64> &old_shape,
                                                        const std::vector<cytnx_uint64> &mapper,
                                                        const std::vector<cytnx_uint64> &invmapper,
                                                        const bool is_inplace) {
  #ifdef UNI_DEBUG
      cytnx_error_msg(
        in->dtype != Type.Uint32,
        "[DEBUG][internal error] in.dtype_str is [%s] but call cuTNPerm_gpu with type Uint32",
        in->dtype_str().c_str());
      cytnx_error_msg(in->device == Device.cpu, "%s",
                      "[DEBUG][internal error] in.device is on cpu but all cuda function.");
  #endif

      std::vector<cytnx_uint64> newshape(old_shape.size());
      for (cytnx_uint64 i = 0; i < old_shape.size(); i++) newshape[i] = old_shape[mapper[i]];

      std::vector<cytnx_uint64> shifter_old(old_shape.size());
      std::vector<cytnx_uint64> shifter_new(old_shape.size());

      cytnx_uint64 accu_old = 1, accu_new = 1;
      for (cytnx_int64 i = old_shape.size() - 1; i >= 0; i--) {
        shifter_old[i] = accu_old;
        shifter_new[i] = accu_new;
        accu_old *= old_shape[i];
        accu_new *= newshape[i];
      }
      std::vector<cytnx_uint64> old_inds(old_shape.size());

      std::vector<cytnx_uint64> permuted_shifter_new(old_shape.size());
      for (unsigned int i = 0; i < old_shape.size(); i++)
        permuted_shifter_new[i] = shifter_new[invmapper[i]];

      /// allocate a GPU for psn-vec/so-vec/tmp des-vec
      cytnx_uint64 *dshifter_old, *dperm_shifter_new;
      cytnx_uint32 *dtmp;
      cytnx_uint64 Nelem = accu_old;

      cudaSetDevice(in->device);  // ensure the following allocation on the same device as src.
      checkCudaErrors(
        cudaMalloc((void **)&dshifter_old, sizeof(cytnx_uint64) * shifter_old.size()));
      checkCudaErrors(cudaMalloc((void **)&dperm_shifter_new,
                                 sizeof(cytnx_uint64) * permuted_shifter_new.size()));
      dtmp = (cytnx_uint32 *)cuMalloc_gpu(sizeof(cytnx_uint32) * in->cap);

      /// copy psn-vec/so-vec to device
      checkCudaErrors(cudaMemcpy(dperm_shifter_new, &permuted_shifter_new[0],
                                 sizeof(cytnx_uint64) * permuted_shifter_new.size(),
                                 cudaMemcpyHostToDevice));
      checkCudaErrors(cudaMemcpy(dshifter_old, &shifter_old[0],
                                 sizeof(cytnx_uint64) * shifter_old.size(),
                                 cudaMemcpyHostToDevice));

      /// calculate how many blocks, and shared mem size, thpb fixed at 256 (need fine tune)
      cytnx_uint64 NBlocks = Nelem / 256;
      if (Nelem % 256) {
        NBlocks += 1;
      }
      cuMovemem_kernel<<<NBlocks, 256, shifter_old.size() * 2 * sizeof(cytnx_uint64)>>>(
        dtmp, (cytnx_uint32 *)in->Mem, dshifter_old, dperm_shifter_new, old_shape.size(), Nelem);

      /// house keeping:
      checkCudaErrors(cudaFree(dshifter_old));
      checkCudaErrors(cudaFree(dperm_shifter_new));

      boost::intrusive_ptr<Storage_base> out(new Uint32Storage());
      if (is_inplace) {
        /// cpy back:
        checkCudaErrors(
          cudaMemcpy(in->Mem, dtmp, sizeof(cytnx_uint32) * Nelem, cudaMemcpyDeviceToDevice));
        cudaFree(dtmp);
        return out;

      } else {
        out->_Init_byptr(dtmp, Nelem, in->device, true, in->cap);
        return out;
      }
    }
    boost::intrusive_ptr<Storage_base> cuTNPerm_gpu_u16(boost::intrusive_ptr<Storage_base> &in,
                                                        const std::vector<cytnx_uint64> &old_shape,
                                                        const std::vector<cytnx_uint64> &mapper,
                                                        const std::vector<cytnx_uint64> &invmapper,
                                                        const bool is_inplace) {
  #ifdef UNI_DEBUG
      cytnx_error_msg(
        in->dtype != Type.Uint16,
        "[DEBUG][internal error] in.dtype_str is [%s] but call cuTNPerm_gpu with type Uint16",
        in->dtype_str().c_str());
      cytnx_error_msg(in->device == Device.cpu, "%s",
                      "[DEBUG][internal error] in.device is on cpu but all cuda function.");
  #endif

      std::vector<cytnx_uint64> newshape(old_shape.size());
      for (cytnx_uint64 i = 0; i < old_shape.size(); i++) newshape[i] = old_shape[mapper[i]];

      std::vector<cytnx_uint64> shifter_old(old_shape.size());
      std::vector<cytnx_uint64> shifter_new(old_shape.size());

      cytnx_uint64 accu_old = 1, accu_new = 1;
      for (cytnx_int64 i = old_shape.size() - 1; i >= 0; i--) {
        shifter_old[i] = accu_old;
        shifter_new[i] = accu_new;
        accu_old *= old_shape[i];
        accu_new *= newshape[i];
      }
      std::vector<cytnx_uint64> old_inds(old_shape.size());

      std::vector<cytnx_uint64> permuted_shifter_new(old_shape.size());
      for (unsigned int i = 0; i < old_shape.size(); i++)
        permuted_shifter_new[i] = shifter_new[invmapper[i]];

      /// allocate a GPU for psn-vec/so-vec/tmp des-vec
      cytnx_uint64 *dshifter_old, *dperm_shifter_new;
      cytnx_uint16 *dtmp;
      cytnx_uint64 Nelem = accu_old;

      cudaSetDevice(in->device);  // ensure the following allocation on the same device as src.
      checkCudaErrors(
        cudaMalloc((void **)&dshifter_old, sizeof(cytnx_uint64) * shifter_old.size()));
      checkCudaErrors(cudaMalloc((void **)&dperm_shifter_new,
                                 sizeof(cytnx_uint64) * permuted_shifter_new.size()));
      dtmp = (cytnx_uint16 *)cuMalloc_gpu(sizeof(cytnx_uint16) * in->cap);

      /// copy psn-vec/so-vec to device
      checkCudaErrors(cudaMemcpy(dperm_shifter_new, &permuted_shifter_new[0],
                                 sizeof(cytnx_uint64) * permuted_shifter_new.size(),
                                 cudaMemcpyHostToDevice));
      checkCudaErrors(cudaMemcpy(dshifter_old, &shifter_old[0],
                                 sizeof(cytnx_uint64) * shifter_old.size(),
                                 cudaMemcpyHostToDevice));

      /// calculate how many blocks, and shared mem size, thpb fixed at 256 (need fine tune)
      cytnx_uint64 NBlocks = Nelem / 256;
      if (Nelem % 256) {
        NBlocks += 1;
      }
      cuMovemem_kernel<<<NBlocks, 256, shifter_old.size() * 2 * sizeof(cytnx_uint64)>>>(
        dtmp, (cytnx_uint16 *)in->Mem, dshifter_old, dperm_shifter_new, old_shape.size(), Nelem);

      /// house keeping:
      checkCudaErrors(cudaFree(dshifter_old));
      checkCudaErrors(cudaFree(dperm_shifter_new));

      boost::intrusive_ptr<Storage_base> out(new Uint16Storage());
      if (is_inplace) {
        /// cpy back:
        checkCudaErrors(
          cudaMemcpy(in->Mem, dtmp, sizeof(cytnx_uint16) * Nelem, cudaMemcpyDeviceToDevice));
        cudaFree(dtmp);
        return out;

      } else {
        out->_Init_byptr(dtmp, Nelem, in->device, true, in->cap);
        return out;
      }
    }
    boost::intrusive_ptr<Storage_base> cuTNPerm_gpu_i16(boost::intrusive_ptr<Storage_base> &in,
                                                        const std::vector<cytnx_uint64> &old_shape,
                                                        const std::vector<cytnx_uint64> &mapper,
                                                        const std::vector<cytnx_uint64> &invmapper,
                                                        const bool is_inplace) {
  #ifdef UNI_DEBUG
      cytnx_error_msg(
        in->dtype != Type.Int16,
        "[DEBUG][internal error] in.dtype_str is [%s] but call cuTNPerm_gpu with type Int16",
        in->dtype_str().c_str());
      cytnx_error_msg(in->device == Device.cpu, "%s",
                      "[DEBUG][internal error] in.device is on cpu but all cuda function.");
  #endif

      std::vector<cytnx_uint64> newshape(old_shape.size());
      for (cytnx_uint64 i = 0; i < old_shape.size(); i++) newshape[i] = old_shape[mapper[i]];

      std::vector<cytnx_uint64> shifter_old(old_shape.size());
      std::vector<cytnx_uint64> shifter_new(old_shape.size());

      cytnx_uint64 accu_old = 1, accu_new = 1;
      for (cytnx_int64 i = old_shape.size() - 1; i >= 0; i--) {
        shifter_old[i] = accu_old;
        shifter_new[i] = accu_new;
        accu_old *= old_shape[i];
        accu_new *= newshape[i];
      }
      std::vector<cytnx_uint64> old_inds(old_shape.size());

      std::vector<cytnx_uint64> permuted_shifter_new(old_shape.size());
      for (unsigned int i = 0; i < old_shape.size(); i++)
        permuted_shifter_new[i] = shifter_new[invmapper[i]];

      /// allocate a GPU for psn-vec/so-vec/tmp des-vec
      cytnx_uint64 *dshifter_old, *dperm_shifter_new;
      cytnx_int16 *dtmp;
      cytnx_uint64 Nelem = accu_old;

      cudaSetDevice(in->device);  // ensure the following allocation on the same device as src.
      checkCudaErrors(
        cudaMalloc((void **)&dshifter_old, sizeof(cytnx_uint64) * shifter_old.size()));
      checkCudaErrors(cudaMalloc((void **)&dperm_shifter_new,
                                 sizeof(cytnx_uint64) * permuted_shifter_new.size()));
      dtmp = (cytnx_int16 *)cuMalloc_gpu(sizeof(cytnx_int16) * in->cap);

      /// copy psn-vec/so-vec to device
      checkCudaErrors(cudaMemcpy(dperm_shifter_new, &permuted_shifter_new[0],
                                 sizeof(cytnx_uint64) * permuted_shifter_new.size(),
                                 cudaMemcpyHostToDevice));
      checkCudaErrors(cudaMemcpy(dshifter_old, &shifter_old[0],
                                 sizeof(cytnx_uint64) * shifter_old.size(),
                                 cudaMemcpyHostToDevice));

      /// calculate how many blocks, and shared mem size, thpb fixed at 256 (need fine tune)
      cytnx_uint64 NBlocks = Nelem / 256;
      if (Nelem % 256) {
        NBlocks += 1;
      }
      cuMovemem_kernel<<<NBlocks, 256, shifter_old.size() * 2 * sizeof(cytnx_uint64)>>>(
        dtmp, (cytnx_int16 *)in->Mem, dshifter_old, dperm_shifter_new, old_shape.size(), Nelem);

      /// house keeping:
      checkCudaErrors(cudaFree(dshifter_old));
      checkCudaErrors(cudaFree(dperm_shifter_new));

      boost::intrusive_ptr<Storage_base> out(new Int16Storage());
      if (is_inplace) {
        /// cpy back:
        checkCudaErrors(
          cudaMemcpy(in->Mem, dtmp, sizeof(cytnx_int16) * Nelem, cudaMemcpyDeviceToDevice));
        cudaFree(dtmp);
        return out;

      } else {
        out->_Init_byptr(dtmp, Nelem, in->device, true, in->cap);
        return out;
      }
    }
    boost::intrusive_ptr<Storage_base> cuTNPerm_gpu_b(boost::intrusive_ptr<Storage_base> &in,
                                                      const std::vector<cytnx_uint64> &old_shape,
                                                      const std::vector<cytnx_uint64> &mapper,
                                                      const std::vector<cytnx_uint64> &invmapper,
                                                      const bool is_inplace) {
  #ifdef UNI_DEBUG
      cytnx_error_msg(
        in->dtype != Type.Bool,
        "[DEBUG][internal error] in.dtype_str is [%s] but call cuTNPerm_gpu with type Bool",
        in->dtype_str().c_str());
      cytnx_error_msg(in->device == Device.cpu, "%s",
                      "[DEBUG][internal error] in.device is on cpu but all cuda function.");
  #endif

      std::vector<cytnx_uint64> newshape(old_shape.size());
      for (cytnx_uint64 i = 0; i < old_shape.size(); i++) newshape[i] = old_shape[mapper[i]];

      std::vector<cytnx_uint64> shifter_old(old_shape.size());
      std::vector<cytnx_uint64> shifter_new(old_shape.size());

      cytnx_uint64 accu_old = 1, accu_new = 1;
      for (cytnx_int64 i = old_shape.size() - 1; i >= 0; i--) {
        shifter_old[i] = accu_old;
        shifter_new[i] = accu_new;
        accu_old *= old_shape[i];
        accu_new *= newshape[i];
      }
      std::vector<cytnx_uint64> old_inds(old_shape.size());

      std::vector<cytnx_uint64> permuted_shifter_new(old_shape.size());
      for (unsigned int i = 0; i < old_shape.size(); i++)
        permuted_shifter_new[i] = shifter_new[invmapper[i]];

      /// allocate a GPU for psn-vec/so-vec/tmp des-vec
      cytnx_uint64 *dshifter_old, *dperm_shifter_new;
      cytnx_bool *dtmp;
      cytnx_uint64 Nelem = accu_old;

      cudaSetDevice(in->device);  // ensure the following allocation on the same device as src.
      checkCudaErrors(
        cudaMalloc((void **)&dshifter_old, sizeof(cytnx_uint64) * shifter_old.size()));
      checkCudaErrors(cudaMalloc((void **)&dperm_shifter_new,
                                 sizeof(cytnx_uint64) * permuted_shifter_new.size()));
      dtmp = (cytnx_bool *)cuMalloc_gpu(sizeof(cytnx_bool) * in->cap);

      /// copy psn-vec/so-vec to device
      checkCudaErrors(cudaMemcpy(dperm_shifter_new, &permuted_shifter_new[0],
                                 sizeof(cytnx_uint64) * permuted_shifter_new.size(),
                                 cudaMemcpyHostToDevice));
      checkCudaErrors(cudaMemcpy(dshifter_old, &shifter_old[0],
                                 sizeof(cytnx_uint64) * shifter_old.size(),
                                 cudaMemcpyHostToDevice));

      /// calculate how many blocks, and shared mem size, thpb fixed at 256 (need fine tune)
      cytnx_uint64 NBlocks = Nelem / 256;
      if (Nelem % 256) {
        NBlocks += 1;
      }
      cuMovemem_kernel<<<NBlocks, 256, shifter_old.size() * 2 * sizeof(cytnx_uint64)>>>(
        dtmp, (cytnx_bool *)in->Mem, dshifter_old, dperm_shifter_new, old_shape.size(), Nelem);

      /// house keeping:
      checkCudaErrors(cudaFree(dshifter_old));
      checkCudaErrors(cudaFree(dperm_shifter_new));

      boost::intrusive_ptr<Storage_base> out(new BoolStorage());
      if (is_inplace) {
        /// cpy back:
        checkCudaErrors(
          cudaMemcpy(in->Mem, dtmp, sizeof(cytnx_bool) * Nelem, cudaMemcpyDeviceToDevice));
        cudaFree(dtmp);
        return out;

      } else {
        out->_Init_byptr(dtmp, Nelem, in->device, true, in->cap);
        return out;
      }
    }

#endif  // UNI_GPU
  }  // namespace utils_internal
}  // namespace cytnx
