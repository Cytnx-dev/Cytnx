#include "cuQuantumGeSvd_internal.hpp"

#include <cstdint>
#include <cstdlib>
#include <vector>

#include "cytnx_error.hpp"

#ifdef UNI_GPU
  #ifdef UNI_CUQUANTUM
    #include <cuda_runtime.h>
    #include <cutensornet.h>
  #endif
#endif

namespace cytnx {
  namespace linalg_internal {

#ifdef UNI_GPU
  #ifdef UNI_CUQUANTUM

    namespace {

      void check_cuda(cudaError_t status, const char *call) {
        cytnx_error_msg(status != cudaSuccess, "[cuQuantumGeSvd] %s failed: %s\n", call,
                        cudaGetErrorString(status));
      }

      void check_cutensornet(cutensornetStatus_t status, const char *call) {
        cytnx_error_msg(status != CUTENSORNET_STATUS_SUCCESS, "[cuQuantumGeSvd] %s failed: %s\n",
                        call, cutensornetGetErrorString(status));
      }

      // Everything one cuTensorNet SVD call acquires. Released on scope exit, so that a failed
      // call, which raises, does not leak the resources acquired before it.
      struct SvdResources {
        cudaStream_t stream = nullptr;
        cutensornetHandle_t handle = nullptr;
        cutensornetTensorDescriptor_t desc_in = nullptr;
        cutensornetTensorDescriptor_t desc_u = nullptr;
        cutensornetTensorDescriptor_t desc_v = nullptr;
        cutensornetTensorSVDConfig_t svd_config = nullptr;
        cutensornetTensorSVDInfo_t svd_info = nullptr;
        cutensornetWorkspaceDescriptor_t work_desc = nullptr;
        void *device_work = nullptr;
        void *host_work = nullptr;

        SvdResources() = default;
        SvdResources(const SvdResources &) = delete;
        SvdResources &operator=(const SvdResources &) = delete;

        // Deliberately unchecked: a destructor must not raise.
        ~SvdResources() {
          if (work_desc) cutensornetDestroyWorkspaceDescriptor(work_desc);
          if (svd_info) cutensornetDestroyTensorSVDInfo(svd_info);
          if (svd_config) cutensornetDestroyTensorSVDConfig(svd_config);
          if (desc_v) cutensornetDestroyTensorDescriptor(desc_v);
          if (desc_u) cutensornetDestroyTensorDescriptor(desc_u);
          if (desc_in) cutensornetDestroyTensorDescriptor(desc_in);
          if (handle) cutensornetDestroy(handle);
          if (device_work) cudaFree(device_work);
          std::free(host_work);
          if (stream) cudaStreamDestroy(stream);
        }
      };

      // Full (untruncated) SVD Tin = U diag(S) vT with cuTensorNet's GESVDJ. `data_type` is the
      // CUDA data type of Tin, U and vT; S has the matching real type. U, S and vT must be
      // allocated by the caller as contiguous tensors of shapes {rows, k}, {k} and {k, cols} with
      // k = min(rows, cols).
      void cuquantum_gesvd(const Tensor &Tin, Tensor &U, Tensor &S, Tensor &vT,
                           cudaDataType_t data_type) {
        check_cuda(cudaSetDevice(Tin.device()), "cudaSetDevice");

        SvdResources res;
        check_cuda(cudaStreamCreate(&res.stream), "cudaStreamCreate");
        check_cutensornet(cutensornetCreate(&res.handle), "cutensornetCreate");

        // A row-major Cytnx matrix is a column-major (Fortran layout, which cuTensorNet assumes
        // for null strides) tensor with its modes in reverse order.
        const std::vector<int32_t> modes_in{'j', 'i'};
        const std::vector<int32_t> modes_u{'s', 'i'};
        const std::vector<int32_t> modes_v{'j', 's'};
        const std::vector<int64_t> extent_in{static_cast<int64_t>(Tin.shape()[1]),
                                             static_cast<int64_t>(Tin.shape()[0])};
        const std::vector<int64_t> extent_u{static_cast<int64_t>(U.shape()[1]),
                                            static_cast<int64_t>(U.shape()[0])};
        const std::vector<int64_t> extent_v{static_cast<int64_t>(vT.shape()[1]),
                                            static_cast<int64_t>(vT.shape()[0])};
        const int64_t *strides = nullptr;

        check_cutensornet(
          cutensornetCreateTensorDescriptor(res.handle, modes_in.size(), extent_in.data(), strides,
                                            modes_in.data(), data_type, &res.desc_in),
          "cutensornetCreateTensorDescriptor");
        check_cutensornet(
          cutensornetCreateTensorDescriptor(res.handle, modes_u.size(), extent_u.data(), strides,
                                            modes_u.data(), data_type, &res.desc_u),
          "cutensornetCreateTensorDescriptor");
        check_cutensornet(
          cutensornetCreateTensorDescriptor(res.handle, modes_v.size(), extent_v.data(), strides,
                                            modes_v.data(), data_type, &res.desc_v),
          "cutensornetCreateTensorDescriptor");

        // No truncation is requested from cuTensorNet. When it truncates by value it also repacks
        // U to the reduced extent, which no longer matches the shape of the tensor U; the callers
        // truncate the full result with cudaMemcpyTruncation instead, as they do for cuSOLVER.
        check_cutensornet(cutensornetCreateTensorSVDConfig(res.handle, &res.svd_config),
                          "cutensornetCreateTensorSVDConfig");
        const double no_cutoff = 0;
        check_cutensornet(cutensornetTensorSVDConfigSetAttribute(
                            res.handle, res.svd_config, CUTENSORNET_TENSOR_SVD_CONFIG_ABS_CUTOFF,
                            &no_cutoff, sizeof(no_cutoff)),
                          "cutensornetTensorSVDConfigSetAttribute(ABS_CUTOFF)");
        check_cutensornet(cutensornetTensorSVDConfigSetAttribute(
                            res.handle, res.svd_config, CUTENSORNET_TENSOR_SVD_CONFIG_REL_CUTOFF,
                            &no_cutoff, sizeof(no_cutoff)),
                          "cutensornetTensorSVDConfigSetAttribute(REL_CUTOFF)");
        const cutensornetTensorSVDAlgo_t svd_algo = CUTENSORNET_TENSOR_SVD_ALGO_GESVDJ;
        check_cutensornet(cutensornetTensorSVDConfigSetAttribute(res.handle, res.svd_config,
                                                                 CUTENSORNET_TENSOR_SVD_CONFIG_ALGO,
                                                                 &svd_algo, sizeof(svd_algo)),
                          "cutensornetTensorSVDConfigSetAttribute(ALGO)");
        const cutensornetGesvdjParams_t gesvdj_params{/*tol=*/0, /*maxSweeps=*/80};
        check_cutensornet(cutensornetTensorSVDConfigSetAttribute(
                            res.handle, res.svd_config, CUTENSORNET_TENSOR_SVD_CONFIG_ALGO_PARAMS,
                            &gesvdj_params, sizeof(gesvdj_params)),
                          "cutensornetTensorSVDConfigSetAttribute(ALGO_PARAMS)");

        check_cutensornet(cutensornetCreateTensorSVDInfo(res.handle, &res.svd_info),
                          "cutensornetCreateTensorSVDInfo");

        check_cutensornet(cutensornetCreateWorkspaceDescriptor(res.handle, &res.work_desc),
                          "cutensornetCreateWorkspaceDescriptor");
        check_cutensornet(
          cutensornetWorkspaceComputeSVDSizes(res.handle, res.desc_in, res.desc_u, res.desc_v,
                                              res.svd_config, res.work_desc),
          "cutensornetWorkspaceComputeSVDSizes");
        int64_t device_work_size = 0;
        int64_t host_work_size = 0;
        check_cutensornet(
          cutensornetWorkspaceGetMemorySize(
            res.handle, res.work_desc, CUTENSORNET_WORKSIZE_PREF_RECOMMENDED,
            CUTENSORNET_MEMSPACE_DEVICE, CUTENSORNET_WORKSPACE_SCRATCH, &device_work_size),
          "cutensornetWorkspaceGetMemorySize(DEVICE)");
        check_cutensornet(
          cutensornetWorkspaceGetMemorySize(
            res.handle, res.work_desc, CUTENSORNET_WORKSIZE_PREF_RECOMMENDED,
            CUTENSORNET_MEMSPACE_HOST, CUTENSORNET_WORKSPACE_SCRATCH, &host_work_size),
          "cutensornetWorkspaceGetMemorySize(HOST)");
        if (device_work_size > 0) {
          check_cuda(cudaMalloc(&res.device_work, device_work_size), "cudaMalloc");
        }
        if (host_work_size > 0) {
          res.host_work = std::malloc(host_work_size);
          cytnx_error_msg(res.host_work == nullptr,
                          "[cuQuantumGeSvd] cannot allocate %lld bytes of host workspace.\n",
                          static_cast<long long>(host_work_size));
        }
        check_cutensornet(cutensornetWorkspaceSetMemory(
                            res.handle, res.work_desc, CUTENSORNET_MEMSPACE_DEVICE,
                            CUTENSORNET_WORKSPACE_SCRATCH, res.device_work, device_work_size),
                          "cutensornetWorkspaceSetMemory(DEVICE)");
        check_cutensornet(cutensornetWorkspaceSetMemory(
                            res.handle, res.work_desc, CUTENSORNET_MEMSPACE_HOST,
                            CUTENSORNET_WORKSPACE_SCRATCH, res.host_work, host_work_size),
                          "cutensornetWorkspaceSetMemory(HOST)");

        check_cutensornet(cutensornetTensorSVD(
                            res.handle, res.desc_in, Tin._impl->storage()._impl->data(), res.desc_u,
                            U._impl->storage()._impl->data(), S._impl->storage()._impl->data(),
                            res.desc_v, vT._impl->storage()._impl->data(), res.svd_config,
                            res.svd_info, res.work_desc, res.stream),
                          "cutensornetTensorSVD");
        check_cuda(cudaStreamSynchronize(res.stream), "cudaStreamSynchronize");

        // U and vT are only valid in the shapes they were allocated with if cuTensorNet kept the
        // full shared extent.
        int64_t reduced_extent = 0;
        check_cutensornet(cutensornetTensorSVDInfoGetAttribute(
                            res.handle, res.svd_info, CUTENSORNET_TENSOR_SVD_INFO_REDUCED_EXTENT,
                            &reduced_extent, sizeof(reduced_extent)),
                          "cutensornetTensorSVDInfoGetAttribute(REDUCED_EXTENT)");
        cytnx_error_msg(reduced_extent != static_cast<int64_t>(S.shape()[0]),
                        "[cuQuantumGeSvd] cuTensorNet returned %lld singular values where %llu "
                        "were requested.\n",
                        static_cast<long long>(reduced_extent),
                        static_cast<unsigned long long>(S.shape()[0]));
      }

    }  // namespace

    void cuQuantumGeSvd_internal_cd(const Tensor &Tin, Tensor &U, Tensor &S, Tensor &vT) {
      cuquantum_gesvd(Tin, U, S, vT, CUDA_C_64F);
    }

    void cuQuantumGeSvd_internal_cf(const Tensor &Tin, Tensor &U, Tensor &S, Tensor &vT) {
      cuquantum_gesvd(Tin, U, S, vT, CUDA_C_32F);
    }

    void cuQuantumGeSvd_internal_d(const Tensor &Tin, Tensor &U, Tensor &S, Tensor &vT) {
      cuquantum_gesvd(Tin, U, S, vT, CUDA_R_64F);
    }

    void cuQuantumGeSvd_internal_f(const Tensor &Tin, Tensor &U, Tensor &S, Tensor &vT) {
      cuquantum_gesvd(Tin, U, S, vT, CUDA_R_32F);
    }

  #endif
#endif

  }  // namespace linalg_internal
}  // namespace cytnx
