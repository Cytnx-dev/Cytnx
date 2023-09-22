#include <stdlib.h>
#include <stdio.h>

#include <unordered_map>
#include <vector>
#include <cassert>

#include "cuQuantumGeSvd_internal.hpp"

#ifdef UNI_GPU
  #ifdef UNI_CUQUANTUM
    #include <cuda_runtime.h>
    #include <cutensornet.h>

  //  #define HANDLE_ERROR(x)                                           \
  //  { const auto err = x;                                             \
  //  if( err != CUTENSORNET_STATUS_SUCCESS )                           \
  //  { printf("Error: %s in line %d\n", cutensornetGetErrorString(err), __LINE__); return err; } \
  //  };

  //  #define HANDLE_CUDA_ERROR(x)                                      \
  //  {  const auto err = x;                                            \
  //     if( err != cudaSuccess )                                       \
  //     { printf("Error: %s in line %d\n", cudaGetErrorString(err), __LINE__); return err; } \
  //  };
    #define HANDLE_ERROR(x)                                                           \
      {                                                                               \
        const auto err = x;                                                           \
        if (err != CUTENSORNET_STATUS_SUCCESS) {                                      \
          printf("Error: %s in line %d\n", cutensornetGetErrorString(err), __LINE__); \
          fflush(stdout);                                                             \
        }                                                                             \
      };

    #define HANDLE_CUDA_ERROR(x)                                                    \
      {                                                                             \
        const auto err = x;                                                         \
        if (err != cudaSuccess) {                                                   \
          printf("CUDA Error: %s in line %d\n", cudaGetErrorString(err), __LINE__); \
          fflush(stdout);                                                           \
        }                                                                           \
      };

struct GPUTimer {
  GPUTimer(cudaStream_t stream) : stream_(stream) {
    cudaEventCreate(&start_);
    cudaEventCreate(&stop_);
  }

  ~GPUTimer() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }

  void start() { cudaEventRecord(start_, stream_); }

  float seconds() {
    cudaEventRecord(stop_, stream_);
    cudaEventSynchronize(stop_);
    float time;
    cudaEventElapsedTime(&time, start_, stop_);
    return time * 1e-3;
  }

 private:
  cudaEvent_t start_, stop_;
  cudaStream_t stream_;
};

  #endif
#endif

// int64_t computeCombinedExtent(const std::unordered_map<int32_t, int64_t> &extentMap,
//                               const std::vector<int32_t> &modes) {
//   int64_t combinedExtent{1};
//   for (auto mode : modes) {
//     auto it = extentMap.find(mode);
//     if (it != extentMap.end()) combinedExtent *= it->second;
//   }
//   return combinedExtent;
// }

namespace cytnx {
  namespace linalg_internal {

#ifdef UNI_GPU
  #ifdef UNI_CUQUANTUM

    void memcpy_truncation_cd(Tensor &U, Tensor &vT, Tensor &S, Tensor &terr,
                              const cytnx_uint64 &truc_dim, const bool &is_U, const bool &is_vT,
                              const unsigned int &return_err) {
      Tensor newU = Tensor({U.shape()[0], truc_dim}, U.dtype(), U.device());
      Tensor newvT = Tensor({truc_dim, vT.shape()[1]}, vT.dtype(), vT.device());
      Tensor newS = Tensor({truc_dim}, S.dtype(), S.device());
      HANDLE_CUDA_ERROR(cudaMemcpy((cytnx_double *)newS._impl->storage()._impl->Mem,
                                   (cytnx_double *)S._impl->storage()._impl->Mem,
                                   truc_dim * sizeof(cytnx_double), cudaMemcpyDeviceToDevice));
      if (is_U) {
        int src = 0;
        int dest = 0;
        // copy with strides.
        for (int i = 0; i < U.shape()[0]; i++) {
          HANDLE_CUDA_ERROR(cudaMemcpy((cytnx_complex128 *)newU._impl->storage()._impl->Mem + src,
                                       (cytnx_complex128 *)U._impl->storage()._impl->Mem + dest,
                                       truc_dim * sizeof(cytnx_complex128),
                                       cudaMemcpyDeviceToDevice));
          src += truc_dim;
          dest += U.shape()[1];
        }
        U = newU;
      }
      if (is_vT) {
        // simply copy a new one dropping the tail.
        HANDLE_CUDA_ERROR(cudaMemcpy((cytnx_complex128 *)newvT._impl->storage()._impl->Mem,
                                     (cytnx_complex128 *)vT._impl->storage()._impl->Mem,
                                     vT.shape()[1] * truc_dim * sizeof(cytnx_complex128),
                                     cudaMemcpyDeviceToDevice));
        vT = newvT;
      }
      if (return_err == 1) {
        Tensor newterr = Tensor({1}, S.dtype(), S.device());
        ((cytnx_double *)newterr._impl->storage()._impl->Mem)[0] =
          ((cytnx_double *)S._impl->storage()._impl->Mem)[truc_dim];
        terr = newterr;
      } else if (return_err) {
        cytnx_uint64 discared_dim = S.shape()[0] - truc_dim;
        Tensor newterr = Tensor({discared_dim}, S.dtype(), S.device());
        HANDLE_CUDA_ERROR(cudaMemcpy((cytnx_double *)newterr._impl->storage()._impl->Mem,
                                     (cytnx_double *)S._impl->storage()._impl->Mem + truc_dim,
                                     discared_dim * sizeof(cytnx_double),
                                     cudaMemcpyDeviceToDevice));
        terr = newterr;
      }
      S = newS;
    }

    void memcpy_truncation_cf(Tensor &U, Tensor &vT, Tensor &S, Tensor &terr,
                              const cytnx_uint64 &truc_dim, const bool &is_U, const bool &is_vT,
                              const unsigned int &return_err) {
      Tensor newU = Tensor({U.shape()[0], truc_dim}, U.dtype(), U.device());
      Tensor newvT = Tensor({truc_dim, vT.shape()[1]}, vT.dtype(), vT.device());
      Tensor newS = Tensor({truc_dim}, S.dtype(), S.device());
      HANDLE_CUDA_ERROR(cudaMemcpy((cytnx_double *)newS._impl->storage()._impl->Mem,
                                   (cytnx_double *)S._impl->storage()._impl->Mem,
                                   truc_dim * sizeof(cytnx_double), cudaMemcpyDeviceToDevice));
      if (is_U) {
        int src = 0;
        int dest = 0;
        // copy with strides.
        for (int i = 0; i < U.shape()[0]; i++) {
          HANDLE_CUDA_ERROR(cudaMemcpy((cytnx_complex64 *)newU._impl->storage()._impl->Mem + src,
                                       (cytnx_complex64 *)U._impl->storage()._impl->Mem + dest,
                                       truc_dim * sizeof(cytnx_complex64),
                                       cudaMemcpyDeviceToDevice));
          src += truc_dim;
          dest += U.shape()[1];
        }
        U = newU;
      }
      if (is_vT) {
        // simply copy a new one dropping the tail.
        HANDLE_CUDA_ERROR(cudaMemcpy((cytnx_complex64 *)newvT._impl->storage()._impl->Mem,
                                     (cytnx_complex64 *)vT._impl->storage()._impl->Mem,
                                     vT.shape()[1] * truc_dim * sizeof(cytnx_complex64),
                                     cudaMemcpyDeviceToDevice));
        vT = newvT;
      }
      if (return_err == 1) {
        Tensor newterr = Tensor({1}, S.dtype(), S.device());
        ((cytnx_double *)newterr._impl->storage()._impl->Mem)[0] =
          ((cytnx_double *)S._impl->storage()._impl->Mem)[truc_dim];
        terr = newterr;
      } else if (return_err) {
        cytnx_uint64 discared_dim = S.shape()[0] - truc_dim;
        Tensor newterr = Tensor({discared_dim}, S.dtype(), S.device());
        HANDLE_CUDA_ERROR(cudaMemcpy((cytnx_double *)newterr._impl->storage()._impl->Mem,
                                     (cytnx_double *)S._impl->storage()._impl->Mem + truc_dim,
                                     discared_dim * sizeof(cytnx_double),
                                     cudaMemcpyDeviceToDevice));
        terr = newterr;
      }
      S = newS;
    }

    void memcpy_truncation_d(Tensor &U, Tensor &vT, Tensor &S, Tensor &terr,
                             const cytnx_uint64 &truc_dim, const bool &is_U, const bool &is_vT,
                             const unsigned int &return_err) {
      Tensor newU = Tensor({U.shape()[0], truc_dim}, U.dtype(), U.device());
      Tensor newvT = Tensor({truc_dim, vT.shape()[1]}, vT.dtype(), vT.device());
      Tensor newS = Tensor({truc_dim}, S.dtype(), S.device());
      HANDLE_CUDA_ERROR(cudaMemcpy((cytnx_double *)newS._impl->storage()._impl->Mem,
                                   (cytnx_double *)S._impl->storage()._impl->Mem,
                                   truc_dim * sizeof(cytnx_double), cudaMemcpyDeviceToDevice));
      if (is_U) {
        int src = 0;
        int dest = 0;
        // copy with strides.
        for (int i = 0; i < U.shape()[0]; i++) {
          HANDLE_CUDA_ERROR(cudaMemcpy((cytnx_double *)newU._impl->storage()._impl->Mem + src,
                                       (cytnx_double *)U._impl->storage()._impl->Mem + dest,
                                       truc_dim * sizeof(cytnx_double), cudaMemcpyDeviceToDevice));
          src += truc_dim;
          dest += U.shape()[1];
        }
        U = newU;
      }
      if (is_vT) {
        // simply copy a new one dropping the tail.
        HANDLE_CUDA_ERROR(cudaMemcpy((cytnx_double *)newvT._impl->storage()._impl->Mem,
                                     (cytnx_double *)vT._impl->storage()._impl->Mem,
                                     vT.shape()[1] * truc_dim * sizeof(cytnx_double),
                                     cudaMemcpyDeviceToDevice));
        vT = newvT;
      }
      if (return_err == 1) {
        Tensor newterr = Tensor({1}, S.dtype(), S.device());
        ((cytnx_double *)newterr._impl->storage()._impl->Mem)[0] =
          ((cytnx_double *)S._impl->storage()._impl->Mem)[truc_dim];
        terr = newterr;
      } else if (return_err) {
        cytnx_uint64 discared_dim = S.shape()[0] - truc_dim;
        Tensor newterr = Tensor({discared_dim}, S.dtype(), S.device());
        HANDLE_CUDA_ERROR(cudaMemcpy((cytnx_double *)newterr._impl->storage()._impl->Mem,
                                     (cytnx_double *)S._impl->storage()._impl->Mem + truc_dim,
                                     discared_dim * sizeof(cytnx_double),
                                     cudaMemcpyDeviceToDevice));
        terr = newterr;
      }
      S = newS;
    }

    void memcpy_truncation_f(Tensor &U, Tensor &vT, Tensor &S, Tensor &terr,
                             const cytnx_uint64 &truc_dim, const bool &is_U, const bool &is_vT,
                             const unsigned int &return_err) {
      Tensor newU = Tensor({U.shape()[0], truc_dim}, U.dtype(), U.device());
      Tensor newvT = Tensor({truc_dim, vT.shape()[1]}, vT.dtype(), vT.device());
      Tensor newS = Tensor({truc_dim}, S.dtype(), S.device());
      HANDLE_CUDA_ERROR(cudaMemcpy((cytnx_double *)newS._impl->storage()._impl->Mem,
                                   (cytnx_double *)S._impl->storage()._impl->Mem,
                                   truc_dim * sizeof(cytnx_double), cudaMemcpyDeviceToDevice));
      if (is_U) {
        int src = 0;
        int dest = 0;
        // copy with strides.
        for (int i = 0; i < U.shape()[0]; i++) {
          HANDLE_CUDA_ERROR(cudaMemcpy((cytnx_float *)newU._impl->storage()._impl->Mem + src,
                                       (cytnx_float *)U._impl->storage()._impl->Mem + dest,
                                       truc_dim * sizeof(cytnx_float), cudaMemcpyDeviceToDevice));
          src += truc_dim;
          dest += U.shape()[1];
        }
        U = newU;
      }
      if (is_vT) {
        // simply copy a new one dropping the tail.
        HANDLE_CUDA_ERROR(cudaMemcpy((cytnx_float *)newvT._impl->storage()._impl->Mem,
                                     (cytnx_float *)vT._impl->storage()._impl->Mem,
                                     vT.shape()[1] * truc_dim * sizeof(cytnx_float),
                                     cudaMemcpyDeviceToDevice));
        vT = newvT;
      }
      if (return_err == 1) {
        Tensor newterr = Tensor({1}, S.dtype(), S.device());
        ((cytnx_double *)newterr._impl->storage()._impl->Mem)[0] =
          ((cytnx_double *)S._impl->storage()._impl->Mem)[truc_dim];
        terr = newterr;
      } else if (return_err) {
        cytnx_uint64 discared_dim = S.shape()[0] - truc_dim;
        Tensor newterr = Tensor({discared_dim}, S.dtype(), S.device());
        HANDLE_CUDA_ERROR(cudaMemcpy((cytnx_double *)newterr._impl->storage()._impl->Mem,
                                     (cytnx_double *)S._impl->storage()._impl->Mem + truc_dim,
                                     discared_dim * sizeof(cytnx_double),
                                     cudaMemcpyDeviceToDevice));
        terr = newterr;
      }
      S = newS;
    }

    /// cuGeSvd
    void cuQuantumGeSvd_internal_cd(const Tensor &Tin, const cytnx_uint64 &keepdim,
                                    const double &err, const unsigned int &return_err, Tensor &U,
                                    Tensor &S, Tensor &vT, Tensor &terr) {
      const size_t cuTensornetVersion = cutensornetGetVersion();
      // printf("cuTensorNet-vers:%ld\n", cuTensornetVersion);
      cudaDeviceProp prop;
      int deviceId = Tin.device();
      HANDLE_CUDA_ERROR(cudaSetDevice(deviceId));
      HANDLE_CUDA_ERROR(cudaGetDevice(&deviceId));
      HANDLE_CUDA_ERROR(cudaGetDeviceProperties(&prop, deviceId));

      // printf("===== device info ======\n");
      // printf("GPU-name:%s\n", prop.name);
      // printf("GPU-clock:%d\n", prop.clockRate);
      // printf("GPU-memoryClock:%d\n", prop.memoryClockRate);
      // printf("GPU-nSM:%d\n", prop.multiProcessorCount);
      // printf("GPU-major:%d\n", prop.major);
      // printf("GPU-minor:%d\n", prop.minor);
      // printf("========================\n");

      typedef float floatType;
      cudaDataType_t typeData = CUDA_C_64F;
      /******************
       * cuTensorNet
       *******************/

      cudaStream_t stream;
      HANDLE_CUDA_ERROR(cudaStreamCreate(&stream));

      cutensornetHandle_t handle;
      HANDLE_ERROR(cutensornetCreate(&handle));

      /**************************
       * Create tensor descriptors
       ***************************/
      std::vector<int32_t> modesT{'j', 'i'};
      std::vector<int32_t> modesU{'s', 'i'};
      std::vector<int32_t> modesV{'j', 's'};

      std::vector<int64_t> extentT{(int64_t)Tin.shape()[1], (int64_t)Tin.shape()[0]};
      std::vector<int64_t> extentU{(int64_t)U.shape()[1], (int64_t)U.shape()[0]};
      std::vector<int64_t> extentV{(int64_t)vT.shape()[1], (int64_t)vT.shape()[0]};

      const int32_t numModesIn = modesT.size();
      const int32_t numModesU = modesU.size();
      const int32_t numModesV = modesV.size();

      void *D_T = Tin._impl->storage()._impl->Mem;
      void *D_U = U._impl->storage()._impl->Mem;
      void *D_S = S._impl->storage()._impl->Mem;
      void *D_V = vT._impl->storage()._impl->Mem;

      cutensornetTensorDescriptor_t descTensorIn;
      cutensornetTensorDescriptor_t descTensorU;
      cutensornetTensorDescriptor_t descTensorV;

      const int64_t *strides = NULL;  // assuming fortran layout for all tensors

      HANDLE_ERROR(cutensornetCreateTensorDescriptor(handle, numModesIn, extentT.data(), strides,
                                                     modesT.data(), typeData, &descTensorIn));
      HANDLE_ERROR(cutensornetCreateTensorDescriptor(handle, numModesU, extentU.data(), strides,
                                                     modesU.data(), typeData, &descTensorU));
      HANDLE_ERROR(cutensornetCreateTensorDescriptor(handle, numModesV, extentV.data(), strides,
                                                     modesV.data(), typeData, &descTensorV));

      // printf("Initialize the cuTensorNet library and create all tensor descriptors.\n");

      /**********************************************
       * Setup SVD algorithm and truncation parameters
       ***********************************************/

      cutensornetTensorSVDConfig_t svdConfig;
      HANDLE_ERROR(cutensornetCreateTensorSVDConfig(handle, &svdConfig));
      double absCutoff;
      // set up truncation parameters
      if (return_err) {
        // do manually truncation instead, so no cuquantum truncate here
        absCutoff = 0;
      } else {
        absCutoff = err;
      }

      HANDLE_ERROR(cutensornetTensorSVDConfigSetAttribute(handle, svdConfig,
                                                          CUTENSORNET_TENSOR_SVD_CONFIG_ABS_CUTOFF,
                                                          &absCutoff, sizeof(absCutoff)));
      double relCutoff = 0;
      HANDLE_ERROR(cutensornetTensorSVDConfigSetAttribute(handle, svdConfig,
                                                          CUTENSORNET_TENSOR_SVD_CONFIG_REL_CUTOFF,
                                                          &relCutoff, sizeof(relCutoff)));

      // optional: choose gesvdj algorithm with customized parameters. Default is gesvd.
      cutensornetTensorSVDAlgo_t svdAlgo = CUTENSORNET_TENSOR_SVD_ALGO_GESVDJ;
      HANDLE_ERROR(cutensornetTensorSVDConfigSetAttribute(
        handle, svdConfig, CUTENSORNET_TENSOR_SVD_CONFIG_ALGO, &svdAlgo, sizeof(svdAlgo)));
      cutensornetGesvdjParams_t gesvdjParams{/*tol=*/0, /*maxSweeps=*/80};
      HANDLE_ERROR(cutensornetTensorSVDConfigSetAttribute(handle, svdConfig,
                                                          CUTENSORNET_TENSOR_SVD_CONFIG_ALGO_PARAMS,
                                                          &gesvdjParams, sizeof(gesvdjParams)));
      // printf("Set up SVDConfig to use GESVDJ algorithm with truncation\n");

      /********************************************************
       * Create SVDInfo to record runtime SVD truncation details
       *********************************************************/

      cutensornetTensorSVDInfo_t svdInfo;
      HANDLE_ERROR(cutensornetCreateTensorSVDInfo(handle, &svdInfo));

      // Sphinx: #6
      /**************************************************************
       * Query the required workspace sizes and allocate memory
       **************************************************************/

      cutensornetWorkspaceDescriptor_t workDesc;
      HANDLE_ERROR(cutensornetCreateWorkspaceDescriptor(handle, &workDesc));
      HANDLE_ERROR(cutensornetWorkspaceComputeSVDSizes(handle, descTensorIn, descTensorU,
                                                       descTensorV, svdConfig, workDesc));
      int64_t hostWorkspaceSize, deviceWorkspaceSize;
      // for tensor SVD, it does not matter which cutensornetWorksizePref_t we pick
      HANDLE_ERROR(cutensornetWorkspaceGetMemorySize(
        handle, workDesc, CUTENSORNET_WORKSIZE_PREF_RECOMMENDED, CUTENSORNET_MEMSPACE_DEVICE,
        CUTENSORNET_WORKSPACE_SCRATCH, &deviceWorkspaceSize));
      HANDLE_ERROR(cutensornetWorkspaceGetMemorySize(
        handle, workDesc, CUTENSORNET_WORKSIZE_PREF_RECOMMENDED, CUTENSORNET_MEMSPACE_HOST,
        CUTENSORNET_WORKSPACE_SCRATCH, &hostWorkspaceSize));

      void *devWork = nullptr, *hostWork = nullptr;
      if (deviceWorkspaceSize > 0) {
        HANDLE_CUDA_ERROR(cudaMalloc(&devWork, deviceWorkspaceSize));
      }
      if (hostWorkspaceSize > 0) {
        hostWork = malloc(hostWorkspaceSize);
      }
      HANDLE_ERROR(cutensornetWorkspaceSetMemory(handle, workDesc, CUTENSORNET_MEMSPACE_DEVICE,
                                                 CUTENSORNET_WORKSPACE_SCRATCH, devWork,
                                                 deviceWorkspaceSize));
      HANDLE_ERROR(cutensornetWorkspaceSetMemory(handle, workDesc, CUTENSORNET_MEMSPACE_HOST,
                                                 CUTENSORNET_WORKSPACE_SCRATCH, hostWork,
                                                 hostWorkspaceSize));

      // Sphinx: #7
      /**********
       * Execution
       ***********/

      GPUTimer timer{stream};
      double minTimeCUTENSOR = 1e100;
      const int numRuns = 1;  // to get stable perf results
      for (int i = 0; i < numRuns; ++i) {
        // // restore output
        // cudaMemsetAsync(D_U, 0, sizeU, stream);
        // cudaMemsetAsync(D_S, 0, sizeS, stream);
        // cudaMemsetAsync(D_V, 0, sizeV, stream);
        // cudaDeviceSynchronize();

        // With value-based truncation, `cutensornetTensorSVD` can potentially update the shared
        // extent in descTensorU/V. We here restore descTensorU/V to the original problem.
        HANDLE_ERROR(cutensornetDestroyTensorDescriptor(descTensorU));
        HANDLE_ERROR(cutensornetDestroyTensorDescriptor(descTensorV));
        HANDLE_ERROR(cutensornetCreateTensorDescriptor(handle, numModesU, extentU.data(), strides,
                                                       modesU.data(), typeData, &descTensorU));
        HANDLE_ERROR(cutensornetCreateTensorDescriptor(handle, numModesV, extentV.data(), strides,
                                                       modesV.data(), typeData, &descTensorV));

        timer.start();
        HANDLE_ERROR(cutensornetTensorSVD(handle, descTensorIn, D_T, descTensorU, D_U, D_S,
                                          descTensorV, D_V, svdConfig, svdInfo, workDesc, stream));
        // Synchronize and measure timing
        auto time = timer.seconds();
        minTimeCUTENSOR = (minTimeCUTENSOR < time) ? minTimeCUTENSOR : time;
      }

      // printf("Performing SVD\n");

      // HANDLE_CUDA_ERROR( cudaMemcpyAsync(U, D_U, sizeU, cudaMemcpyDeviceToHost) );
      // HANDLE_CUDA_ERROR( cudaMemcpyAsync(S, D_S, sizeS, cudaMemcpyDeviceToHost) );
      // HANDLE_CUDA_ERROR( cudaMemcpyAsync(V, D_V, sizeV, cudaMemcpyDeviceToHost) );

      // Sphinx: #8
      /*************************************
       * Query runtime truncation information
       **************************************/

      double discardedWeight{0};
      int64_t reducedExtent{0};
      cutensornetGesvdjStatus_t gesvdjStatus;
      cudaDeviceSynchronize();  // device synchronization.
      HANDLE_ERROR(cutensornetTensorSVDInfoGetAttribute(
        handle, svdInfo, CUTENSORNET_TENSOR_SVD_INFO_DISCARDED_WEIGHT, &discardedWeight,
        sizeof(discardedWeight)));
      HANDLE_ERROR(cutensornetTensorSVDInfoGetAttribute(handle, svdInfo,
                                                        CUTENSORNET_TENSOR_SVD_INFO_REDUCED_EXTENT,
                                                        &reducedExtent, sizeof(reducedExtent)));
      HANDLE_ERROR(cutensornetTensorSVDInfoGetAttribute(handle, svdInfo,
                                                        CUTENSORNET_TENSOR_SVD_INFO_ALGO_STATUS,
                                                        &gesvdjStatus, sizeof(gesvdjStatus)));

      printf("elapsed time: %.2f ms\n", minTimeCUTENSOR * 1000.f);
      printf("GESVDJ residual: %.4f, runtime sweeps = %d\n", gesvdjStatus.residual,
             gesvdjStatus.sweeps);
      printf("reduced extent found at runtime: %lu\n", reducedExtent);
      std::cout << "discarded weight: " << discardedWeight << std::endl;
      // printf("discarded weight: %.15f\n", discardedWeight);

      // Sphinx: #9
      /***************
       * Free resources
       ****************/

      HANDLE_ERROR(cutensornetDestroyTensorDescriptor(descTensorIn));
      HANDLE_ERROR(cutensornetDestroyTensorDescriptor(descTensorU));
      HANDLE_ERROR(cutensornetDestroyTensorDescriptor(descTensorV));
      HANDLE_ERROR(cutensornetDestroyTensorSVDConfig(svdConfig));
      HANDLE_ERROR(cutensornetDestroyTensorSVDInfo(svdInfo));
      HANDLE_ERROR(cutensornetDestroyWorkspaceDescriptor(workDesc));
      HANDLE_ERROR(cutensornetDestroy(handle));

      if (devWork) cudaFree(devWork);
      if (hostWork) free(hostWork);
      // printf("Free resource and exit.\n");

      // Manually truncation
      cytnx_uint64 Kdim = keepdim;
      cytnx_uint64 nums = S.storage().size();
      if (nums < keepdim) {
        Kdim = nums;
      }
      cytnx_uint64 truc_dim = Kdim;
      for (cytnx_int64 i = Kdim - 1; i >= 0; i--) {
        if (((cytnx_double *)S._impl->storage()._impl->Mem)[i] < err) {
          truc_dim--;
        } else {
          break;
        }
      }
      if (truc_dim == 0) {
        truc_dim = 1;
      }
      if (truc_dim != nums) {
        memcpy_truncation_cd(U, vT, S, terr, truc_dim, true, true, return_err);
      }
    }

    void cuQuantumGeSvd_internal_cf(const Tensor &Tin, const cytnx_uint64 &keepdim,
                                    const double &err, const unsigned int &return_err, Tensor &U,
                                    Tensor &S, Tensor &vT, Tensor &terr) {
      const size_t cuTensornetVersion = cutensornetGetVersion();
      // printf("cuTensorNet-vers:%ld\n", cuTensornetVersion);

      cudaDeviceProp prop;
      int deviceId = Tin.device();
      HANDLE_CUDA_ERROR(cudaSetDevice(deviceId));
      HANDLE_CUDA_ERROR(cudaGetDevice(&deviceId));
      HANDLE_CUDA_ERROR(cudaGetDeviceProperties(&prop, deviceId));

      // printf("===== device info ======\n");
      // printf("GPU-name:%s\n", prop.name);
      // printf("GPU-clock:%d\n", prop.clockRate);
      // printf("GPU-memoryClock:%d\n", prop.memoryClockRate);
      // printf("GPU-nSM:%d\n", prop.multiProcessorCount);
      // printf("GPU-major:%d\n", prop.major);
      // printf("GPU-minor:%d\n", prop.minor);
      // printf("========================\n");

      typedef float floatType;
      cudaDataType_t typeData = CUDA_C_32F;
      /******************
       * cuTensorNet
       *******************/

      cudaStream_t stream;
      HANDLE_CUDA_ERROR(cudaStreamCreate(&stream));

      cutensornetHandle_t handle;
      HANDLE_ERROR(cutensornetCreate(&handle));

      /**************************
       * Create tensor descriptors
       ***************************/
      std::vector<int32_t> modesT{'j', 'i'};
      std::vector<int32_t> modesU{'s', 'i'};
      std::vector<int32_t> modesV{'j', 's'};

      std::vector<int64_t> extentT{(int64_t)Tin.shape()[1], (int64_t)Tin.shape()[0]};
      std::vector<int64_t> extentU{(int64_t)U.shape()[1], (int64_t)U.shape()[0]};
      std::vector<int64_t> extentV{(int64_t)vT.shape()[1], (int64_t)vT.shape()[0]};

      const int32_t numModesIn = modesT.size();
      const int32_t numModesU = modesU.size();
      const int32_t numModesV = modesV.size();

      void *D_T = Tin._impl->storage()._impl->Mem;
      void *D_U = U._impl->storage()._impl->Mem;
      void *D_S = S._impl->storage()._impl->Mem;
      void *D_V = vT._impl->storage()._impl->Mem;

      cutensornetTensorDescriptor_t descTensorIn;
      cutensornetTensorDescriptor_t descTensorU;
      cutensornetTensorDescriptor_t descTensorV;

      const int64_t *strides = NULL;  // assuming fortran layout for all tensors

      HANDLE_ERROR(cutensornetCreateTensorDescriptor(handle, numModesIn, extentT.data(), strides,
                                                     modesT.data(), typeData, &descTensorIn));
      HANDLE_ERROR(cutensornetCreateTensorDescriptor(handle, numModesU, extentU.data(), strides,
                                                     modesU.data(), typeData, &descTensorU));
      HANDLE_ERROR(cutensornetCreateTensorDescriptor(handle, numModesV, extentV.data(), strides,
                                                     modesV.data(), typeData, &descTensorV));

      // printf("Initialize the cuTensorNet library and create all tensor descriptors.\n");

      /**********************************************
       * Setup SVD algorithm and truncation parameters
       ***********************************************/

      cutensornetTensorSVDConfig_t svdConfig;
      HANDLE_ERROR(cutensornetCreateTensorSVDConfig(handle, &svdConfig));
      double absCutoff;
      // set up truncation parameters
      if (return_err) {
        // do manually truncation instead, so no cuquantum truncate here
        absCutoff = 0;
      } else {
        absCutoff = err;
      }

      HANDLE_ERROR(cutensornetTensorSVDConfigSetAttribute(handle, svdConfig,
                                                          CUTENSORNET_TENSOR_SVD_CONFIG_ABS_CUTOFF,
                                                          &absCutoff, sizeof(absCutoff)));
      double relCutoff = 0;
      HANDLE_ERROR(cutensornetTensorSVDConfigSetAttribute(handle, svdConfig,
                                                          CUTENSORNET_TENSOR_SVD_CONFIG_REL_CUTOFF,
                                                          &relCutoff, sizeof(relCutoff)));

      // optional: choose gesvdj algorithm with customized parameters. Default is gesvd.
      cutensornetTensorSVDAlgo_t svdAlgo = CUTENSORNET_TENSOR_SVD_ALGO_GESVDJ;
      HANDLE_ERROR(cutensornetTensorSVDConfigSetAttribute(
        handle, svdConfig, CUTENSORNET_TENSOR_SVD_CONFIG_ALGO, &svdAlgo, sizeof(svdAlgo)));
      cutensornetGesvdjParams_t gesvdjParams{/*tol=*/0, /*maxSweeps=*/80};
      HANDLE_ERROR(cutensornetTensorSVDConfigSetAttribute(handle, svdConfig,
                                                          CUTENSORNET_TENSOR_SVD_CONFIG_ALGO_PARAMS,
                                                          &gesvdjParams, sizeof(gesvdjParams)));
      // printf("Set up SVDConfig to use GESVDJ algorithm with truncation\n");

      /********************************************************
       * Create SVDInfo to record runtime SVD truncation details
       *********************************************************/

      cutensornetTensorSVDInfo_t svdInfo;
      HANDLE_ERROR(cutensornetCreateTensorSVDInfo(handle, &svdInfo));

      // Sphinx: #6
      /**************************************************************
       * Query the required workspace sizes and allocate memory
       **************************************************************/

      cutensornetWorkspaceDescriptor_t workDesc;
      HANDLE_ERROR(cutensornetCreateWorkspaceDescriptor(handle, &workDesc));
      HANDLE_ERROR(cutensornetWorkspaceComputeSVDSizes(handle, descTensorIn, descTensorU,
                                                       descTensorV, svdConfig, workDesc));
      int64_t hostWorkspaceSize, deviceWorkspaceSize;
      // for tensor SVD, it does not matter which cutensornetWorksizePref_t we pick
      HANDLE_ERROR(cutensornetWorkspaceGetMemorySize(
        handle, workDesc, CUTENSORNET_WORKSIZE_PREF_RECOMMENDED, CUTENSORNET_MEMSPACE_DEVICE,
        CUTENSORNET_WORKSPACE_SCRATCH, &deviceWorkspaceSize));
      HANDLE_ERROR(cutensornetWorkspaceGetMemorySize(
        handle, workDesc, CUTENSORNET_WORKSIZE_PREF_RECOMMENDED, CUTENSORNET_MEMSPACE_HOST,
        CUTENSORNET_WORKSPACE_SCRATCH, &hostWorkspaceSize));

      void *devWork = nullptr, *hostWork = nullptr;
      if (deviceWorkspaceSize > 0) {
        HANDLE_CUDA_ERROR(cudaMalloc(&devWork, deviceWorkspaceSize));
      }
      if (hostWorkspaceSize > 0) {
        hostWork = malloc(hostWorkspaceSize);
      }
      HANDLE_ERROR(cutensornetWorkspaceSetMemory(handle, workDesc, CUTENSORNET_MEMSPACE_DEVICE,
                                                 CUTENSORNET_WORKSPACE_SCRATCH, devWork,
                                                 deviceWorkspaceSize));
      HANDLE_ERROR(cutensornetWorkspaceSetMemory(handle, workDesc, CUTENSORNET_MEMSPACE_HOST,
                                                 CUTENSORNET_WORKSPACE_SCRATCH, hostWork,
                                                 hostWorkspaceSize));

      // Sphinx: #7
      /**********
       * Execution
       ***********/

      GPUTimer timer{stream};
      double minTimeCUTENSOR = 1e100;
      const int numRuns = 1;  // to get stable perf results
      for (int i = 0; i < numRuns; ++i) {
        // // restore output
        // cudaMemsetAsync(D_U, 0, sizeU, stream);
        // cudaMemsetAsync(D_S, 0, sizeS, stream);
        // cudaMemsetAsync(D_V, 0, sizeV, stream);
        // cudaDeviceSynchronize();

        // With value-based truncation, `cutensornetTensorSVD` can potentially update the shared
        // extent in descTensorU/V. We here restore descTensorU/V to the original problem.
        HANDLE_ERROR(cutensornetDestroyTensorDescriptor(descTensorU));
        HANDLE_ERROR(cutensornetDestroyTensorDescriptor(descTensorV));
        HANDLE_ERROR(cutensornetCreateTensorDescriptor(handle, numModesU, extentU.data(), strides,
                                                       modesU.data(), typeData, &descTensorU));
        HANDLE_ERROR(cutensornetCreateTensorDescriptor(handle, numModesV, extentV.data(), strides,
                                                       modesV.data(), typeData, &descTensorV));

        timer.start();
        HANDLE_ERROR(cutensornetTensorSVD(handle, descTensorIn, D_T, descTensorU, D_U, D_S,
                                          descTensorV, D_V, svdConfig, svdInfo, workDesc, stream));
        // Synchronize and measure timing
        auto time = timer.seconds();
        minTimeCUTENSOR = (minTimeCUTENSOR < time) ? minTimeCUTENSOR : time;
      }

      // printf("Performing SVD\n");

      // HANDLE_CUDA_ERROR( cudaMemcpyAsync(U, D_U, sizeU, cudaMemcpyDeviceToHost) );
      // HANDLE_CUDA_ERROR( cudaMemcpyAsync(S, D_S, sizeS, cudaMemcpyDeviceToHost) );
      // HANDLE_CUDA_ERROR( cudaMemcpyAsync(V, D_V, sizeV, cudaMemcpyDeviceToHost) );

      // Sphinx: #8
      /*************************************
       * Query runtime truncation information
       **************************************/

      double discardedWeight{0};
      int64_t reducedExtent{0};
      cutensornetGesvdjStatus_t gesvdjStatus;
      cudaDeviceSynchronize();  // device synchronization.
      HANDLE_ERROR(cutensornetTensorSVDInfoGetAttribute(
        handle, svdInfo, CUTENSORNET_TENSOR_SVD_INFO_DISCARDED_WEIGHT, &discardedWeight,
        sizeof(discardedWeight)));
      HANDLE_ERROR(cutensornetTensorSVDInfoGetAttribute(handle, svdInfo,
                                                        CUTENSORNET_TENSOR_SVD_INFO_REDUCED_EXTENT,
                                                        &reducedExtent, sizeof(reducedExtent)));
      HANDLE_ERROR(cutensornetTensorSVDInfoGetAttribute(handle, svdInfo,
                                                        CUTENSORNET_TENSOR_SVD_INFO_ALGO_STATUS,
                                                        &gesvdjStatus, sizeof(gesvdjStatus)));

      printf("elapsed time: %.2f ms\n", minTimeCUTENSOR * 1000.f);
      printf("GESVDJ residual: %.4f, runtime sweeps = %d\n", gesvdjStatus.residual,
             gesvdjStatus.sweeps);
      printf("reduced extent found at runtime: %lu\n", reducedExtent);
      std::cout << "discarded weight: " << discardedWeight << std::endl;
      // printf("discarded weight: %.15f\n", discardedWeight);

      // Sphinx: #9
      /***************
       * Free resources
       ****************/

      HANDLE_ERROR(cutensornetDestroyTensorDescriptor(descTensorIn));
      HANDLE_ERROR(cutensornetDestroyTensorDescriptor(descTensorU));
      HANDLE_ERROR(cutensornetDestroyTensorDescriptor(descTensorV));
      HANDLE_ERROR(cutensornetDestroyTensorSVDConfig(svdConfig));
      HANDLE_ERROR(cutensornetDestroyTensorSVDInfo(svdInfo));
      HANDLE_ERROR(cutensornetDestroyWorkspaceDescriptor(workDesc));
      HANDLE_ERROR(cutensornetDestroy(handle));

      if (devWork) cudaFree(devWork);
      if (hostWork) free(hostWork);
      // printf("Free resource and exit.\n");

      // Manually truncation
      cytnx_uint64 Kdim = keepdim;
      cytnx_uint64 nums = S.storage().size();
      if (nums < keepdim) {
        Kdim = nums;
      }
      cytnx_uint64 truc_dim = Kdim;
      for (cytnx_int64 i = Kdim - 1; i >= 0; i--) {
        if (((cytnx_double *)S._impl->storage()._impl->Mem)[i] < err) {
          truc_dim--;
        } else {
          break;
        }
      }
      if (truc_dim == 0) {
        truc_dim = 1;
      }
      if (truc_dim != nums) {
        memcpy_truncation_cf(U, vT, S, terr, truc_dim, true, true, return_err);
      }
    }

    void cuQuantumGeSvd_internal_d(const Tensor &Tin, const cytnx_uint64 &keepdim,
                                   const double &err, const unsigned int &return_err, Tensor &U,
                                   Tensor &S, Tensor &vT, Tensor &terr) {
      const size_t cuTensornetVersion = cutensornetGetVersion();
      // printf("cuTensorNet-vers:%ld\n", cuTensornetVersion);

      cudaDeviceProp prop;
      int deviceId = Tin.device();
      HANDLE_CUDA_ERROR(cudaSetDevice(deviceId));
      HANDLE_CUDA_ERROR(cudaGetDevice(&deviceId));
      HANDLE_CUDA_ERROR(cudaGetDeviceProperties(&prop, deviceId));

      // printf("===== device info ======\n");
      // printf("GPU-name:%s\n", prop.name);
      // printf("GPU-clock:%d\n", prop.clockRate);
      // printf("GPU-memoryClock:%d\n", prop.memoryClockRate);
      // printf("GPU-nSM:%d\n", prop.multiProcessorCount);
      // printf("GPU-major:%d\n", prop.major);
      // printf("GPU-minor:%d\n", prop.minor);
      // printf("========================\n");

      typedef float floatType;
      cudaDataType_t typeData = CUDA_R_64F;
      /******************
       * cuTensorNet
       *******************/

      cudaStream_t stream;
      HANDLE_CUDA_ERROR(cudaStreamCreate(&stream));

      cutensornetHandle_t handle;
      HANDLE_ERROR(cutensornetCreate(&handle));

      /**************************
       * Create tensor descriptors
       ***************************/
      std::vector<int32_t> modesT{'j', 'i'};
      std::vector<int32_t> modesU{'s', 'i'};
      std::vector<int32_t> modesV{'j', 's'};

      std::vector<int64_t> extentT{(int64_t)Tin.shape()[1], (int64_t)Tin.shape()[0]};
      std::vector<int64_t> extentU{(int64_t)U.shape()[1], (int64_t)U.shape()[0]};
      std::vector<int64_t> extentV{(int64_t)vT.shape()[1], (int64_t)vT.shape()[0]};

      const int32_t numModesIn = modesT.size();
      const int32_t numModesU = modesU.size();
      const int32_t numModesV = modesV.size();

      void *D_T = Tin._impl->storage()._impl->Mem;
      void *D_U = U._impl->storage()._impl->Mem;
      void *D_S = S._impl->storage()._impl->Mem;
      void *D_V = vT._impl->storage()._impl->Mem;

      cutensornetTensorDescriptor_t descTensorIn;
      cutensornetTensorDescriptor_t descTensorU;
      cutensornetTensorDescriptor_t descTensorV;

      const int64_t *strides = NULL;  // assuming fortran layout for all tensors

      HANDLE_ERROR(cutensornetCreateTensorDescriptor(handle, numModesIn, extentT.data(), strides,
                                                     modesT.data(), typeData, &descTensorIn));
      HANDLE_ERROR(cutensornetCreateTensorDescriptor(handle, numModesU, extentU.data(), strides,
                                                     modesU.data(), typeData, &descTensorU));
      HANDLE_ERROR(cutensornetCreateTensorDescriptor(handle, numModesV, extentV.data(), strides,
                                                     modesV.data(), typeData, &descTensorV));

      // printf("Initialize the cuTensorNet library and create all tensor descriptors.\n");

      /**********************************************
       * Setup SVD algorithm and truncation parameters
       ***********************************************/

      cutensornetTensorSVDConfig_t svdConfig;
      HANDLE_ERROR(cutensornetCreateTensorSVDConfig(handle, &svdConfig));
      double absCutoff;
      // set up truncation parameters
      if (return_err) {
        // do manually truncation instead, so no cuquantum truncate here
        absCutoff = 0;
      } else {
        absCutoff = err;
      }

      HANDLE_ERROR(cutensornetTensorSVDConfigSetAttribute(handle, svdConfig,
                                                          CUTENSORNET_TENSOR_SVD_CONFIG_ABS_CUTOFF,
                                                          &absCutoff, sizeof(absCutoff)));
      double relCutoff = 0;
      HANDLE_ERROR(cutensornetTensorSVDConfigSetAttribute(handle, svdConfig,
                                                          CUTENSORNET_TENSOR_SVD_CONFIG_REL_CUTOFF,
                                                          &relCutoff, sizeof(relCutoff)));

      // optional: choose gesvdj algorithm with customized parameters. Default is gesvd.
      cutensornetTensorSVDAlgo_t svdAlgo = CUTENSORNET_TENSOR_SVD_ALGO_GESVDJ;
      HANDLE_ERROR(cutensornetTensorSVDConfigSetAttribute(
        handle, svdConfig, CUTENSORNET_TENSOR_SVD_CONFIG_ALGO, &svdAlgo, sizeof(svdAlgo)));
      cutensornetGesvdjParams_t gesvdjParams{/*tol=*/0, /*maxSweeps=*/80};
      HANDLE_ERROR(cutensornetTensorSVDConfigSetAttribute(handle, svdConfig,
                                                          CUTENSORNET_TENSOR_SVD_CONFIG_ALGO_PARAMS,
                                                          &gesvdjParams, sizeof(gesvdjParams)));
      // printf("Set up SVDConfig to use GESVDJ algorithm with truncation\n");

      /********************************************************
       * Create SVDInfo to record runtime SVD truncation details
       *********************************************************/

      cutensornetTensorSVDInfo_t svdInfo;
      HANDLE_ERROR(cutensornetCreateTensorSVDInfo(handle, &svdInfo));

      // Sphinx: #6
      /**************************************************************
       * Query the required workspace sizes and allocate memory
       **************************************************************/

      cutensornetWorkspaceDescriptor_t workDesc;
      HANDLE_ERROR(cutensornetCreateWorkspaceDescriptor(handle, &workDesc));
      HANDLE_ERROR(cutensornetWorkspaceComputeSVDSizes(handle, descTensorIn, descTensorU,
                                                       descTensorV, svdConfig, workDesc));
      int64_t hostWorkspaceSize, deviceWorkspaceSize;
      // for tensor SVD, it does not matter which cutensornetWorksizePref_t we pick
      HANDLE_ERROR(cutensornetWorkspaceGetMemorySize(
        handle, workDesc, CUTENSORNET_WORKSIZE_PREF_RECOMMENDED, CUTENSORNET_MEMSPACE_DEVICE,
        CUTENSORNET_WORKSPACE_SCRATCH, &deviceWorkspaceSize));
      HANDLE_ERROR(cutensornetWorkspaceGetMemorySize(
        handle, workDesc, CUTENSORNET_WORKSIZE_PREF_RECOMMENDED, CUTENSORNET_MEMSPACE_HOST,
        CUTENSORNET_WORKSPACE_SCRATCH, &hostWorkspaceSize));

      void *devWork = nullptr, *hostWork = nullptr;
      if (deviceWorkspaceSize > 0) {
        HANDLE_CUDA_ERROR(cudaMalloc(&devWork, deviceWorkspaceSize));
      }
      if (hostWorkspaceSize > 0) {
        hostWork = malloc(hostWorkspaceSize);
      }
      HANDLE_ERROR(cutensornetWorkspaceSetMemory(handle, workDesc, CUTENSORNET_MEMSPACE_DEVICE,
                                                 CUTENSORNET_WORKSPACE_SCRATCH, devWork,
                                                 deviceWorkspaceSize));
      HANDLE_ERROR(cutensornetWorkspaceSetMemory(handle, workDesc, CUTENSORNET_MEMSPACE_HOST,
                                                 CUTENSORNET_WORKSPACE_SCRATCH, hostWork,
                                                 hostWorkspaceSize));

      // Sphinx: #7
      /**********
       * Execution
       ***********/

      GPUTimer timer{stream};
      double minTimeCUTENSOR = 1e100;
      const int numRuns = 1;  // to get stable perf results
      for (int i = 0; i < numRuns; ++i) {
        // // restore output
        // cudaMemsetAsync(D_U, 0, sizeU, stream);
        // cudaMemsetAsync(D_S, 0, sizeS, stream);
        // cudaMemsetAsync(D_V, 0, sizeV, stream);
        // cudaDeviceSynchronize();

        // With value-based truncation, `cutensornetTensorSVD` can potentially update the shared
        // extent in descTensorU/V. We here restore descTensorU/V to the original problem.
        HANDLE_ERROR(cutensornetDestroyTensorDescriptor(descTensorU));
        HANDLE_ERROR(cutensornetDestroyTensorDescriptor(descTensorV));
        HANDLE_ERROR(cutensornetCreateTensorDescriptor(handle, numModesU, extentU.data(), strides,
                                                       modesU.data(), typeData, &descTensorU));
        HANDLE_ERROR(cutensornetCreateTensorDescriptor(handle, numModesV, extentV.data(), strides,
                                                       modesV.data(), typeData, &descTensorV));

        timer.start();
        HANDLE_ERROR(cutensornetTensorSVD(handle, descTensorIn, D_T, descTensorU, D_U, D_S,
                                          descTensorV, D_V, svdConfig, svdInfo, workDesc, stream));
        // Synchronize and measure timing
        auto time = timer.seconds();
        minTimeCUTENSOR = (minTimeCUTENSOR < time) ? minTimeCUTENSOR : time;
      }

      // printf("Performing SVD\n");

      // HANDLE_CUDA_ERROR( cudaMemcpyAsync(U, D_U, sizeU, cudaMemcpyDeviceToHost) );
      // HANDLE_CUDA_ERROR( cudaMemcpyAsync(S, D_S, sizeS, cudaMemcpyDeviceToHost) );
      // HANDLE_CUDA_ERROR( cudaMemcpyAsync(V, D_V, sizeV, cudaMemcpyDeviceToHost) );

      // Sphinx: #8
      /*************************************
       * Query runtime truncation information
       **************************************/

      double discardedWeight{0};
      int64_t reducedExtent{0};
      cutensornetGesvdjStatus_t gesvdjStatus;
      cudaDeviceSynchronize();  // device synchronization.
      HANDLE_ERROR(cutensornetTensorSVDInfoGetAttribute(
        handle, svdInfo, CUTENSORNET_TENSOR_SVD_INFO_DISCARDED_WEIGHT, &discardedWeight,
        sizeof(discardedWeight)));
      HANDLE_ERROR(cutensornetTensorSVDInfoGetAttribute(handle, svdInfo,
                                                        CUTENSORNET_TENSOR_SVD_INFO_REDUCED_EXTENT,
                                                        &reducedExtent, sizeof(reducedExtent)));
      HANDLE_ERROR(cutensornetTensorSVDInfoGetAttribute(handle, svdInfo,
                                                        CUTENSORNET_TENSOR_SVD_INFO_ALGO_STATUS,
                                                        &gesvdjStatus, sizeof(gesvdjStatus)));

      printf("elapsed time: %.2f ms\n", minTimeCUTENSOR * 1000.f);
      printf("GESVDJ residual: %.4f, runtime sweeps = %d\n", gesvdjStatus.residual,
             gesvdjStatus.sweeps);
      printf("reduced extent found at runtime: %lu\n", reducedExtent);
      std::cout << "discarded weight: " << discardedWeight << std::endl;
      // printf("discarded weight: %.15f\n", discardedWeight);

      // Sphinx: #9
      /***************
       * Free resources
       ****************/

      HANDLE_ERROR(cutensornetDestroyTensorDescriptor(descTensorIn));
      HANDLE_ERROR(cutensornetDestroyTensorDescriptor(descTensorU));
      HANDLE_ERROR(cutensornetDestroyTensorDescriptor(descTensorV));
      HANDLE_ERROR(cutensornetDestroyTensorSVDConfig(svdConfig));
      HANDLE_ERROR(cutensornetDestroyTensorSVDInfo(svdInfo));
      HANDLE_ERROR(cutensornetDestroyWorkspaceDescriptor(workDesc));
      HANDLE_ERROR(cutensornetDestroy(handle));

      if (devWork) cudaFree(devWork);
      if (hostWork) free(hostWork);
      // printf("Free resource and exit.\n");

      // Manually truncation
      cytnx_uint64 Kdim = keepdim;
      cytnx_uint64 nums = S.storage().size();
      if (nums < keepdim) {
        Kdim = nums;
      }
      cytnx_uint64 truc_dim = Kdim;
      for (cytnx_int64 i = Kdim - 1; i >= 0; i--) {
        if (((cytnx_double *)S._impl->storage()._impl->Mem)[i] < err) {
          truc_dim--;
        } else {
          break;
        }
      }
      if (truc_dim == 0) {
        truc_dim = 1;
      }
      if (truc_dim != nums) {
        memcpy_truncation_d(U, vT, S, terr, truc_dim, true, true, return_err);
      }
    }

    void cuQuantumGeSvd_internal_f(const Tensor &Tin, const cytnx_uint64 &keepdim,
                                   const double &err, const unsigned int &return_err, Tensor &U,
                                   Tensor &S, Tensor &vT, Tensor &terr) {
      const size_t cuTensornetVersion = cutensornetGetVersion();
      // printf("cuTensorNet-vers:%ld\n", cuTensornetVersion);

      cudaDeviceProp prop;
      int deviceId = Tin.device();
      HANDLE_CUDA_ERROR(cudaSetDevice(deviceId));
      HANDLE_CUDA_ERROR(cudaGetDevice(&deviceId));
      HANDLE_CUDA_ERROR(cudaGetDeviceProperties(&prop, deviceId));

      // printf("===== device info ======\n");
      // printf("GPU-name:%s\n", prop.name);
      // printf("GPU-clock:%d\n", prop.clockRate);
      // printf("GPU-memoryClock:%d\n", prop.memoryClockRate);
      // printf("GPU-nSM:%d\n", prop.multiProcessorCount);
      // printf("GPU-major:%d\n", prop.major);
      // printf("GPU-minor:%d\n", prop.minor);
      // printf("========================\n");

      typedef float floatType;
      cudaDataType_t typeData = CUDA_R_32F;
      /******************
       * cuTensorNet
       *******************/

      cudaStream_t stream;
      HANDLE_CUDA_ERROR(cudaStreamCreate(&stream));

      cutensornetHandle_t handle;
      HANDLE_ERROR(cutensornetCreate(&handle));

      /**************************
       * Create tensor descriptors
       ***************************/
      std::vector<int32_t> modesT{'j', 'i'};
      std::vector<int32_t> modesU{'s', 'i'};
      std::vector<int32_t> modesV{'j', 's'};

      std::vector<int64_t> extentT{(int64_t)Tin.shape()[1], (int64_t)Tin.shape()[0]};
      std::vector<int64_t> extentU{(int64_t)U.shape()[1], (int64_t)U.shape()[0]};
      std::vector<int64_t> extentV{(int64_t)vT.shape()[1], (int64_t)vT.shape()[0]};

      const int32_t numModesIn = modesT.size();
      const int32_t numModesU = modesU.size();
      const int32_t numModesV = modesV.size();

      void *D_T = Tin._impl->storage()._impl->Mem;
      void *D_U = U._impl->storage()._impl->Mem;
      void *D_S = S._impl->storage()._impl->Mem;
      void *D_V = vT._impl->storage()._impl->Mem;

      cutensornetTensorDescriptor_t descTensorIn;
      cutensornetTensorDescriptor_t descTensorU;
      cutensornetTensorDescriptor_t descTensorV;

      const int64_t *strides = NULL;  // assuming fortran layout for all tensors

      HANDLE_ERROR(cutensornetCreateTensorDescriptor(handle, numModesIn, extentT.data(), strides,
                                                     modesT.data(), typeData, &descTensorIn));
      HANDLE_ERROR(cutensornetCreateTensorDescriptor(handle, numModesU, extentU.data(), strides,
                                                     modesU.data(), typeData, &descTensorU));
      HANDLE_ERROR(cutensornetCreateTensorDescriptor(handle, numModesV, extentV.data(), strides,
                                                     modesV.data(), typeData, &descTensorV));

      // printf("Initialize the cuTensorNet library and create all tensor descriptors.\n");

      /**********************************************
       * Setup SVD algorithm and truncation parameters
       ***********************************************/

      cutensornetTensorSVDConfig_t svdConfig;
      HANDLE_ERROR(cutensornetCreateTensorSVDConfig(handle, &svdConfig));
      double absCutoff;
      // set up truncation parameters
      if (return_err) {
        // do manually truncation instead, so no cuquantum truncate here
        absCutoff = 0;
      } else {
        absCutoff = err;
      }

      HANDLE_ERROR(cutensornetTensorSVDConfigSetAttribute(handle, svdConfig,
                                                          CUTENSORNET_TENSOR_SVD_CONFIG_ABS_CUTOFF,
                                                          &absCutoff, sizeof(absCutoff)));
      double relCutoff = 0;
      HANDLE_ERROR(cutensornetTensorSVDConfigSetAttribute(handle, svdConfig,
                                                          CUTENSORNET_TENSOR_SVD_CONFIG_REL_CUTOFF,
                                                          &relCutoff, sizeof(relCutoff)));

      // optional: choose gesvdj algorithm with customized parameters. Default is gesvd.
      cutensornetTensorSVDAlgo_t svdAlgo = CUTENSORNET_TENSOR_SVD_ALGO_GESVDJ;
      HANDLE_ERROR(cutensornetTensorSVDConfigSetAttribute(
        handle, svdConfig, CUTENSORNET_TENSOR_SVD_CONFIG_ALGO, &svdAlgo, sizeof(svdAlgo)));
      cutensornetGesvdjParams_t gesvdjParams{/*tol=*/0, /*maxSweeps=*/80};
      HANDLE_ERROR(cutensornetTensorSVDConfigSetAttribute(handle, svdConfig,
                                                          CUTENSORNET_TENSOR_SVD_CONFIG_ALGO_PARAMS,
                                                          &gesvdjParams, sizeof(gesvdjParams)));
      // printf("Set up SVDConfig to use GESVDJ algorithm with truncation\n");

      /********************************************************
       * Create SVDInfo to record runtime SVD truncation details
       *********************************************************/

      cutensornetTensorSVDInfo_t svdInfo;
      HANDLE_ERROR(cutensornetCreateTensorSVDInfo(handle, &svdInfo));

      // Sphinx: #6
      /**************************************************************
       * Query the required workspace sizes and allocate memory
       **************************************************************/

      cutensornetWorkspaceDescriptor_t workDesc;
      HANDLE_ERROR(cutensornetCreateWorkspaceDescriptor(handle, &workDesc));
      HANDLE_ERROR(cutensornetWorkspaceComputeSVDSizes(handle, descTensorIn, descTensorU,
                                                       descTensorV, svdConfig, workDesc));
      int64_t hostWorkspaceSize, deviceWorkspaceSize;
      // for tensor SVD, it does not matter which cutensornetWorksizePref_t we pick
      HANDLE_ERROR(cutensornetWorkspaceGetMemorySize(
        handle, workDesc, CUTENSORNET_WORKSIZE_PREF_RECOMMENDED, CUTENSORNET_MEMSPACE_DEVICE,
        CUTENSORNET_WORKSPACE_SCRATCH, &deviceWorkspaceSize));
      HANDLE_ERROR(cutensornetWorkspaceGetMemorySize(
        handle, workDesc, CUTENSORNET_WORKSIZE_PREF_RECOMMENDED, CUTENSORNET_MEMSPACE_HOST,
        CUTENSORNET_WORKSPACE_SCRATCH, &hostWorkspaceSize));

      void *devWork = nullptr, *hostWork = nullptr;
      if (deviceWorkspaceSize > 0) {
        HANDLE_CUDA_ERROR(cudaMalloc(&devWork, deviceWorkspaceSize));
      }
      if (hostWorkspaceSize > 0) {
        hostWork = malloc(hostWorkspaceSize);
      }
      HANDLE_ERROR(cutensornetWorkspaceSetMemory(handle, workDesc, CUTENSORNET_MEMSPACE_DEVICE,
                                                 CUTENSORNET_WORKSPACE_SCRATCH, devWork,
                                                 deviceWorkspaceSize));
      HANDLE_ERROR(cutensornetWorkspaceSetMemory(handle, workDesc, CUTENSORNET_MEMSPACE_HOST,
                                                 CUTENSORNET_WORKSPACE_SCRATCH, hostWork,
                                                 hostWorkspaceSize));

      // Sphinx: #7
      /**********
       * Execution
       ***********/

      GPUTimer timer{stream};
      double minTimeCUTENSOR = 1e100;
      const int numRuns = 1;  // to get stable perf results
      for (int i = 0; i < numRuns; ++i) {
        // // restore output
        // cudaMemsetAsync(D_U, 0, sizeU, stream);
        // cudaMemsetAsync(D_S, 0, sizeS, stream);
        // cudaMemsetAsync(D_V, 0, sizeV, stream);
        // cudaDeviceSynchronize();

        // With value-based truncation, `cutensornetTensorSVD` can potentially update the shared
        // extent in descTensorU/V. We here restore descTensorU/V to the original problem.
        HANDLE_ERROR(cutensornetDestroyTensorDescriptor(descTensorU));
        HANDLE_ERROR(cutensornetDestroyTensorDescriptor(descTensorV));
        HANDLE_ERROR(cutensornetCreateTensorDescriptor(handle, numModesU, extentU.data(), strides,
                                                       modesU.data(), typeData, &descTensorU));
        HANDLE_ERROR(cutensornetCreateTensorDescriptor(handle, numModesV, extentV.data(), strides,
                                                       modesV.data(), typeData, &descTensorV));

        timer.start();
        HANDLE_ERROR(cutensornetTensorSVD(handle, descTensorIn, D_T, descTensorU, D_U, D_S,
                                          descTensorV, D_V, svdConfig, svdInfo, workDesc, stream));
        // Synchronize and measure timing
        auto time = timer.seconds();
        minTimeCUTENSOR = (minTimeCUTENSOR < time) ? minTimeCUTENSOR : time;
      }

      // printf("Performing SVD\n");

      // HANDLE_CUDA_ERROR( cudaMemcpyAsync(U, D_U, sizeU, cudaMemcpyDeviceToHost) );
      // HANDLE_CUDA_ERROR( cudaMemcpyAsync(S, D_S, sizeS, cudaMemcpyDeviceToHost) );
      // HANDLE_CUDA_ERROR( cudaMemcpyAsync(V, D_V, sizeV, cudaMemcpyDeviceToHost) );

      // Sphinx: #8
      /*************************************
       * Query runtime truncation information
       **************************************/

      double discardedWeight{0};
      int64_t reducedExtent{0};
      cutensornetGesvdjStatus_t gesvdjStatus;
      cudaDeviceSynchronize();  // device synchronization.
      HANDLE_ERROR(cutensornetTensorSVDInfoGetAttribute(
        handle, svdInfo, CUTENSORNET_TENSOR_SVD_INFO_DISCARDED_WEIGHT, &discardedWeight,
        sizeof(discardedWeight)));
      HANDLE_ERROR(cutensornetTensorSVDInfoGetAttribute(handle, svdInfo,
                                                        CUTENSORNET_TENSOR_SVD_INFO_REDUCED_EXTENT,
                                                        &reducedExtent, sizeof(reducedExtent)));
      HANDLE_ERROR(cutensornetTensorSVDInfoGetAttribute(handle, svdInfo,
                                                        CUTENSORNET_TENSOR_SVD_INFO_ALGO_STATUS,
                                                        &gesvdjStatus, sizeof(gesvdjStatus)));

      printf("elapsed time: %.2f ms\n", minTimeCUTENSOR * 1000.f);
      printf("GESVDJ residual: %.4f, runtime sweeps = %d\n", gesvdjStatus.residual,
             gesvdjStatus.sweeps);
      printf("reduced extent found at runtime: %lu\n", reducedExtent);
      std::cout << "discarded weight: " << discardedWeight << std::endl;
      // printf("discarded weight: %.15f\n", discardedWeight);

      // Sphinx: #9
      /***************
       * Free resources
       ****************/

      HANDLE_ERROR(cutensornetDestroyTensorDescriptor(descTensorIn));
      HANDLE_ERROR(cutensornetDestroyTensorDescriptor(descTensorU));
      HANDLE_ERROR(cutensornetDestroyTensorDescriptor(descTensorV));
      HANDLE_ERROR(cutensornetDestroyTensorSVDConfig(svdConfig));
      HANDLE_ERROR(cutensornetDestroyTensorSVDInfo(svdInfo));
      HANDLE_ERROR(cutensornetDestroyWorkspaceDescriptor(workDesc));
      HANDLE_ERROR(cutensornetDestroy(handle));

      if (devWork) cudaFree(devWork);
      if (hostWork) free(hostWork);
      // printf("Free resource and exit.\n");
    }

  #endif
#endif
  }  // namespace linalg_internal
}  // namespace cytnx
